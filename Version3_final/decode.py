import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torchtext
import spacy
import string
# import gensim
import pickle
import random
from torchtext.data import Field, Dataset, Example
import numpy as np
import math
from torch.autograd import Variable
import time
# from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import csv
import matplotlib.pyplot as plt
from queue import PriorityQueue
import dill
from torch.nn.parallel import DistributedDataParallel as DDP
import queue
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#torch.cuda.empty_cache()
#print("Cache Emptied!\n")

##local files
import config
import models

def load_data():
    train_data = torch.load(config.train_data_path_final,pickle_module=dill)
    val_data = torch.load(config.eval_data_path_final,pickle_module=dill)
    test_data = torch.load(config.test_data_path_final,pickle_module=dill)
    return train_data,val_data,test_data

print("Loading data")
train_examples, val_examples, test_examples = load_data()
train_examples = train_examples[0:50000]
val_examples = val_examples[0:9000]
print(len(train_examples))
print("Length of eval:",len(val_examples))
print("Length of test:",len(test_examples))
print("Data is loaded")

##Making the vocab
en = spacy.load('en_core_web_sm')
def tokenize_en(text):
  ret = [tok.text for tok in en.tokenizer(str(text))]
  return ret
TEXT_en = torchtext.data.Field(
  tokenize    = tokenize_en,
  lower       = True,
  init_token  = '<sos>',
  eos_token   = '<eos>'
)
fields = [('src',TEXT_en),('trg',TEXT_en)]
train_dataset = torchtext.data.Dataset(train_examples,fields)
val_dataset = torchtext.data.Dataset(val_examples,fields)
test_dataset = torchtext.data.Dataset(test_examples,fields)
BATCH_SIZE = 1
print(BATCH_SIZE)
train_iterator = torchtext.data.BucketIterator(
    train_dataset,BATCH_SIZE,sort_within_batch=False,device=device,repeat=False,sort_key=False
)
val_iterator = torchtext.data.BucketIterator(
    val_dataset,BATCH_SIZE,sort_within_batch=False,device=device,repeat=False,sort_key=False
)
test_iterator = torchtext.data.BucketIterator(
    test_dataset,BATCH_SIZE,sort_within_batch=False,device=device,repeat=False,sort_key=False
)
print("iterators are made and the vocab is being built")
#TEXT_en.build_vocab(train_dataset.src,train_dataset.trg,val_dataset.src,val_dataset.trg,min_freq=27)
TEXT_en.build_vocab(train_dataset.src,train_dataset.trg,val_dataset.src,val_dataset.trg,min_freq=15)
vocab = TEXT_en.vocab
TEXT_en.build_vocab(train_dataset.src,train_dataset.trg,val_dataset.src,val_dataset.trg)
vocab_ext = TEXT_en.vocab
print("Size of encoder vocab is:",len(vocab))
print("Size of extended vocab is:",len(vocab_ext))

INPUT_DIM = len(vocab)
OUTPUT_DIM = len(vocab)
ENC_EMB_DIM = config.ENC_EMB_DIM
DEC_EMB_DIM = config.DEC_EMB_DIM
ENC_HID_DIM = config.ENC_HID_DIM
DEC_HID_DIM = config.DEC_HID_DIM
N_LAYERS = config.N_LAYERS
ENC_DROPOUT = config.ENC_DROPOUT
DEC_DROPOUT = config.DEC_DROPOUT

encoder = models.encoderRNN(INPUT_DIM,ENC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT)
#encoder = nn.DataParallel(encoder).cuda()
encoder.cuda()
decoder = models.decoderRNN(OUTPUT_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,DEC_DROPOUT)
#decoder = nn.DataParallel(decoder).cuda()
decoder.cuda()

model = models.Seq2Seq(encoder,decoder,load=True)

##Define the optimizer and all
#params= list(model.encoder.module.parameters())+list(model.decoder.module.parameters())
params = list(model.encoder.parameters())+list(model.decoder.parameters())
optimizer = optim.Adagrad(params,lr=0.15,initial_accumulator_value=0.1)
TRG_PAD_IDX = vocab.stoi['<pad>']

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_probs, state, context, coverage):
        return Beam(
            tokens = self.tokens + [token],
            log_probs = self.log_probs + log_probs,
            state = state,
            context = context,
            coverage = coverage
        )

    def avg_log_prob(self):
        return (self.log_probs)/len(self.tokens)

    def latest_token(self):
        return self.tokens[-1]

class BeamSearch(object):
    def __init__(self, vocab, vocab_ext, iterator, model):
        self.vocab = vocab
        self.iterator = iterator
        self.model = model
        self.vocab_ext = vocab_ext

    def sort_beams(self, beams):
        return sorted(beams, key = lambda h: h.avg_log_prob(), reverse=True)

    def print_sent(self, sent):
        print('src: ',sent['src'])
        print('\nacutal: ',sent['actual'])
        print('\npredicted: ',sent['predicted'])
        print("\n")

    def src_to_ids(self,src_words, vocab):
      extend_ids = []
      ids = []
      oov = []
      for word in src_words:
        word_id = vocab.stoi[word]
        if word_id == vocab.stoi['<unk>']:
          if word not in oov:
            oov.append(word)
          index = len(vocab) + oov.index(word)
          extend_ids.append(index)
        else:
          extend_ids.append(vocab.stoi[word])
        ids.append(vocab.stoi[word])
      return torch.LongTensor(ids).cuda(), torch.LongTensor(extend_ids).cuda(), oov


    def trg_to_ids(self,trg_words, vocab, oov):
      extend_ids = []
      ids = []
      for word in trg_words:
        word_id = vocab.stoi[word]
        if word_id == vocab.stoi['<unk>']:
          if word in oov:
            word_index = len(vocab) + oov.index(word)
            extend_ids.append(word_index)
          else:
            extend_ids.append(vocab.stoi['<unk>'])
        else:
          extend_ids.append(word_id)
        ids.append(word_id)
      return torch.LongTensor(ids).cuda(), torch.LongTensor(extend_ids).cuda()

    def expand_vocab(self,batch,vocab_ext,vocab):
      new_src_extend_vocab = Variable(torch.zeros((config.BEAM_SIZE,batch.src.shape[0]),dtype=torch.int64)).cuda()
      new_src_original_vocab = Variable(torch.zeros((config.BEAM_SIZE,batch.src.shape[0]),dtype=torch.int64)).cuda()
      new_trg_extend_vocab = Variable(torch.zeros((config.BEAM_SIZE,batch.trg.shape[0]),dtype=torch.int64)).cuda()
      new_trg_original_vocab = Variable(torch.zeros((config.BEAM_SIZE,batch.trg.shape[0]),dtype=torch.int64)).cuda()
      new_src_extend_vocab = new_src_extend_vocab.permute(1,0)
      new_src_original_vocab = new_src_original_vocab.permute(1,0)
      new_trg_extend_vocab = new_trg_extend_vocab.permute(1,0)
      new_trg_original_vocab = new_trg_original_vocab.permute(1,0)
      batch_oov = []
      max_len = 0
      for i in range(batch.src.shape[1]):
        src_sent = " ".join([vocab_ext.itos[j] for j in list(batch.src[:,i])])
        trg_sent = " ".join([vocab_ext.itos[j] for j in list(batch.trg[:,i])])
        new_src_original_vocab[:,i], new_src_extend_vocab[:,i], oov = self.src_to_ids(src_sent.split(), vocab)
        new_trg_original_vocab[:,i] ,new_trg_extend_vocab[:,i] = self.trg_to_ids(trg_sent.split(), vocab, oov)
        batch_oov = oov
        max_len = max(max_len, len(oov))
      extra_zeros = None
      if max_len >0:
        extra_zeros = Variable(torch.zeros((config.BEAM_SIZE, max_len))).cuda()
      return new_src_original_vocab[:config.MAX_ENC_LEN,:], new_trg_original_vocab, new_src_extend_vocab[:config.MAX_ENC_LEN,:], new_trg_extend_vocab, batch_oov, extra_zeros

    def decode(self):
        start = time.time()
        for i, batch in enumerate(self.iterator):
            if i>2:
                break
            with torch.no_grad():
                best_summary,article_oov = self.beam_search(batch)

            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = []
            for i in output_ids:
                if i<len(self.vocab):
                    decoded_words.append(self.vocab.itos[i])
                else:
                    decoded_words.append(article_oov[i-len(self.vocab)])

            sent = {'src':'','actual':'','predicted':''}
            sent['predicted'] = " ".join(decoded_words)
            src = batch.src
            trg = batch.trg
            sent['src']=" ".join([self.vocab_ext.itos[t] for t in src])
            sent['src'] = sent['src'].replace('<sos>',"").replace("<eos>","").replace('<pad>',"").strip()

            sent['actual']=" ".join([self.vocab_ext.itos[t] for t in trg])
            sent['actual'] = sent['actual'].replace('<sos>',"").replace("<eos>","").replace('<pad>',"").strip()
            self.print_sent(sent)
            """Write the function for checking rogue score and take average"""
        end = time.time()
        """Code for printing the time taken for decoding the entire batch"""

    def beam_search(self,batch):
        self.model.encoder.eval()
        self.model.decoder.eval()

        src, trg, src_extend_vocab, trg_extend_vocab, article_oovs, extra_zeros = self.expand_vocab(batch, self.vocab_ext, self.vocab)
        src = src.permute(1,0)
        enc_padding_mask = ~(src==TRG_PAD_IDX)
        dec_padding_mask = ~(trg==TRG_PAD_IDX)
        enc_lens = torch.sum(enc_padding_mask,dim=1)
        dec_len = trg.shape[0]
        trg_for_check = trg_extend_vocab

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(src,enc_lens)

        h_0, c_0 = encoder_hidden
        h_0 = h_0.squeeze()
        c_0 = c_0.squeeze()

        coverage_0 = None
        if config.IS_COVERAGE:
            coverage_0 = Variable(torch.zeros(src.shape[1])).cuda()


        beams = [Beam(
            tokens=[self.vocab.stoi['<sos>']],
            log_probs=0.0,
            state=(h_0[0],c_0[0]),
            context=Variable(torch.zeros((BATCH_SIZE,2* config.DEC_HID_DIM))).cuda(),
            coverage=(coverage_0 if config.IS_COVERAGE else None)) for _ in range(config.BEAM_SIZE)]

        results = []
        steps = 0

        while steps<config.MAX_DEC_LEN+20 and len(results) < config.BEAM_SIZE and steps<dec_len:
            latest_tokens = [h.latest_token() for h in beams]
            latest_tokens = [t if t<len(self.vocab) else self.vocab.stoi['<unk>'] for t in latest_tokens]
            input = Variable(torch.LongTensor(latest_tokens)).cuda()

            ## Teacher forcing
            force = random.random()
            if force>1.0:
                input = torch.full((config.BEAM_SIZE,1), int(trg[steps,0]), dtype=torch.int64).cuda()
                input = input.squeeze()

            all_state_h = []
            all_state_c = []
            all_state_context = []
            for i in beams:
                state_h, state_c = i.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_state_context.append(i.context[0])

            hidden = (torch.stack(all_state_h,0).unsqueeze(0), torch.stack(all_state_c,0).unsqueeze(0))
            c_t_1 = torch.stack(all_state_context,0)

            coverage_t_1=None
            if config.IS_COVERAGE:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)



            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(input, hidden, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1, src_extend_vocab, coverage_t_1, extra_zeros)

            log_probs = torch.log(final_dist + 1e-12)
            #print(log_probs,torch.max(log_probs),log_probs.shape)
            topk_log_probs, topk_idx = torch.topk(log_probs, config.BEAM_SIZE*2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps==0 else len(beams)

            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i].unsqueeze(0)
                coverage_i = coverage_t[0] if config.IS_COVERAGE else None

                for j in range(config.BEAM_SIZE*2):
                    new_beams = h.extend(
                        token=topk_idx[i,j].item(),
                        log_probs=topk_log_probs[i,j].item(),
                        state=state_i,
                        context=context_i,
                        coverage=coverage_i
                    )
                    all_beams.append(new_beams)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.stoi['<eos>']:
                    if steps>=35:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams)  == config.BEAM_SIZE or len(results)==config.BEAM_SIZE:
                    break

            steps+=1

        if(len(results)==0):
            results = beams

        beam_sorted = self.sort_beams(results)

        return beam_sorted[0], article_oovs

beam_Searcher = BeamSearch(vocab, vocab_ext, test_iterator, model)
beam_Searcher.decode()
