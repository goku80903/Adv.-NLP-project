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
torch.cuda.empty_cache()
print("Cache Emptied!\n")

##local files
import config
import models

def load_data():
    train_data = torch.load(config.train_data_path_final,pickle_module=dill)
    val_data = torch.load(config.eval_data_path_final,pickle_module=dill)
    return train_data,val_data

print("Loading data")
train_examples, val_examples = load_data()
train_examples = train_examples[0:100]
#val_examples = val_examples[0:100]
print(len(train_examples))
print("Length of eval:,",len(val_examples))
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
BATCH_SIZE = 1
print(BATCH_SIZE)
train_iterator = torchtext.data.BucketIterator(
    train_dataset,BATCH_SIZE,sort_within_batch=False,device=device,repeat=False,sort_key=False
)
val_iterator = torchtext.data.BucketIterator(
    val_dataset,BATCH_SIZE,sort_within_batch=False,device=device,repeat=False,sort_key=False
)
print("iterators are made and the vocab is being built")
TEXT_en.build_vocab(train_dataset.src,train_dataset.trg,val_dataset.src,val_dataset.trg,min_freq=27)
vocab = TEXT_en.vocab
print("Size of encoder vocab is:",len(vocab))

INPUT_DIM = len(vocab)
OUTPUT_DIM = len(vocab)
ENC_EMB_DIM = config.ENC_EMB_DIM
DEC_EMB_DIM = config.DEC_EMB_DIM
ENC_HID_DIM = config.ENC_HID_DIM
DEC_HID_DIM = config.DEC_HID_DIM
N_LAYERS = config.N_LAYERS
ENC_DROPOUT = config.ENC_DROPOUT
DEC_DROPOUT = config.DEC_DROPOUT

attention = models.Attention(ENC_HID_DIM,DEC_HID_DIM)
encoder = models.encoderRNN(INPUT_DIM,ENC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT)
#encoder = nn.DataParallel(encoder).cuda()
encoder.cuda()
decoder = models.decoderRNN(OUTPUT_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,DEC_DROPOUT,attention)
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
    def __init__(self, tokens, log_probs, state, context):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context

    def extend(self, token, log_probs, state, context):
        return Beam(
            tokens = self.tokens + [token],
            log_probs = self.log_probs + log_probs,
            state = state,
            context = context
        )
    
    def avg_log_prob(self):
        return (log_probs)/len(self.tokens)

    def latest_token(self):
        return self.tokens[-1]

class BeamSearch(object):
    def __init__(self, vocab, iterator, model):
        self.vocab = vocab
        self.iterator = iterator
        self.model = model
    
    def sort_beams(self, beams):
        return sorted(beams, key = lambda h: h.avg_log_prob, reverse=True)

    def print_sent(self, sent):
        print('src: ',sent['src'])
        print('\nacutal: ',sent['actual'])
        print('\npredicted: ',sent['predicted'])
        print("\n")

    def decode(self):
        start = time.time()
        for i, batch in self.iterator:
            with torch.no_grad():
                best_summary = self.beam_search(batch)

            output_ids = [int(t) for t in best_summary]
            decoded_words = [self.vocab.itos[t] for t in output_ids]

            sent = [{'src':'','actual':'','predicted':''}]
            sent['predicted'] = " ".join(decoded_words)
            src = batch.src
            trg = batch.trg
            sent['src']=" ".join[vocab.itos[t] for t in src[0]]
            sent['src'] = sent['src'].replace('<sos>',"").replace("<eos>","").replace('<pad>',"").strip()
        
            sent['actual']=" ".join[vocab.itos[t] for t in trg[0]]
            sent['actual'] = sent['actual'].replace('<sos>',"").replace("<eos>","").replace('<pad>',"").strip()
            self.print_sent(sent)
            """Write the function for checking rogue score and take average"""
        end = time.time()
        """Code for printing the time taken for decoding the entire batch"""

    def beam_search(self,batch):
        self.model.encoder.eval()
        self.model.decoder.eval()

        src = batch.src
        src = src.permute(1,0)
        trg = batch.trg
        enc_padding_mask = ~(src==TRG_PAD_IDX)
        dec_padding_mask = ~(trg==TRG_PAD_IDX)
        enc_len = src.shape[0]
        dec_len = trg.shape[0]

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(src)

        h_0, c_0 = encoder_hidden
        h_0 = h_0.squeeze()
        c_0 = c_0.squeeze()

        beams = [Beam(
            tokens=self.vocab.stoi['<sos>'],
            log_probs=0.0,
            state=(h_0[0],c_0[0]),
            context=Variable(torch.zeros((BATCH_SIZE,2* config.DEC_HID_DIM)))[0]
        ) for _ in range(config.BEAM_SIZE)]

        results = []
        steps = 0

        while steps<config.MAX_DEC_LEN+20 and len(results) < config.BEAM_SIZE:
            latest_token = [h.latest_token for h in beams]
            input = Variable(torch.LongTensor(latest_token))

            all_state_h = []
            all_state_c = []
            all_state_context = []
            for i in beams:
                state_h, state_c = i.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_state_context.append(i.context)
            
            hidden = (torch.stack(all_state_h,0).unsqueeze(0), torch.stack(all_state_c,0).unsqueeze(0))
            c_t_1 = torch.stack(all_state_context,0)

            final_dist, s_t, c_t, attn_dist = self.model.decoder(input, hidden, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_idx = torch.topk(log_probs, config.BEAM_SIZE*2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps==0 else len(beams)

            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]

                for j in range(config.BEAM_SIZE*2):
                    new_beams = h.extend(
                        token=topk_idx[i,j].item(),
                        log_probs=topk_log_probs[i,j].item(),
                        state=state_i,
                        context=context_i
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

        return beam_sorted[0]