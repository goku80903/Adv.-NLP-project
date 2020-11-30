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
import random
from queue import PriorityQueue
import dill
import argparse
import warnings
import logging
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
train_examples = train_examples[0:50000]
val_examples = val_examples[0:9000]
print("Length of train:",len(train_examples))
print("Length of eval:",len(val_examples))
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
BATCH_SIZE = config.batch_size
print(BATCH_SIZE)
train_iterator = torchtext.data.BucketIterator(
    train_dataset,BATCH_SIZE,device=device,repeat=False,sort_within_batch=True,sort_key = lambda x : len(x.src)
)
val_iterator = torchtext.data.BucketIterator(
    val_dataset,BATCH_SIZE,device=device,repeat=False,sort_within_batch=True,sort_key = lambda x: len(x.src)
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

model = models.Seq2Seq(encoder,decoder,load=False)

##Define the optimizer and all
#params= list(model.encoder.module.parameters())+list(model.decoder.module.parameters())
params = list(model.encoder.parameters())+list(model.decoder.parameters())
optimizer = optim.Adagrad(params,lr=0.15,initial_accumulator_value=0.1)
TRG_PAD_IDX = vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def src_to_ids(src_words, vocab):
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


def trg_to_ids(trg_words, vocab, oov):
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

def expand_vocab(batch,vocab_ext,vocab):
  new_src_extend_vocab = Variable(torch.zeros(batch.src.shape,dtype=torch.int64)).cuda()
  new_src_original_vocab = Variable(torch.zeros(batch.src.shape,dtype=torch.int64)).cuda()
  new_trg_extend_vocab = Variable(torch.zeros(batch.trg.shape,dtype=torch.int64)).cuda()
  new_trg_original_vocab = Variable(torch.zeros(batch.trg.shape,dtype=torch.int64)).cuda()
  batch_oov = []
  max_len = 0
  for i in range(batch.src.shape[1]):
    src_sent = " ".join([vocab_ext.itos[j] for j in list(batch.src[:,i])])
    trg_sent = " ".join([vocab_ext.itos[j] for j in list(batch.trg[:,i])])
    new_src_original_vocab[:,i], new_src_extend_vocab[:,i], oov = src_to_ids(src_sent.split(), vocab)
    new_trg_original_vocab[:,i] ,new_trg_extend_vocab[:,i] = trg_to_ids(trg_sent.split(), vocab, oov)
    batch_oov.append(oov)
    max_len = max(max_len, len(oov))
  extra_zeros = None
  if max_len >0:
    extra_zeros = Variable(torch.zeros((batch.src.shape[1], max_len))).cuda()
  return new_src_original_vocab[:,:config.MAX_ENC_LEN], new_trg_original_vocab, new_src_extend_vocab[:,:config.MAX_ENC_LEN], new_trg_extend_vocab, batch_oov, extra_zeros

def train(model,iterator,optimizer,clip=2):
  model.encoder.train()
  model.decoder.train()
  total_loss = 0
  for i,batch in enumerate(iterator):
    """Code for expanding the vocab"""
    src, trg, src_extend_vocab, trg_extend_vocab, article_oovs, extra_zeros = expand_vocab(batch, vocab_ext, vocab)
    src = src.permute(1,0)
    enc_padding_mask = ~(src==TRG_PAD_IDX)
    dec_padding_mask = ~(trg==TRG_PAD_IDX)
    enc_lens = torch.sum(enc_padding_mask,dim=1)
    dec_len = trg.shape[1]
    trg_for_check = trg_extend_vocab

    optimizer.zero_grad()
    encoder_output, encoder_features, encoder_hidden = model.encoder(src,enc_lens)
    step_losses = []
    input = trg[0,:]
    c_t = Variable(torch.zeros((src.shape[0], 2* config.ENC_HID_DIM))).cuda()
    coverage=None
    if config.IS_COVERAGE:
      coverage = Variable(torch.zeros(src.size())).cuda()

    for t in range(1,min(dec_len,config.MAX_DEC_LEN)):
      final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = model.decoder(input, encoder_hidden, encoder_output, encoder_features, enc_padding_mask, c_t, src_extend_vocab, coverage, extra_zeros)

      gold_probs = torch.gather(final_dist, 1, trg_for_check[t].unsqueeze(1)).squeeze()
      step_loss = -torch.log(gold_probs+1e-12)
      del gold_probs

      if config.IS_COVERAGE:
        step_coverage_loss = torch.sum(torch.min(attn_dist,coverage),1)
        step_loss = step_loss + step_coverage_loss
        coverage = next_coverage
        del step_coverage_loss


      step_mask = dec_padding_mask[t]
      step_loss = step_mask * step_loss
      step_losses.append(step_loss)
      input = trg[t,:]

    sum_losses = torch.sum(torch.stack(step_losses,1),1)
    batch_avg_loss = sum_losses/dec_len
    loss = torch.mean(batch_avg_loss)

    loss.backward()

    nn.utils.clip_grad_norm_(model.encoder.parameters(),clip)
    nn.utils.clip_grad_norm_(model.decoder.parameters(),clip)

    optimizer.step()
    total_loss+=loss.item()
    del step_losses

  return (total_loss)/len(iterator)

def evaluate(model, iterator):
  model.encoder.eval()
  model.decoder.eval()
  total_loss = 0
  for i,batch in enumerate(iterator):
    """Code for expanding the vocab"""
    src, trg, src_extend_vocab, trg_extend_vocab, article_oovs,extra_zeros = expand_vocab(batch, vocab_ext, vocab)
    src = src.permute(1,0)
    enc_padding_mask = ~(src==TRG_PAD_IDX)
    dec_padding_mask = ~(trg==TRG_PAD_IDX)
    enc_lens = torch.sum(enc_padding_mask,dim=1)
    dec_len = trg.shape[1]
    trg_for_check = trg_extend_vocab

    encoder_output, encoder_features, encoder_hidden = model.encoder(src,enc_lens)
    step_losses = []
    input = trg[0,:]
    c_t = Variable(torch.zeros((src.shape[0], 2* config.ENC_HID_DIM))).cuda()
    coverage=None
    if config.IS_COVERAGE:
      coverage = Variable(torch.zeros(src.size())).cuda()

    for t in range(1,dec_len):
      final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = model.decoder(input, encoder_hidden, encoder_output, encoder_features, enc_padding_mask, c_t, src_extend_vocab, coverage, extra_zeros)
      target = trg_for_check[t]
      gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
      step_loss = -torch.log(gold_probs+1e-12)
      del gold_probs

      if config.IS_COVERAGE:
        step_coverage_loss = torch.sum(torch.min(attn_dist,coverage),1)
        step_loss = step_loss + step_coverage_loss
        coverage = next_coverage
        del step_coverage_loss

      input = trg[t]
      step_mask = dec_padding_mask[t]
      step_loss = step_mask * step_loss
      step_losses.append(step_loss)

    sum_losses = torch.sum(torch.stack(step_losses,1),1)
    batch_avg_loss = sum_losses/dec_len
    loss = torch.mean(batch_avg_loss)

    total_loss+=loss.item()
    del step_losses

  return (total_loss)/len(iterator)

def train_iters(model, train_iterator, val_iterator, optimizer):
  EPOCH = config.EPOCH
  best_val_loss = float('inf')
  epoch=0
  #for epoch in range(EPOCH):
  while best_val_loss>1:
    start_time = time.time()
    print("starting training...")
    logging.info("starting training...")
    train_loss = train(model,train_iterator,optimizer)
    logging.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    if train_loss<1.1:
        break
    del train_loss
    print("Starting eval loss..")
    logging.info("Starting eval loss..")
    with torch.no_grad():
        valid_loss = evaluate(model,val_iterator)
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    logging.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print("Evaluation is completed!")
    logging.info("Evaluation is completed!")
    end_time = time.time()
    epoch_mins, epoch_sec = epoch_time(start_time,end_time)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_sec}s')
    logging.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_sec}s')
    """Code for saving the model"""
    state = {
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if valid_loss < best_val_loss:
        best_val_loss= valid_loss
        print("YESSS!!!")
        torch.save(state,config.model_path)
    epoch+=1
    del valid_loss

  state = {
      'encoder_state_dict': model.encoder.state_dict(),
      'decoder_state_dict': model.decoder.state_dict(),
      'optimizer': optimizer.state_dict()
  }
  torch.save(state,config.model_final_path)

train_iters(model,train_iterator,val_iterator,optimizer)
"Training completed"

def decode_random(model,iterator,vocab):
  model.encoder.eval()
  model.decoder.eval()
  total_loss = 0
  for i,batch in enumerate(iterator):
    """Code for expanding the vocab"""
    src, trg, src_extend_vocab, trg_extend_vocab, article_oovs,extra_zeros = expand_vocab(batch, vocab_ext, vocab)
    src = src.permute(1,0)
    enc_padding_mask = ~(src==TRG_PAD_IDX)
    dec_padding_mask = ~(trg==TRG_PAD_IDX)
    enc_lens = torch.sum(enc_padding_mask,dim=1)
    dec_len = trg.shape[1]
    trg_for_check = trg_extend_vocab

    encoder_output, encoder_features, encoder_hidden = model.encoder(src,enc_lens)
    step_losses = []
    input = trg[0,:]
    c_t = Variable(torch.zeros((src.shape[0], 2* config.ENC_HID_DIM))).cuda()
    coverage=None
    if config.IS_COVERAGE:
      coverage = Variable(torch.zeros(src.size())).cuda()

    for t in range(1,dec_len):
      final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = model.decoder(input, encoder_hidden, encoder_output, encoder_features, enc_padding_mask, c_t, src_extend_vocab, coverage, extra_zeros)
      target = trg_for_check[t]
      coverage = next_coverage

      input = trg[t]

      log_probs = torch.log(final_dist)
      topk_log, topk_ids = torch.topk(log_probs, 2)
      print(topk_ids, trg_for_check[t])
      print("fake target :",trg[t])


with torch.no_grad():
    check = decode_random(model,train_iterator,vocab)
print(check)
