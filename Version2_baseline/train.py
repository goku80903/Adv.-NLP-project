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
from queue import PriorityQueue
import dill
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
import logging
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
print("Cache Emptied!\n")
logging.info("Cache Emptied!")

##local files
import config
import models


def load_data():
    train_data = torch.load(config.train_data_path_final,pickle_module=dill)
    val_data = torch.load(config.eval_data_path_final,pickle_module=dill)
    return train_data,val_data

print("Loading data")
train_examples, val_examples = load_data()
print("Length of train:",len(train_examples))
logging.info("Length of train: %d",len(train_examples))
print("Length of eval:,",len(val_examples))
logging.info("Length of eval: %d",len(val_examples))
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
    train_dataset,BATCH_SIZE,device=device,repeat=False,sort_within_batch = True,sort_key = lambda x : len(x.src)
)
val_iterator = torchtext.data.BucketIterator(
    val_dataset,BATCH_SIZE,device=device,repeat=False,sort_within_batch = True,sort_key = lambda x : len(x.src)
)
print("iterators are made and the vocab is being built")
TEXT_en.build_vocab(train_dataset.src,train_dataset.trg,val_dataset.src,val_dataset.trg,min_freq=27)
#TEXT_en.build_vocab(train_dataset.src,train_dataset.trg,val_dataset.src,val_dataset.trg,min_freq=10)
vocab = TEXT_en.vocab
print("Size of encoder vocab is:",len(vocab))
logging.info("Size of encoder vocab is: %d",len(vocab))

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
print(encoder)
print(decoder)

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


def train(model,iterator,optimizer,clip=2):
  model.encoder.train()
  model.decoder.train()
  total_loss = 0
  for i,batch in enumerate(iterator):
    src = batch.src
    src = src.permute(1,0)
    src = src[:,:config.MAX_ENC_LEN]
    trg = batch.trg
    trg = trg.permute(1,0)
    enc_padding_mask = ~(src==TRG_PAD_IDX)
    dec_padding_mask = ~(trg==TRG_PAD_IDX)
    enc_lens = torch.sum(enc_padding_mask,dim=1)
    dec_len = trg.shape[1]-1

    optimizer.zero_grad()
    encoder_output, encoder_features, encoder_hidden = model.encoder(src,enc_lens)
    step_losses = []
    input = trg[:,0]
    for t in range(1,min(dec_len,config.MAX_DEC_LEN)):
      final_dist, s_t, c_t, attn_dist = model.decoder(input, encoder_hidden, encoder_output, encoder_features, enc_padding_mask)

      gold_probs = torch.gather(final_dist, 1, trg[:,t].unsqueeze(1)).squeeze()
      step_loss = -torch.log(gold_probs)
      del gold_probs

      step_mask = dec_padding_mask[:,t]
      step_loss = step_mask * step_loss
      step_losses.append(step_loss)

      input = trg[:,t]


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
    src = batch.src
    trg = batch.trg
    src = src.permute(1,0)
    trg = trg.permute(1,0)
    enc_padding_mask = ~(src==TRG_PAD_IDX)
    dec_padding_mask = ~(trg==TRG_PAD_IDX)
    enc_lens = torch.sum(enc_padding_mask,dim=1)
    dec_len = trg.shape[1]-1
    trg_for_check = trg[::1]

    encoder_output, encoder_features, encoder_hidden = model.encoder(src,enc_lens)

    step_losses = []
    input = trg[:,0]
    for t in range(dec_len):
      final_dist, s_t, c_t, attn_dist = model.decoder(input, encoder_hidden, encoder_output, encoder_features, enc_padding_mask)
      target = trg_for_check[:,t]
      gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
      step_loss = -torch.log(gold_probs+1e-12)
      del gold_probs

      input = trg[:,t]
      step_mask = dec_padding_mask[:,t]
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
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    logging.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    if train_loss<1:
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
def evaluate_random(model, iterator):
  torch.no_grad()
  model.encoder.eval()
  model.decoder.eval()
  total_loss = 0
  for i,batch in enumerate(iterator):
    src = batch.src
    trg = batch.trg
    src = src.permute(1,0)
    trg = trg.permute(1,0)
    enc_padding_mask = ~(src==TRG_PAD_IDX)
    dec_padding_mask = ~(trg==TRG_PAD_IDX)
    enc_lens = torch.sum(enc_padding_mask,dim=1)
    dec_len = trg.shape[1]-1
    trg_for_check = trg[::1]

    encoder_output, encoder_features, encoder_hidden = model.encoder(src,enc_lens)

    step_losses = []
    input = trg[:,0]
    for t in range(min(5,dec_len)):
      final_dist, s_t, c_t, attn_dist = model.decoder(input, encoder_hidden, encoder_output, encoder_features, enc_padding_mask)
      target = trg_for_check[:,t]
      gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
      step_loss = -torch.log(gold_probs+1e-12)
      del gold_probs

      input = trg[:,t]
      step_mask = dec_padding_mask[:,t]
      step_loss = step_mask * step_loss
      step_losses.append(step_loss)
      print(input)

      log_probs = torch.log(final_dist)
      topk_log_probs, topk_ids = torch.topk(log_probs, 2)
      print(topk_ids)

    sum_losses = torch.sum(torch.stack(step_losses,1),1)
    batch_avg_loss = sum_losses/dec_len
    loss = torch.mean(batch_avg_loss)

    total_loss+=loss.item()
    del step_losses

evaluate_random(model,train_iterator)
