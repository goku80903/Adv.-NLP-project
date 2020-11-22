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
#train_examples = train_examples[0:100]
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
BATCH_SIZE = config.batch_size
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

def train(model,iterator,optimizer,clip=2):
  model.encoder.train()
  model.decoder.train()
  total_loss = 0
  for i,batch in enumerate(iterator):
    src = batch.src
    src = src.permute(1,0)
    trg = batch.trg
    enc_padding_mask = ~(src==TRG_PAD_IDX)
    dec_padding_mask = ~(trg==TRG_PAD_IDX)
    enc_len = src.shape[0]
    dec_len = trg.shape[0]
    trg_for_check = trg[1::]

    optimizer.zero_grad()
    encoder_output, encoder_features, encoder_hidden = model.encoder(src)
    step_losses = []
    input = trg[0,:]
    for t in range(1,min(dec_len,config.MAX_DEC_LEN)):
      final_dist, s_t, c_t, attn_dist = model.decoder(input, encoder_hidden, encoder_output, encoder_features, enc_padding_mask)

      gold_probs = torch.gather(final_dist, 1, trg[t].unsqueeze(1)).squeeze()
      step_loss = -torch.log(gold_probs+1e-12)
      del gold_probs

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
    src = batch.src
    trg = batch.trg
    src = src.permute(1,0)
    enc_padding_mask = ~(src==TRG_PAD_IDX)
    dec_padding_mask = ~(trg==TRG_PAD_IDX)
    enc_len = src.shape[0]
    dec_len = trg.shape[0]
    trg_for_check = trg[1::]

    encoder_output, encoder_features, encoder_hidden = model.encoder(src)

    step_losses = []
    input = trg[0,:]
    c_t = Variable(torch.zeros((BATCH_SIZE, 2*config.DEC_HID_DIM)))
    for t in range(1,dec_len):
      final_dist, s_t, c_t, attn_dist = model.decoder(input, encoder_hidden, encoder_output, encoder_features, enc_padding_mask, c_t)
      target = trg[t]
      gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
      step_loss = -torch.log(gold_probs+1e-12)
      del gold_probs

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
  for epoch in range(EPOCH):
    start_time = time.time()
    print("starting training...")
    train_loss = train(model,train_iterator,optimizer)
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    del train_loss
    print("Starting eval loss..")
    with torch.no_grad():
        valid_loss = evaluate(model,val_iterator)
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print("Evaluation is completed!")
    end_time = time.time()
    epoch_mins, epoch_sec = epoch_time(start_time,end_time)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_sec}s')
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
    del valid_loss

train_iters(model,train_iterator,val_iterator,optimizer)
"Training completed"

def decode_random(model,iterator,vocab):
    decode_index = random.random(0,len(iterator))
    sent = [{'src':'','actual':'','predicted':''}]*config.batch_size
    for i,batch in enumerate(iterator):
        if(i==decode_index):
            src = batch.src
            trg = batch.trg
            enc_padding_mask = ~(src==TRG_PAD_IDX)
            dec_padding_mask = ~(trg==TRG_PAD_IDX)
            enc_len = src.shape[0]
            dec_len = trg.shape[0]
            trg_for_check = trg[1::]

            for t in range(enc_len):
                for i in range(config.batch_size):
                    curr = sent[i]
                    curr['src']+=vocab.itos[src[t,i]]+' '
                    sent[i]=curr
            encoder_output, encoder_features, encoder_hidden = model.encoder(src)
            input = trg[0,:]
            for t in range(1,dec_len):
                final_dist, s_t, c_t, attn_dist = model.decoder(input, encoder_hidden, encoder_output, encoder_features, enc_padding_mask)
                log_probs = torch.log(final_dist)
                topk_log_probs, topk_ids = torch.topk(log_probs, 2)
                target = trg[t]
                for i in range(config.batch_size):
                    curr = sent[i]
                    curr['actual']+=vocab.itos[target[i]]+' '
                    curr['predicted']+=vocab.itos[topk_ids[i,0]]+' '
                input = topk_ids[:,0]
    return sent

check = decode_random(model,train_iterator,vocab)
print(check)
