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
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

##local files
import config
import models

def load_data():
    train_data = torch.load(config.train_data_path_final,pickle_module=dill)
    val_data = torch.load(config.eval_data_path_final,pickle_module=dill)
    return train_data,val_data

print("Loading data")
train_examples, val_examples = load_data()
print(len(train_examples))
#train_examples = train_examples[0:10000]
#val_examples = val_examples[0:10000]
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
TEXT_dn = torchtext.data.Field(
  tokenize    = tokenize_en,
  lower       = True,
  init_token  = '<sos>',
  eos_token   = '<eos>'
)
fields = [('src',TEXT_en),('trg',TEXT_dn)]
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
TEXT_en.build_vocab(train_dataset)
enc_vocab = TEXT_en.vocab
TEXT_dn.build_vocab(val_dataset)
dec_vocab = TEXT_dn.vocab
print("Size of encoder vocab is:",len(enc_vocab))
print("Size of decoder vocab is:",len(dec_vocab))

INPUT_DIM = len(enc_vocab)
OUTPUT_DIM = len(dec_vocab)
ENC_EMB_DIM = config.ENC_EMB_DIM
DEC_EMB_DIM = config.DEC_EMB_DIM
ENC_HID_DIM = config.ENC_HID_DIM
DEC_HID_DIM = config.DEC_HID_DIM
N_LAYERS = config.N_LAYERS
ENC_DROPOUT = config.ENC_DROPOUT
DEC_DROPOUT = config.DEC_DROPOUT

attention = models.Attention(ENC_HID_DIM,DEC_HID_DIM)
encoder = models.encoderRNN(INPUT_DIM,ENC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT)
decoder = models.decoderRNN(OUTPUT_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,DEC_DROPOUT,attention)

model = models.Seq2Seq(encoder,decoder,device,enc_vocab,dec_vocab)
model = nn.DataParallel(model).cuda()

print("model initialized")
#logging.info("model initialized")

def init_weights(m):
  for name,param in m.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data,mean=0,std=0.01)
    else:
      nn.init.constant_(param.data,0)
model.apply(init_weights)

##Define the optimizer and all
optimizer = optim.Adagrad(model.parameters(),lr=0.15,initial_accumulator_value=0.1)
TRG_PAD_IDX = dec_vocab.stoi['<sos>']
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model,iterator,optimizer,criterion,clip=2):
  model.train()
  total_loss = 0
  for i,batch in enumerate(iterator):
    src = batch.src
    trg = batch.trg
    optimizer.zero_grad()
    output,sent = model(src,trg)
    output_dim = output.shape[-1]
    output = output[1:].view(-1,output_dim).to(device)
    trg = trg[1:].view(-1).to(device)
    loss = criterion(output,trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    total_loss +=loss.item()
  return total_loss/len(iterator)

def evaluate(model,iterator,criterion):
  model.eval()
  epoch_loss = 0
  with torch.no_grad():
    for i,batch in enumerate(iterator):
      src = batch.src
      trg = batch.trg
      output,sent=model(src,trg,0)
      output_dim= output.shape[-1]
      output = output[1:].view(-1, output_dim).to(device)
      trg = trg[1:].view(-1).to(device)
      loss = criterion(output,trg)
      epoch_loss+=loss.item()
  return epoch_loss/len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

model.load_state_dict(torch.load(config.model_path))

print("training is starting")
#logging.info("training is starting")
"""
EPOCH = config.EPOCH
best_valid_loss = float('inf')
train_loss_total = []
test_loss_total = []
for epoch in range(EPOCH):
  start_time = time.time()
  print("starting training..")
  train_loss = train(model,train_iterator,optimizer,criterion)
  print("Training completed!\nStarting eval loss..")
  valid_loss = evaluate(model,val_iterator,criterion)
  print("Evaluation is completed!")
  end_time = time.time()
  epoch_mins, epoch_sec = epoch_time(start_time,end_time)
  train_loss_total.append(train_loss)
  test_loss_total.append(valid_loss)
  if valid_loss<best_valid_loss:
    print("YES")
    best_valid_loss = valid_loss
    torch.save(model.state_dict(),config.model_path)
  print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_sec}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
"""
def evaluate_random(model,iterator,writer = 0):
  model.eval()
  bleu_score = 0
  saved = random.randint(0,len(iterator))
  for i,batch in enumerate(iterator):
    if i==saved:
        src = batch.src
        trg = batch.trg
        output,sent=model(src,trg,0,1)
       # print(output.shape)
       # print(output[0].shape)
       # print(output[0].argmax(1).shape)
       # print(output[0].argmax(1)[1])
       # print(output[1].argmax(1)[2])
        output_dim= output.shape[0]
       # decheck = models.greedy_decoder(output,src,trg,enc_vocab,dec_vocab)
       # ans = decheck.decode()
       # for j in ans:
       #   check_src = j['src'].replace('<sos>','').replace('<eos>','').replace('<pad>','').strip()
       #   print(check_src)
       #   print()
        # print(sent)
        for j in sent:
          check_src = " ".join([dec_vocab.itos[j] for j in list(np.array([i.tolist() for i in j['src']]).flatten())])
          check_actual = " ".join([dec_vocab.itos[j] for j in list(np.array([i.tolist() for i in j['actual']]).flatten())])
          check_predicted = " ".join([dec_vocab.itos[j] for j in list(np.array([i.tolist() for i in j['predicted']]).flatten())])
          print("Src: ",check_src)
          print("Actual: ",check_actual)
          print("Predicted: ",check_predicted)
          print()
print("random evaluation is starting...\n")
# logging.info("random evaluation is starting..\n")

evaluate_random(model,val_iterator)
