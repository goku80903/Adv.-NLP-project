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
import gensim
import pickle
import random
from torchtext.data import Field, Dataset, Example
import numpy as np
import math
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
from queue import PriorityQueue
import dill

class greedy_decoder(object):
    def __init__(self,outputs,src,trg,enc_vocab,dec_vocab):
        self.outputs = outputs
        self.batch_size = outputs.shape[1]
        self.trg_length = outputs.shape[0]
        self.sent = [{'src':'','actual':'','predicted':''}]*self.batch_size
        self.trg = trg
        self.dec_vocab = dec_vocab
        self.enc_vocab = enc_vocab
        self.src = src
    def decode(self):
        sent = self.sent
        dec_vocab = self.dec_vocab
        enc_vocab = self.enc_vocab
        for t in range(self.src.shape[0]):
            for i in range(self.batch_size):
                curr = sent[i]
                curr['src']+=enc_vocab.itos[self.src[t,i]]+" "
        for t in range(1,self.trg_length):
            output = self.outputs[t]
            top1 = output.argmax(1)
            for i in range(self.batch_size):
                curr = sent[i]
                curr['predicted']+=dec_vocab.itos[top1[i]]+" "
                curr['actual']+=dec_vocab.itos[self.trg[t,i]]+" "
                sent[i]=curr
        return sent

class encoderRNN(nn.Module):
    def __init__(self,input_size,emb_size,enc_hid_dim,dec_hid_dim,dropout):
        super(encoderRNN,self).__init__()
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size,emb_size)
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights))
        self.rnn = nn.GRU(emb_size,enc_hid_dim,bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(enc_hid_dim*2,dec_hid_dim)

    def forward(self,input):
        embedded = self.dropout(self.embedding(input))
        # embedded = self.embedding(input)
        outputs,hidden=self.rnn(embedded)
        hidden = self.out(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self,enc_hid_dim,dec_hid_dim):
        super(Attention,self).__init__()

        self.attn=nn.Linear((enc_hid_dim*2)+dec_hid_dim,dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim,1,bias=False)

    def forward(self,hidden,encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1,src_len,1)
        encoder_outputs = encoder_outputs.permute(1,0,2)
        energy = torch.tanh(self.attn(torch.cat((hidden,encoder_outputs), dim = 2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention,dim=1)

class decoderRNN(nn.Module):
    def __init__(self,output_size,emb_size,enc_hid_dim,dec_hid_dim,dropout,attention):
        super(decoderRNN,self).__init__()

        self.output_dim = output_size
        self.emb_dim = emb_size
        self.hidden_dim = dec_hid_dim

        self.attention = attention
        self.embedding = nn.Embedding(output_size,emb_size)
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights))
        self.rnn = nn.GRU((enc_hid_dim*2)+emb_size,dec_hid_dim)
        self.out = nn.Linear((enc_hid_dim*2)+dec_hid_dim+emb_size,output_size)
        self.dropout=nn.Dropout(dropout)

    def forward(self, input,hidden,encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        # embedded = self.embedding(input)
        a = self.attention(hidden,encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1,0,2)
        weighted = torch.bmm(a,encoder_outputs)
        weighted = weighted.permute(1,0,2)
        rnn_input = torch.cat((embedded,weighted),dim=2)
        output,hidden=self.rnn(rnn_input,hidden.unsqueeze(0))
        assert (output==hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.out(torch.cat((output,weighted,embedded),dim=1))
        return prediction,hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device,enc_vocab,dec_vocab):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

    def forward(self,src,trg,training=1,teaching = 0.5):
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab = self.decoder.output_dim
        enc_vocab = self.enc_vocab
        dec_vocab = self.dec_vocab

        outputs = torch.zeros(trg_length,batch_size,trg_vocab).to(self.device)
        encoder_outputs,hidden = self.encoder(src)
        input = trg[0,:]
        sent=[]
        for i in range(batch_size):
            sent.append({'src':'','predicted':'','actual':''})
        for t in range(src.shape[0]):
            for i in range(batch_size):
                curr = sent[i]
                curr['src']+=enc_vocab.itos[src[t,i]]+" "
        for t in range(1,trg_length):
            output , hidden = self.decoder(input,hidden,encoder_outputs)
            outputs[t]=output
            force = random.random() < teaching
            top1 = output.argmax(1)
            input = top1 if force else trg[t]
            # sent = greedy_search(sent,output,trg,t)
            for i in range(batch_size):
                curr = sent[i]
                curr['predicted']+=dec_vocab.itos[top1[i]]+" "
                curr['actual']+=dec_vocab.itos[trg[t,i]]+" "
                sent[i]=curr
        if training==0:
            #print("Eval")
            for i in range(len(sent)):
                actual = sent[i]['actual'].replace('<sos>','').replace("<eos>","").replace("<pad>","").strip()
                src = sent[i]['src'].replace('<sos>','').replace("<eos>","").replace("<pad>","").strip()
                predicted = sent[i]['predicted'].replace('<sos>','').replace("<eos>","").replace("<pad>","").strip()
                actual_tensor = torch.tensor([dec_vocab.stoi[j] for j in actual.split()]).cuda()
                src_tensor = torch.tensor([dec_vocab.stoi[j] for j in src.split()]).cuda()
                predicted_tensor = torch.tensor([dec_vocab.stoi[j] for j in predicted.split()]).cuda()
                sent[i]['actual']=actual_tensor
                sent[i]['src']=src_tensor
                sent[i]['predicted']=predicted_tensor
        #else:
            #print("Training")
        return outputs,sent
