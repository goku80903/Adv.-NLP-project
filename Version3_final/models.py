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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
    Dont forget to mark the shapes of all the tensors so there is not mistake
"""
## Local imports
import config

class greedy_decoder(object):
    def __init__(self,outputs,src,trg,vocab):
        self.outputs = outputs
        self.batch_size = outputs.shape[1]
        self.trg_length = outputs.shape[0]
        self.sent = [{'src':'','actual':'','predicted':''}]*self.batch_size
        self.trg = trg
        self.vocab = vocab
        self.src = src
    def decode(self):
        sent = self.sent
        vocab = self.vocab
        for t in range(self.src.shape[0]):
            for i in range(self.batch_size):
                curr = sent[i]
                curr['src']+=vocab.itos[self.src[t,i]]+" "
                sent[i]=curr
        for t in range(1,self.trg_length):
            output = self.outputs[t]
            top1 = output.argmax(1)
            for i in range(self.batch_size):
                curr = sent[i]
                curr['predicted']+=vocab.itos[top1[i]]+" "
                curr['actual']+=vocab.itos[self.trg[t,i]]+" "
                sent[i]=curr
        return sent

"""For initializing the various layers, was giving nan before if initialized randomly"""
def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if(name.startswith('weight_')):
                wt = getattr(lstm,name)
                wt.data.uniform_(-0.02,0.02)
            elif(name.startswith('bias_')):
                bias = getattr(lstm,name)
                n= bias.size(0)
                start,end = n//4, n//2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=1e-4)
    if linear.bias is not None:
        linear.bias.data.normal_(std=1e-4)

def init_wt_normal(wt):
    wt.data.normal_(std=1e-4)

def init_wt_uniform(wt):
    wt.data.uniform_(-0.02,0.02)


class encoderRNN(nn.Module):
    def __init__(self,input_size,emb_size,enc_hid_dim,dec_hid_dim,dropout):
        super(encoderRNN,self).__init__()
        self.input_size = input_size
        self.emb_size = emb_size
        self.hid_dim=enc_hid_dim

        self.embedding = nn.Embedding(input_size,emb_size)
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights))
        self.rnn = nn.LSTM(emb_size,enc_hid_dim,num_layers=1,bidirectional=True,batch_first=True)
        self.reduce_h = nn.Linear(enc_hid_dim*2, enc_hid_dim)
        self.reduce_c = nn.Linear(enc_hid_dim*2, enc_hid_dim)
        self.out = nn.Linear(enc_hid_dim*2,dec_hid_dim*2)
        """Initializing the differnt layers"""
        #init_wt_normal(self.embedding.weight)
        #init_lstm_wt(self.rnn)
        #init_linear_wt(self.reduce_h)
        #init_linear_wt(self.reduce_c)

    def forward(self,input,seq_lens):
        #embedded = self.dropout(self.embedding(input))
        embedded = self.embedding(input)
        packed =  pack_padded_sequence(embedded, seq_lens, batch_first=True)
        outputs,hidden=self.rnn(packed)

        encoder_outputs, _ = pad_packed_sequence(outputs,batch_first = True,padding_value=1)
        encoder_outputs = encoder_outputs.contiguous()

        del outputs

        features = encoder_outputs.view(-1,2*self.hid_dim)
        features = self.out(features)

        h, c = hidden
        h = h.transpose(0,1).contiguous().view(-1,self.hid_dim* 2)
        h = F.relu(self.reduce_h(h))
        c = c.transpose(0,1).contiguous().view(-1,self.hid_dim* 2)
        c = F.relu(self.reduce_c(c))
        hidden = (h.unsqueeze(0),c.unsqueeze(0))

        return encoder_outputs, features, hidden

class Attention(nn.Module):
    def __init__(self,enc_hid_dim,dec_hid_dim):
        super(Attention,self).__init__()
        self.hid_dim = enc_hid_dim

        if config.IS_COVERAGE:
            self.W_c= nn.Linear(1, enc_hid_dim* 2, bias=False)

        self.attn= nn.Linear(enc_hid_dim*2, dec_hid_dim*2)
        self.v = nn.Linear(2*dec_hid_dim ,1,bias=False)

    def forward(self,hidden, encoder_outputs, encoder_features ,enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_feature=self.attn(hidden)
        dec_feature=dec_feature.unsqueeze(1).expand(b,t_k,n).contiguous()
        dec_feature=dec_feature.view(-1,n)

        attn_feature= encoder_features+dec_feature
        if config.IS_COVERAGE:
            coverage_input = coverage.view(-1,1)
            coverage_feature = self.W_c(coverage_input)
            attn_feature = attn_feature + coverage_feature

        e=F.tanh(attn_feature)
        scores=self.v(e)
        scores=scores.view(-1,t_k)

        attn_dist_ = F.softmax(scores,dim=1)*enc_padding_mask
        normalization_fac = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_/normalization_fac

        attn_dist=attn_dist.unsqueeze(1)
        c_t = torch.bmm(attn_dist, encoder_outputs)
        c_t = c_t.view(-1, self.hid_dim*2)
        attn_dist = attn_dist.view(-1,t_k)

        if config.IS_COVERAGE:
            coverage = coverage.view(-1,t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage# context vector and attention distribution

class decoderRNN(nn.Module):
    def __init__(self,output_size,emb_size,enc_hid_dim,dec_hid_dim,dropout):
        super(decoderRNN,self).__init__()
        self.attention = Attention(enc_hid_dim,dec_hid_dim)
        self.hid_dim = enc_hid_dim
        self.output_dim = output_size

        self.embedding = nn.Embedding(output_size, emb_size)
        init_wt_normal(self.embedding.weight)

        self.context = nn.Linear(dec_hid_dim*2 + emb_size, emb_size)

        self.rnn = nn.LSTM(emb_size,dec_hid_dim,num_layers=1, bidirectional=False,batch_first=True)
        #init_lstm_wt(self.rnn)

        if config.IS_POINTER_GEN:
            self.p_gen_linear = nn.Linear(enc_hid_dim* 4+emb_size, 1)

        self.out1 = nn.Linear(dec_hid_dim *3, dec_hid_dim)
        self.out2 = nn.Linear(dec_hid_dim,output_size)
        #init_linear_wt(self.out2)

    def forward(self, input, hidden, encoder_outputs, encoder_features, enc_padding_mask, c_t_1, input_extend_vocab, coverage, extra_zeros):
        #embedded = self.dropout(self.embedding(input))
        embedded = self.embedding(input)
        c_t_1 = c_t_1[0:embedded.shape[0],:]
        embedded = self.context(torch.cat((c_t_1,embedded),1))
        lstm_out, s_t = self.rnn(embedded.unsqueeze(1),hidden)
        h_dec, c_dec = s_t
        h_c_dec = torch.cat((h_dec.view(-1, self.hid_dim),c_dec.view(-1, self.hid_dim)),1)

        c_t, attn_dist, coverage = self.attention(h_c_dec, encoder_outputs, encoder_features, enc_padding_mask,coverage)

        p_gen = None
        if config.IS_POINTER_GEN:
            p_gen_input = torch.cat((c_t, h_c_dec, embedded), 1)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1,self.hid_dim), c_t), 1)

        output = self.out1(output)
        output = self.out2(output)

        final_output = F.softmax(output,dim=1)

        if config.IS_POINTER_GEN:
            vocab_dist_ = p_gen * final_output
            attn_dist_ = (1- p_gen)*attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
            input_extend_vocab = input_extend_vocab.permute(1,0)

            final_output = vocab_dist_.scatter_add(1, input_extend_vocab, attn_dist_)

        return final_output, s_t, c_t, attn_dist, p_gen, coverage

class Seq2Seq(object):
    def __init__(self,encoder,decoder,is_eval=False, load=False):
        super(Seq2Seq,self).__init__()
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()

        self.encoder = encoder
        self.decoder = decoder
        #self.encoder.module.embedding.weight = self.decoder.module.embedding.weight
        self.decoder.embedding.weight = self.encoder.embedding.weight
        if load==True:
            state = torch.load(config.model_final_path)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
