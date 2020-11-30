import glob
import random
import struct
import csv
import numpy as np
import pickle
from tensorflow.core.example import example_pb2
import torchtext
from torchtext.data import Field, Dataset, Example
import re
import spacy
import dill
import torch
import config
import logging
import sys
import warnings
import os
import silence_tensorflow.auto
warnings.filterwarnings("ignore")

input = "config."+str(sys.argv[1])+"_data_path"
output = "config."+str(sys.argv[1])+"_data_path_final"

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

filelist = glob.glob(eval(input))
check = []
for f in filelist:
  reader = open(f,'rb')
  while True:
    len_bytes = reader.read(8)
    if not len_bytes : break
    str_len = struct.unpack('q',len_bytes)[0]
    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    check.append(example_pb2.Example.FromString(example_str))


articles=[]
abstract = []
for i,item in enumerate(check):
  tf_example = item
  articles.append(str(tf_example.features.feature['article'].bytes_list.value[0],'utf-8'))
  abstract.append(str(tf_example.features.feature['abstract'].bytes_list.value[0],'utf-8'))

# articles=articles[0:100000]
# abstract=abstract[0:100000]

full_data = []
for i,item in enumerate(articles):
    curr = [articles[i],abstract[i]]
    full_data.append(curr)

def check_contractions(phrase):
  phrase = re.sub(r"won\'t", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)
  return phrase
def remove_puncts(s):
  s = re.sub(r'[`~\$\@;\[\]:#]', r'', s.lower())
  s = re.sub(' +',' ',s)
  s = s.strip('-')
  s = s.strip('।')
  s = s.strip(' ')
  ret = ''
  for i in s.split():
    if i!='' and i!="-rrb-" and i!="-lrb-" and i!="cnn" and i!="'" and i!="''" and i!='``' and i!='--' and i!='lrb-' and i!='<s>':
      if i=='</s>':
        ret+='. '
      else:
        ret+=check_contractions(i)+" "
  ret = re.sub('\. \.+',' .',ret)
  ret = ret.strip().strip('.')
  ret = re.sub(' +',' ',ret)
  return ret

logging.info('Creating the training examples')

full_data = [[remove_puncts(i[0]),remove_puncts(i[1])] for i in full_data]

fields = [('src',TEXT_en),('trg',TEXT_en)]
examples = []
for i in full_data:
  temp = Example.fromlist([i[0],i[1]],fields=fields)
  examples.append(temp)

train_check = torchtext.data.Dataset(examples,fields)

BATCH_SIZE = config.batch_size
train_iterator = torchtext.data.BucketIterator(
    train_check,BATCH_SIZE,sort_within_batch=False,repeat=False,sort_key=False
)

torch.save(train_check.examples,eval(output),pickle_module=dill)
logging.info("Made the training examples and saved")
