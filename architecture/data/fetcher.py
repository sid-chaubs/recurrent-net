from architecture.constants import IMDB_URL, IMDB_FILE

import gzip
import os
import pickle
import random
import wget


class Fetcher:

  @staticmethod
  def retrieve(final = False, val = 5000, seed = 0, voc = None, char = False):
    
    cst = 'char' if char else 'word'
    
    imdb_url = IMDB_URL.format(cst)
    imdb_file = IMDB_FILE.format(cst)
    
    if not os.path.exists(imdb_file):
      wget.download(imdb_url)
    
    with gzip.open(imdb_file) as file:
      sequences, labels, i2w, w2i = pickle.load(file)
    
    if voc is not None and voc < len(i2w):
      nw_sequences = {}
      
      i2w = i2w[:voc]
      w2i = {w: i for i, w in enumerate(i2w)}
      
      mx, unk = voc, w2i['.unk']
      for key, seqs in sequences.items():
        nw_sequences[key] = []
        for seq in seqs:
          seq = [s if s < mx else unk for s in seq]
          nw_sequences[key].append(seq)
      
      sequences = nw_sequences
    
    if final:
      return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2
    
    # Make a validation split
    random.seed(seed)
    
    x_train, y_train = [], []
    x_val, y_val = [], []
    
    val_ind = set(random.sample(range(len(sequences['train'])), k = val))
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
      if i in val_ind:
        x_val.append(s)
        y_val.append(l)
      else:
        x_train.append(s)
        y_train.append(l)
    
    return (x_train, y_train), (x_val, y_val), (i2w, w2i), 2
