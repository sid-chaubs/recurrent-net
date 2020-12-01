from architecture.data.dataset import IMDBDataset
from architecture.data.fetcher import Fetcher
from architecture.constants import DEFAULT_BATCH_SIZE

import torch.utils.data as data
from copy import deepcopy
import torch


class Loader:
  
  def __init__(self, batch_size = DEFAULT_BATCH_SIZE, final = False):
    self.final = final
    self.training_loader = None
    self.benchmarking_loader = None
    self.index_word_map = None
    self.word_index_map = None
    self.num_classes = None
    self.batch_size = batch_size

  def pad(self, batch, token: int = 0):
    inputs, labels = zip(*deepcopy(batch))
    
    padded = deepcopy(inputs)
    max_len = len(max(inputs, key = len))

    for i in range(len(padded)):
      num_zeros = max_len - len(padded[i])
      padded[i].extend([token] * num_zeros)

    # should each entry be a tuple
    return torch.tensor(padded, dtype = torch.long), torch.tensor(labels)

  def collate(self, batch: list):
    return self.pad(batch)

  def hydrate(self):
    (training_inputs, training_labels), (benchmarking_inputs, benchmarking_labels), (index_to_word, word_to_index), cls = Fetcher.retrieve(self.final)

    training_set = IMDBDataset(training_inputs, training_labels)
    self.training_loader = data.DataLoader(training_set, batch_size = self.batch_size, collate_fn = self.collate)

    benchmarking_set = IMDBDataset(benchmarking_inputs, benchmarking_labels)
    self.benchmarking_loader = data.DataLoader(benchmarking_set, batch_size = self.batch_size, collate_fn = self.collate)

    self.index_word_map = index_to_word
    self.word_index_map = word_to_index
    self.num_classes = cls
