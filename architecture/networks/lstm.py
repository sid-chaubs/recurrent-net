import torch
import torch.nn.functional as functional
import torch.nn as nn
from architecture.constants import MODEL_VARIATION_PYTORCH_LSTM


class LSTM(torch.nn.Module):

  def __init__(self, vocab_size: int, hidden: int = 300, num_cls: int = 2):
    super(LSTM, self).__init__()

    self.variation = MODEL_VARIATION_PYTORCH_LSTM
    self.layers = 1
    self.hidden_dim = 300

    self.num_cls = num_cls
    self.embeddings = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = hidden)
    self.lstm = nn.LSTM(input_size = hidden, hidden_size = self.hidden_dim)
    self.linear2 = torch.nn.Linear(in_features = hidden, out_features = num_cls)

  def forward(self, x):
    x = self.embeddings(x)
    x, _ = self.lstm(x)
    x = functional.relu(x)
    x = torch.max(x, dim = 1).values  # global maxpool over time dimension
    x = self.linear2(x)

    return x
