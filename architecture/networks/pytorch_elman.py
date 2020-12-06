import torch
import torch.nn.functional as functional
import torch.nn as nn
from architecture.constants import MODEL_VARIATION_PYTORCH_ELMAN


class PyTorchElman(torch.nn.Module):

  def __init__(self, vocab_size: int, hidden: int = 300, num_cls: int = 2):
    super(PyTorchElman, self).__init__()

    self.variation = MODEL_VARIATION_PYTORCH_ELMAN
    self.layers = 1
    self.hidden_dim = 300

    self.num_cls = num_cls
    self.embeddings = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = hidden)
    self.rnn = nn.RNN(num_layers = 1, input_size = hidden, hidden_size = self.hidden_dim, nonlinearity = 'relu')
    self.linear2 = torch.nn.Linear(in_features = hidden, out_features = num_cls)

  def forward(self, x):
    x = self.embeddings(x)
    hidden = self.init_hidden(x.size()[1])
    x, hidden = self.rnn(x, hidden)

    x = functional.relu(x)
    x = torch.max(x, dim = 1).values  # global maxpool over time dimension
    x = self.linear2(x)

    return x, hidden

  def init_hidden(self, batch_size: int):
    return torch.zeros(self.layers, batch_size, self.hidden_dim)
