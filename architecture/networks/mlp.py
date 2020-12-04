import torch
import torch.nn.functional as functional
from architecture.constants import MODEL_VARIATION_MLP


class MLP(torch.nn.Module):

  def __init__(self, vocab_size: int, hidden: int = 300, num_cls: int = 2):
    super(MLP, self).__init__()

    self.variation = MODEL_VARIATION_MLP
    self.num_cls = num_cls
    self.embeddings = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = hidden)
    self.linear1 = torch.nn.Linear(in_features = hidden, out_features = hidden)
    self.linear2 = torch.nn.Linear(in_features = hidden, out_features = num_cls)

  def forward(self, x):
    x = self.embeddings(x)
    x = self.linear1(x)
    x = functional.relu(x)
    x = torch.max(x, dim = 1).values # global maxpool over time dimension
    x = self.linear2(x)

    return x
