import torch
import torch.nn.functional as functional
from architecture.modules.elman import Elman as ElmanLayer
from architecture.constants import MODEL_VARIATION_ELMAN


class Elman(torch.nn.Module):

  def __init__(self, vocab_size: int, hidden: int = 300, num_cls: int = 2):
    super(Elman, self).__init__()

    self.variation = MODEL_VARIATION_ELMAN
    self.num_cls = num_cls
    self.embeddings = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = hidden)
    self.elman = ElmanLayer(in_features = hidden, out_features = hidden, context_features = 300)
    self.linear2 = torch.nn.Linear(in_features = hidden, out_features = num_cls)

  def forward(self, x):
    x = self.embeddings(x)
    x, hidden = self.elman(x)
    x = functional.relu(x)
    x = torch.max(x, dim = 1).values  # global maxpool over time dimension
    x = self.linear2(x)

    return x, hidden
