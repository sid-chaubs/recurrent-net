import torch
import torch.nn.functional as functional


class Elman(torch.nn.Module):

  def __init__(self, in_features: int = 300, out_features: int = 300, context_features: int = 300):
    super(Elman, self).__init__()

    self.linear1 = torch.nn.Linear(in_features = (in_features + context_features), out_features = context_features)
    self.linear2 = torch.nn.Linear(in_features = context_features, out_features = out_features)

  def forward(self, x, hidden = None):
    batch_size, timeframe, embedding_dim = x.size()

    if hidden is None:
      hidden = torch.zeros(batch_size, embedding_dim, dtype = torch.float)

    outputs = []
    for time in range(timeframe):
      inp = torch.cat([x[:, time, :], hidden], dim = 1)

      linear1 = self.linear1(inp)
      hidden = linear1.clone()
      activation = functional.sigmoid(linear1)
      linear2 = self.linear2(activation)
      outputs.append(linear2[:, None, :])

    return torch.cat(outputs, dim = 1), hidden
