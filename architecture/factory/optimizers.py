import architecture.constants as constants
import torch


class Optimizers:
  
  @staticmethod
  def generate(model: torch.nn, learning_rate: float):
    if model.variation == constants.MODEL_VARIATION_MLP:
      return torch.optim.Adam(model.parameters(), lr = learning_rate)

    elif model.variation == constants.MODEL_VARIATION_ELMAN:
      return torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
    
    raise LookupError('Unable to find the requested object')
