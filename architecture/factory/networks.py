import architecture.constants as constants

from architecture.networks.custom_elman import Elman
from architecture.networks.mlp import MLP
from architecture.networks.pytorch_elman import PyTorchElman
from architecture.networks.lstm import LSTM


class Networks:

  @staticmethod
  def generate(variation: str, configs: dict):
    if variation == constants.MODEL_VARIATION_MLP:
      return MLP(**configs)

    elif variation == constants.MODEL_VARIATION_ELMAN:
      return Elman(**configs)

    elif variation == constants.MODEL_VARIATION_PYTORCH_LSTM:
      return LSTM(**configs)

    elif variation == constants.MODEL_VARIATION_PYTORCH_ELMAN:
      return PyTorchElman(**configs)

    raise LookupError('Unable to find the requested object')
