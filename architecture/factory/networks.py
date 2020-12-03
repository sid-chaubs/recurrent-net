import architecture.constants as constants

from architecture.networks.elman import Elman
from architecture.networks.mlp import MLP


class Networks:

  @staticmethod
  def generate(variation: str, configs: dict):
    if variation == constants.MODEL_VARIATION_MLP:
      return MLP(**configs)

    elif variation == constants.MODEL_VARIATION_ELMAN:
      return Elman(**configs)
    
    raise LookupError('Unable to find the requested object')
