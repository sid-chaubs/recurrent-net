import torch

from architecture.metrics import Metrics
from architecture.constants import *
from architecture.data.loader import Loader
from architecture.factory.networks import Networks

import numpy
from copy import deepcopy
from architecture.checkpoint import Checkpoint
import os

loader = Loader(final = False)
loader.hydrate()

model_configs = {
  'vocab_size': len(loader.word_index_map)
}

output = deepcopy(TRAINING_CONFIGS)
for variation in MODEL_VARIATIONS:
  print('Benchmarking for variation', variation)
  for index, config in TRAINING_CONFIGS.items():
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epochs = config['epochs']

    # if not trained, then train else benchmark
    filepath = Checkpoint.get_path(epochs, learning_rate, batch_size, variation)

    if not os.path.exists(filepath):
      message = f'Could not find a trained model for the following configuration {str(config)}'
      raise LookupError(message)
      exit(0)

    loader = Loader(batch_size = batch_size)
    loader.hydrate()

    checkpoint = torch.load(filepath)

    # set the data loader
    data_loader = loader.benchmarking_loader

    # load the network
    model = Networks.generate(variation, {'vocab_size': len(loader.word_index_map)})
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)

    benchmarking_data = iter(loader.benchmarking_loader)
    predictions = list()
    labels = list()

    with torch.no_grad():
      while True:
        try:
          benchmarking_inputs, benchmarking_labels = benchmarking_data.next()
        except StopIteration:
          break

        if model.variation in [MODEL_VARIATION_ELMAN, MODEL_VARIATION_PYTORCH_ELMAN]:
          output, hidden = model(benchmarking_inputs)
          _, result = output.max(1)
        else:
          _, result = model(benchmarking_inputs).max(1)

        predictions.extend(numpy.asarray(result))
        labels.extend(benchmarking_labels.detach().numpy())

