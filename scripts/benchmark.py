import torch

from architecture.checkpoint import Checkpoint
from architecture.metrics import Metrics
from architecture.constants import *
from architecture.data.loader import Loader
from architecture.factory.networks import Networks
import numpy
import json
from copy import deepcopy

loader = Loader(final = False)
loader.hydrate()

model_configs = {
  'vocab_size': len(loader.word_index_map)
}

output = deepcopy(TRAINING_CONFIGS)
for variation in MODEL_VARIATIONS:
  print('Benchmarking model version ' + str(variation))

  for key, config in output.items():
      epochs = config['epochs']
      learning_rate = config['learning_rate']
      batch_size = config['batch_size']
  
      filepath = Checkpoint.get_path(epochs, learning_rate, batch_size, variation)
      checkpoint = torch.load(filepath)

      model = Networks.generate(variation, model_configs)
      model_state = checkpoint['model_state']
      model.load_state_dict(model_state)
  
      num_correct = 0
      num_samples = 0
  
      data = iter(loader.benchmarking_loader)
  
      predictions = list()
      labels = list()
  
      with torch.no_grad():
        while True:
          try:
            batch_inputs, batch_labels = data.next()
          except StopIteration:
            break
  
          _, result = model(batch_inputs).max(1)
          predictions.extend(numpy.asarray(result))
          labels.extend(batch_labels.detach().numpy())
  
      config['accuracy'] = Metrics.accuracy(predictions, labels)

  filepath = f'results-{variation}.json'
  with open(filepath, 'w') as fp:
    json.dump(output, fp)
