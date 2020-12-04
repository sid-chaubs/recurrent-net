import torch

from architecture.trainer import Trainer
from architecture.metrics import Metrics
from architecture.constants import *
from architecture.data.loader import Loader
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
  print('Train model version ' + str(variation))
  
  for index, config in TRAINING_CONFIGS.items():
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epochs = config['epochs']

    loader = Loader(batch_size = batch_size)
    loader.hydrate()

    configs = {
      'model': {
        'vocab_size': len(loader.word_index_map)
      },
      'variation': variation,
      'loader': loader.training_loader,
      'learning_rate': learning_rate,
      'epochs': epochs
    }

    trainer = Trainer()
    model, loss_history = trainer.train(variation, configs)

    # benchmark the currently trained model
    num_correct = 0
    num_samples = 0
  
    benchmarking_data = iter(loader.benchmarking_loader)
  
    predictions = list()
    labels = list()
  
    with torch.no_grad():
      while True:
        try:
          benchmarking_inputs, benchmarking_labels = benchmarking_data.next()
        except StopIteration:
          break
  
        _, result = model(benchmarking_inputs).max(1)
        predictions.extend(numpy.asarray(result))
        labels.extend(benchmarking_labels.detach().numpy())
    
    config['accuracy'] = Metrics.accuracy(predictions, labels)
    print(config)
