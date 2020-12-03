from architecture.constants import *
from architecture.data.loader import Loader
from architecture.trainer import Trainer
from architecture.checkpoint import Checkpoint

import os

for variation in MODEL_VARIATIONS:
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

    if not os.path.exists(Checkpoint.get_path(epochs, learning_rate, batch_size, variation)):
      trainer = Trainer()
      model, loss_history = trainer.train(configs, save = True)
