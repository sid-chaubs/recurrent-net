from architecture.constants import TRAINING_CONFIGS
from architecture.data.loader import Loader
from architecture.trainer import Trainer
from architecture.checkpoint import Checkpoint

import os

MODEL_VERSIONS = [
  'v.0.0.1',
  'v.0.0.2'
]

for version in MODEL_VERSIONS:
  for index, config in TRAINING_CONFIGS.items():
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epochs = config['epochs']
  
    loader = Loader(batch_size = batch_size)
    loader.hydrate()
  
    if not os.path.exists(Checkpoint.get_path(epochs, learning_rate, batch_size, version)):
      trainer = Trainer()
      
      training_configs = {
        'loader': loader.training_loader,
        'epochs': epochs,
        'learning_rate': learning_rate
      }
      
      model_configs = {
        'vocab_size': len(loader.word_index_map)
      }
  
      model, loss_history = trainer.train(
        training_configs,
        model_configs,
        save = True
      )
