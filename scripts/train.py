from architecture.constants import TRAINING_CONFIGS
from architecture.data.loader import Loader
from architecture.trainer import Trainer
from architecture.checkpoint import Checkpoint

import os

for index, config in TRAINING_CONFIGS.items():
  learning_rate = config['learning_rate']
  batch_size = config['batch_size']
  epochs = config['epochs']

  loader = Loader(batch_size = batch_size)
  loader.hydrate()

  if not os.path.exists(Checkpoint.get_path(epochs, learning_rate, batch_size)):
    trainer = Trainer()
    model, loss_history = trainer.train(
      loader = loader.training_loader,
      vocab_size = len(loader.word_index_map),
      epochs = epochs,
      learning_rate = learning_rate,
      save = True
    )
