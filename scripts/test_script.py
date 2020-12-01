from architecture.constants import TRAINING_CONFIGS
from architecture.data.loader import Loader
from architecture.trainer import Trainer

print('Downloading data...')
loader = Loader()
loader.hydrate()
print('Download complete...')

# set this to false if you don't want to retrain your models for specific configs
retrain = True

for index, config in TRAINING_CONFIGS.items():
  learning_rate = config['learning_rate']
  batch_size = config['batch_size']
  epochs = config['epochs']

  print(f'Training network with training batch size: {batch_size} and learning rate: {learning_rate} over {epochs} epoch(s).')

  trainer = Trainer()
  model, loss_history = trainer.train(
    loader = loader.training_loader,
    vocab_size = len(loader.word_index_map),
    epochs = epochs,
    learning_rate = learning_rate,
    save = True
  )
