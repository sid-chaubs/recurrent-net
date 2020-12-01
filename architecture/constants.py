IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
IMDB_FILE = 'imdb.{}.pkl.gz'

PAD = '.pad'
START = '.start'
END = '.end'
UNK = '.unk'


TRAINING_CONFIGS = {
  1: {
    'learning_rate': 0.002,
    'batch_size': 500,
    'epochs': 1
  },
  2: {
    'learning_rate': 0.003,
    'batch_size': 500,
    'epochs': 1
  },
  3: {
    'learning_rate': 0.004,
    'batch_size': 500,
    'epochs': 1
  },
  4: {
    'learning_rate': 0.005,
    'batch_size': 500,
    'epochs': 1
  },
  5: {
    'learning_rate': 0.007,
    'batch_size': 500,
    'epochs': 1
  },
  6: {
    'learning_rate': 0.009,
    'batch_size': 500,
    'epochs': 1
  },
  8: {
    'learning_rate': 0.01,
    'batch_size': 500,
    'epochs': 1
  },
  9: {
    'learning_rate': 0.0125,
    'batch_size': 500,
    'epochs': 1
  },
  10: {
    'learning_rate': 0.025,
    'batch_size': 500,
    'epochs': 1
  }
}

DEFAULT_BATCH_SIZE = 500

