IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
IMDB_FILE = 'imdb.{}.pkl.gz'

PAD = '.pad'
START = '.start'
END = '.end'
UNK = '.unk'

DEFAULT_DATA_BATCH_SIZE = 100

MODEL_VARIATION_MLP = 'mlp'
MODEL_VARIATION_ELMAN = 'elman'
MODEL_VARIATION_PYTORCH_ELMAN = 'pytorch-elman'
MODEL_VARIATION_PYTORCH_LSTM = 'pytorch-lstm'

MODEL_VARIATIONS = [
  MODEL_VARIATION_MLP,
  MODEL_VARIATION_ELMAN,
  # TODO: Add PyTorch LSTM, ELMAN
  # MODEL_VARIATION_PYTORCH_ELMAN,
  # MODEL_VARIATION_PYTORCH_LSTM,
]

TRAINING_CONFIGS = {
    1: {
      'learning_rate': 0.002,
      'batch_size': 100,
      'epochs': 1
    },
    2: {
      'learning_rate': 0.003,
      'batch_size': 100,
      'epochs': 1
    },
    3: {
      'learning_rate': 0.004,
      'batch_size': 100,
      'epochs': 1
    },
    4: {
      'learning_rate': 0.005,
      'batch_size': 100,
      'epochs': 1
    },
    5: {
      'learning_rate': 0.007,
      'batch_size': 100,
      'epochs': 1
    },
    6: {
      'learning_rate': 0.009,
      'batch_size': 100,
      'epochs': 1
    },
    8: {
      'learning_rate': 0.01,
      'batch_size': 100,
      'epochs': 1
    },
    9: {
      'learning_rate': 0.0125,
      'batch_size': 100,
      'epochs': 1
    },
    10: {
      'learning_rate': 0.025,
      'batch_size': 100,
      'epochs': 1
    }
}
