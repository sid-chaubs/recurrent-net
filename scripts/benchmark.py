import torch

from architecture.checkpoint import Checkpoint
from architecture.networks.v1 import Network
from architecture.metrics import Metrics
from architecture.constants import TRAINING_CONFIGS
from architecture.data.loader import Loader
import numpy

print('Downloading data...')
loader = Loader(final = False)

# if this is set to True, we will run the script using testing data else we will run it using validation data
loader.hydrate()
print('Download complete...')

output = ''

for key, config in TRAINING_CONFIGS.items():
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
  
    filepath = Checkpoint.get_path(epochs, learning_rate, batch_size)
    checkpoint = torch.load(filepath)
  
    # set the data loader
    data_loader = loader.benchmarking_loader

    # load the network
    model = Network(vocab_size = len(loader.word_index_map))
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)

    num_correct = 0
    num_samples = 0

    predicted = list()
    data = iter(data_loader)

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

    output += Metrics.summary(labels, predictions)


print(output)