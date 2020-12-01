import torch
from architecture.checkpoint import Checkpoint
from architecture.constants import TRAINING_CONFIGS
import matplotlib.pyplot as plt
import os


plot_id = 0
for key, config in TRAINING_CONFIGS.items():
  plot_id += 1
  
  epochs = config['epochs']
  learning_rate = config['learning_rate']
  batch_size = config['batch_size']

  filepath = Checkpoint.get_path(epochs, learning_rate, batch_size)
  if not os.path.exists(filepath):
    continue

  checkpoint = torch.load(filepath)
  y_label = 'y' + str(plot_id)

  plt.plot('x', y_label, data = checkpoint['loss_history'], marker = '', linewidth = 2)

#   plt.plot(checkpoint['loss_history'])
#   plt.title(f'Learning Rate: {learning_rate}, Batch Size: {batch_size}')
#   plt.ylabel('Loss')
#   plt.xlabel('Epochs')

plt.legend()
# plt.show()
