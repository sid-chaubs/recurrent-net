from pathlib import Path
from architecture.network import Network
import torch
import torch.optim
import os
import hashlib


class Checkpoint:
  
  def __init__(self, model: Network, loss_history: list, epochs: int, learning_rate: float, batch_size: int, optimizer: torch.optim):
    self.model = model
    self.loss_history = loss_history
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.optimizer = optimizer

  def save(self):
    directory = Checkpoint.get_directory()

    if not os.path.isdir(directory):
      os.mkdir(directory)

    filepath = Checkpoint.get_path(self.epochs, self.learning_rate, self.batch_size)
    if os.path.exists(filepath):
      # delete the old file
      os.remove(filepath)

    torch.save({
      'model_state': self.model.state_dict(),
      'epochs': self.epochs,
      'optimizer_state': self.optimizer.state_dict(),
      'loss_history': self.loss_history,
      'batch_size': self.batch_size,
      'learning_rate': self.learning_rate
    }, filepath)

  @staticmethod
  def get_directory():
    return Path('../checkpoints').resolve()

  @staticmethod
  def get_filename(epochs: int, learning_rate: float, batch_size: int):
    return f'epochs-{epochs}-learning-rate-{learning_rate}-batch-size-{batch_size}'

  @staticmethod
  def get_path(epochs: int, learning_rate: float, batch_size: int) -> str:
    return f'{Checkpoint.get_directory()}/{Checkpoint.get_filename(epochs, learning_rate, batch_size)}.point'
