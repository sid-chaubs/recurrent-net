from pathlib import Path
from architecture.networks.v1 import Network
import torch
import torch.optim


class Checkpoint:
  
  def __init__(self, model: Network, loss_history: list, epochs: int, learning_rate: float, batch_size: int, optimizer: torch.optim):
    self.model = model
    self.loss_history = loss_history
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.optimizer = optimizer

  def save(self):
    directory = Path(Checkpoint.get_directory(self.model.version))
  
    if not directory.exists():
      directory.mkdir(parents = True)

    filepath = Path(Checkpoint.get_path(self.epochs, self.learning_rate, self.batch_size, self.model.version))

    if filepath.exists():
      # delete the old file
      filepath.unlink()

    torch.save({
      'model_state': self.model.state_dict(),
      'epochs': self.epochs,
      'optimizer_state': self.optimizer.state_dict(),
      'loss_history': self.loss_history,
      'batch_size': self.batch_size,
      'learning_rate': self.learning_rate
    }, str(filepath.resolve()))

  @staticmethod
  def get_directory(version: str = ''):
    if version == '':
      return Path(f'../checkpoints').resolve()

    return Path(f'../checkpoints/{version}').resolve()

  @staticmethod
  def get_filename(epochs: int, learning_rate: float, batch_size: int):
    return f'epochs-{epochs}-learning-rate-{learning_rate}-batch-size-{batch_size}'

  @staticmethod
  def get_path(epochs: int, learning_rate: float, batch_size: int, version: str = '') -> str:
    return f'{Checkpoint.get_directory(version)}/{Checkpoint.get_filename(epochs, learning_rate, batch_size)}.point'
