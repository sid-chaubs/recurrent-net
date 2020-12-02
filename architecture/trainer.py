from architecture.networks.v1 import Network as Base
from architecture.networks.v2 import Network as Elman
from architecture.checkpoint import Checkpoint

import torch
import torch.utils.data as data
import torch.optim as optim


class Trainer:

  @staticmethod
  def train(training_configs: data.dataloader, model_configs: int, save: bool = True) -> [object, list]:
    loader = training_configs['loader']
    epochs = training_configs['epochs']
    learning_rate = training_configs['learning_rate']
  
    model = Elman(**model_configs)

    # update this
    loss_function = torch.nn.CrossEntropyLoss()
    loss_history = list()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

    for current_epoch in range(epochs):
      print(f'Currently running on epoch {current_epoch}')

      iterable = iter(loader)
      mean_loss = 0

      # loop over the training data in batches of size defined previously
      while True:
        try:
          training_images, training_labels = iterable.next()
        except StopIteration:
          break

        optimizer.zero_grad()
        output = model(training_images)
        loss = loss_function(output, training_labels)
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()

        print('Loss for current training batch: ', loss.item())

      mean_loss /= loader.batch_size
      loss_history.append(mean_loss)
      
    # save once we have trained the model over the given number of epochs
    if save:
      checkpoint = Checkpoint(model, loss_history, epochs, learning_rate, loader.batch_size, optimizer)
      checkpoint.save()

    return model, loss_history
