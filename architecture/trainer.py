import torch
from architecture.factory.networks import Networks
from architecture.factory.optimizers import Optimizers
from architecture.checkpoint import Checkpoint
from architecture.constants import *


class Trainer:
  
  @staticmethod
  def train(model_variation: str, configs: dict, save: bool = True) -> [object, list]:
    print('Training models for config: ', str(configs))

    epochs = configs['epochs']
    loader = configs['loader']
    learning_rate = configs['learning_rate']

    model = Networks.generate(model_variation, configs['model'])
    optimizer = Optimizers.generate(model, learning_rate)

    # TODO: Update this to use a factory method
    loss_function = torch.nn.CrossEntropyLoss()
    loss_history = list()

    for current_epoch in range(epochs):
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
 
      mean_loss /= loader.batch_size
      loss_history.append(mean_loss)
    
    # save once we have trained the model over the given number of epochs
    if save:
      checkpoint = Checkpoint(model, loss_history, epochs, learning_rate, loader.batch_size, optimizer)
      checkpoint.save()

    return model, loss_history
