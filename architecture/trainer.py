import torch
from architecture.factory.networks import Networks
from architecture.factory.optimizers import Optimizers
from architecture.checkpoint import Checkpoint
from architecture.constants import *
from architecture.data.loader import Loader
from architecture.metrics import Metrics
import numpy


class Trainer:


  @staticmethod
  def benchmark(model, batch_size = 100, final = False):
    loader = Loader(batch_size = batch_size, final = final)
    loader.hydrate()

    benchmarking_data = iter(loader.benchmarking_loader)
    predictions = list()
    labels = list()
  
    with torch.no_grad():
      while True:
        try:
          benchmarking_inputs, benchmarking_labels = benchmarking_data.next()
        except StopIteration:
          break
      
        if model.variation in [MODEL_VARIATION_ELMAN, MODEL_VARIATION_PYTORCH_ELMAN]:
          output, hidden = model(benchmarking_inputs)
          _, result = output.max(1)
        else:
          _, result = model(benchmarking_inputs).max(1)
      
        predictions.extend(numpy.asarray(result))
        labels.extend(benchmarking_labels.detach().numpy())
  
      if final:
        print('Test results: ', Metrics.accuracy(predictions, labels))
      else:
        print('Validation results: ', Metrics.accuracy(predictions, labels))
  
  
  @staticmethod
  def train(model_variation: str, configs: dict, save: bool = False) -> [object, list]:
    print('Training models for config: ', str(configs))

    epochs = configs['epochs']
    loader = configs['loader']
    learning_rate = configs['learning_rate']

    model = Networks.generate(model_variation, configs['model'])
    optimizer = Optimizers.generate(model, learning_rate)

    loss_function = torch.nn.CrossEntropyLoss()
    loss_history = list()

    hidden = None

    for current_epoch in range(epochs):
      print('Currently on epoch:', (current_epoch + 1), ', model variation: ', model_variation)
      iterable = iter(loader)
      mean_loss = 0

      # loop over the training data in batches of size defined previously
      while True:
        try:
          training_inputs, training_labels = iterable.next()
        except StopIteration:
          break

        optimizer.zero_grad()

        if model.variation in [MODEL_VARIATION_ELMAN, MODEL_VARIATION_PYTORCH_ELMAN]:
          output, hidden = model(training_inputs)
        else:
          output = model(training_inputs)

        loss = loss_function(output, training_labels)
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()
 
      mean_loss /= loader.batch_size
      loss_history.append(mean_loss)
    
      # benchmark for the current epoch
      Trainer.benchmark(model, final = True)
    
    # save once we have trained the model over the given number of epochs
    if save:
      checkpoint = Checkpoint(model, loss_history, epochs, learning_rate, loader.batch_size, optimizer)
      checkpoint.save()
    
    Trainer.benchmark(model, final = True)

    return model, loss_history
