import torch
from sklearn.metrics import classification_report, accuracy_score
import numpy


class Metrics:

 @staticmethod
 def summary(predictions, labels) -> None:
   return classification_report(predictions, labels)
 
 @staticmethod
 def accuracy(predictions, labels) -> float:
   return accuracy_score(predictions, labels)
