import torch
from sklearn.metrics import classification_report
import numpy


class Metrics:

 @staticmethod
 def summary(predicted, labels) -> None:
   return classification_report(predicted, labels)
