import torch.utils.data as data


class IMDBDataset(data.Dataset):

  def __init__(self, inputs: list, labels: list):
    super(IMDBDataset).__init__()

    if len(inputs) != len(labels):
      raise AssertionError('Mismatch between labels and inputs')

    self.inputs = inputs
    self.labels = labels

  def __getitem__(self, idx):
    return self.inputs[idx], self.labels[idx]

  def __len__(self):
    return len(self.inputs)
