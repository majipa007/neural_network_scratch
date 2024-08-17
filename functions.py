import numpy as np



def accuracy(softmax_outputs, class_targets):
  predictions = np.argmax(softmax_outputs, axis=1)
  if len(class_targets) == 2:
    class_targets = np.argmax(class_targets, axis=1)
  acc = np.mean(predictions==class_targets)
  return acc