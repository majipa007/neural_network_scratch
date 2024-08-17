import numpy as np

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.output = None
    self.weights = 0.01 * np.random.uniform(-1, 1, (n_inputs, n_neurons))
    self.biases = np.random.uniform(-1, 1,n_neurons)
  def forward(self, inputs):
    self.output = inputs@self.weights + self.biases
    return self.output