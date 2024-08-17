import numpy as np

class Activation_RELU:
  def __init__(self):
    self.output = None

  def forward(self, inp):
    self.output = np.maximum(0,inp)

class Activation_Softmax:
  def __init__(self):
    self.output = None

  def forward(self, inp):
    # Get un normalized probabilities
    exp_values = np.exp(inp - np.max(inp,axis=1, keepdims=True))
    # Get probabilities
    probabilities = exp_values/np.sum(exp_values,axis=1, keepdims=True)
    self.output =  probabilities