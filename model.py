import os
from Layer_Dense import Layer_Dense
import json
import numpy as np
from activation import *

class Model:
    def __init__(self, input_shape=784):
        self.input_shape = input_shape
        self.layers = []
        self.architecture = {}

    def add(self, layer):
        self.layers.append(layer)

    def save(self, folder_path):
        """Save model weights, biases, and architecture"""
        os.makedirs(folder_path, exist_ok=True)

        # Save architecture information
        self.architecture = {
            'input_shape': self.input_shape,
            'layer_types': [layer.__class__.__name__ for layer in self.layers],
            'layer_shapes': []
        }

        # Save weights and biases for Dense layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Layer_Dense):
                np.save(os.path.join(folder_path, f'weights_{i}.npy'), layer.weights)
                np.save(os.path.join(folder_path, f'biases_{i}.npy'), layer.biases)
                self.architecture['layer_shapes'].append({
                    'weights': layer.weights.shape,
                    'biases': layer.biases.shape
                })
            else:
                self.architecture['layer_shapes'].append(None)

        # Save architecture information
        with open(os.path.join(folder_path, 'architecture.json'), 'w') as f:
            json.dump(self.architecture, f)

    @classmethod
    def load(cls, folder_path):
        """Load a saved model"""
        # Load architecture
        with open(os.path.join(folder_path, 'architecture.json'), 'r') as f:
            architecture = json.load(f)

        # Create new model
        model = cls(input_shape=architecture['input_shape'])

        # Reconstruct layers
        dense_layer_count = 0
        for layer_type, layer_shape in zip(architecture['layer_types'], architecture['layer_shapes']):
            if layer_type == 'Layer_Dense':
                layer = Layer_Dense(layer_shape['weights'][0], layer_shape['weights'][1])
                layer.weights = np.load(os.path.join(folder_path, f'weights_{dense_layer_count}.npy'))
                layer.biases = np.load(os.path.join(folder_path, f'biases_{dense_layer_count}.npy'))
                dense_layer_count += 1
            elif layer_type == 'Activation_ReLU':
                layer = Activation_ReLU()
            elif layer_type == 'Activation_Softmax':
                layer = Activation_Softmax()
            model.add(layer)

        return model

    def forward(self, X):
        """Forward pass through all layers"""
        current_output = X
        for layer in self.layers:
            layer.forward(current_output)
            current_output = layer.output
        return current_output

    def predict(self, X):
        """Make predictions"""
        self.forward(X)
        output = self.layers[-1].output
        return np.argmax(output, axis=1)