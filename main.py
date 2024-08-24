from forward_pass import Layer_Dense
from activation import Activation_RELU, Activation_Softmax
from loss import Categorical_Cross_Entropy
from nnfs.datasets import spiral_data
from functions import *
import nnfs

#--------------------------initializing the class objects-------------------------------
nnfs.init()
relu = Activation_RELU()
sm = Activation_Softmax()
loss_fuction = Categorical_Cross_Entropy()


#--------------------------defining the network-------------------------------

dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 4)
dense3 = Layer_Dense(4, 3)

#--------------------------initializing values-------------------------------

X, y = spiral_data(samples=10, classes=3)
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense2_weights = dense2.weights.copy()
best_dense3_weights = dense3.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_biases = dense2.biases.copy()
best_dense3_biases = dense3.biases.copy()

#--------------------------feed forward-------------------------------

for iteration in range(100000):
    dense1.weights+=0.05*np.random.randn(2, 3)
    dense2.weights+=0.05*np.random.randn(3, 4)
    dense3.weights+=0.05*np.random.randn(4, 3)
    dense1.biases+=0.05*np.random.randn(1, 3)
    dense2.biases+=0.05*np.random.randn(1,4)
    dense3.biases+=0.05*np.random.randn(1,3)

    dense1.forward(X)
    relu.forward(dense1.output)
    dense2.forward(relu.output)
    relu.forward(dense2.output)
    dense3.forward(relu.output)
    sm.forward(dense3.output)


#---------------------------loss and accuracy-------------------------------------

    loss = loss_fuction.calculate(sm.output,y)
    acc = accuracy(sm.output, y)

    # print("output", sm.output)
    # print("truth", y)

#---------------------------random weight update-------------------------------------
    if loss < lowest_loss:
        best_dense1_weights = dense1.weights.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense3_weights = dense3.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_biases = dense2.biases.copy()
        best_dense3_biases = dense3.biases.copy()
        lowest_loss = loss
        print("Accuracy", acc)
        print("Loss", loss)
    else:
        dense1.weights = best_dense1_weights.copy()
        dense2.weights = best_dense2_weights.copy()
        dense3.weights = best_dense3_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.biases = best_dense2_biases.copy()
        dense3.biases = best_dense3_biases.copy()