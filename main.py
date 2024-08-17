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

dense1 = Layer_Dense(2, 126)
dense2 = Layer_Dense(126, 512)
dense3 = Layer_Dense(512, 3)

#--------------------------feed forward-------------------------------


X, y = spiral_data(samples=10, classes=3)
dense1.forward(X)
relu.forward(dense1.output)
dense2.forward(relu.output)
relu.forward(dense2.output)
dense3.forward(relu.output)
sm.forward(dense3.output)


#---------------------------loss and accuracy-------------------------------------

loss = loss_fuction.calculate(sm.output,y)
acc = accuracy(sm.output, y)
print("Accuracy", acc)
print("Loss", loss)
print("output", sm.output)
print("truth", y)

