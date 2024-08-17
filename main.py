from forward_pass import Layer_Dense
from activation import Activation_RELU, Activation_Softmax
from loss import Categorical_Cross_Entropy
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()
relu = Activation_RELU()
sm = Activation_Softmax()
dense1 = Layer_Dense(2, 3)
loss_fuction = Categorical_Cross_Entropy()


X, y = spiral_data(samples=100, classes=3)
print(y.shape)
var = X.shape
dense1.forward(X)
relu.forward(dense1.output[0])
sm.forward(dense1.output[0])
print("dense output     ", dense1.output[0])
print("relu activated   ", relu.output)
print("softmax activated",sm.output)
relu.forward(dense1.output)
sm.forward(dense1.output)
loss_relu = loss_fuction.calculate(relu.output,y)
loss_sm = loss_fuction.calculate(sm.output,y)

print("relu:    ",loss_relu,"\nsoftmax: ", loss_sm)