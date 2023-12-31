from chillml import Network
from chillml.layers import FullyConnected, Activation
from chillml.losses import MeanSquaredError
from chillml.activations import Tanh

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

# Network
net = Network([
    FullyConnected(28*28, 100),
    Activation(Tanh),
    FullyConnected(100, 50),
    Activation(Tanh),
    FullyConnected(50, 10),
    Activation(Tanh)
], MeanSquaredError)

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.train(x_train[0:1000], y_train[0:1000], epochs=300, learning_rate=0.1)

# test on 3 samples
out = net.forward(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])
