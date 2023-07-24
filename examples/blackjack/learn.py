from chillml import Network
from chillml.layers import FullyConnected, Activation
from chillml.losses import MeanSquaredError
from chillml.activations import Sigmoid, Softmax

layers = [
    FullyConnected(3, 10), # input: player sum, dealer sum, hit/stand
    Activation(Sigmoid),
    FullyConnected(10, 10),
    Activation(Sigmoid),
    FullyConnected(10, 1),
    Activation(Softmax) # output: win/lose
]

agent = Network(layers, MeanSquaredError)

agent.train()
agent.forward()