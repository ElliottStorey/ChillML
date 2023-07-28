from chillml import Network
from chillml.layers import FullyConnected
from chillml.losses import MeanSquaredError
import numpy as np

'''
Sample Square

##
#*

Input: [[1, 1, 1, 0.2]]
Output: [[0.8]]
'''

training_inputs = [np.array([[np.random.random() for _ in range(4)]]) for _ in range(500)]
training_outputs = [np.array([[np.mean(square)]]) for square in training_inputs]
testing_inputs = [np.array([[np.random.random() for _ in range(4)]]) for _ in range(500)]
testing_outputs = [np.array([[np.mean(square)]]) for square in testing_inputs]

layers = [
    FullyConnected(4, 1)
]

network = Network(layers, MeanSquaredError)

prediction = network.forward(training_inputs[0])
print(f'''
Prediction: {np.squeeze(prediction)}
Actual: {np.squeeze(training_outputs[0])}
''')

network.train(training_inputs, training_outputs, 75, 0.01)

weights = network.layers[0].weights
print(f'''
Weights:
{weights}
''')

prediction = network.forward(training_inputs[0])
print(f'''
Prediction: {np.squeeze(prediction)}
Actual: {np.squeeze(training_outputs[0])}
''')