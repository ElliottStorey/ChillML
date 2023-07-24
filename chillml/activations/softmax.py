import numpy as np

class Softmax:
    def calculate(input):
        return np.exp(input) / np.sum(np.exp(input))

    def calculate_derivative(input):
        return Softmax.calculate(input) * (1 - Softmax.calculate(input))