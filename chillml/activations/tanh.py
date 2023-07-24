import numpy as np

class Tanh:
    def calculate(input):
       return np.tanh(input)

    def calculate_derivative(input):
        return 1 - np.tanh(input) ** 2