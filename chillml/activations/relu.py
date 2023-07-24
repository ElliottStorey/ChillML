import numpy as np

class Relu:
    def calculate(input):
        return np.maximum(0, input)

    def calculate_derivative(input):
        return (input > 0).astype(int)