import numpy as np

class Sigmoid:
    def calculate(input):
        return 1 / (1 + np.exp(-input))
    
    def calculate_derivative(input):
        return Sigmoid.calculate(input) * (1 - Sigmoid.calculate(input))