import numpy as np

class Sigmoid:
    def calculate(input):
        return 1 / (1 + np.exp(-input))
    
    def calculate_derivative(input):
        return calculate(input) * (1 - calculate(input))