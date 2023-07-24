import numpy as np

class MeanSquaredError:
    def calculate(actual_output, predicted_output):
        return np.mean((actual_output - predicted_output) ** 2)
    
    def calculate_derivative(actual_output, predicted_output):
        return 2 * (predicted_output - actual_output) / len(actual_output)