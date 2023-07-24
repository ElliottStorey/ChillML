import numpy as np

class BinaryCrossEntropy:
    def calculate(actual_output, predicted_output):
        predicted_output = np.clip(predicted_output, 1e-15, 1 - 1e-15)
        return -1/len(actual_output) * np.sum(actual_output * np.log(predicted_output) + (1 - actual_output) * np.log(1 - predicted_output))
    def calculate_derivative(actual_output, predicted_output):
        predicted_output = np.clip(predicted_output, 1e-15, 1 - 1e-15)
        return 1/len(actual_output) * (predicted_output - actual_output) / (predicted_output * (1 - predicted_output))