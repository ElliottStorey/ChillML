class Activation:
    def __init__(self, activation):
        self.activation = activation

    def forward(self, input):
        self.input = input
        return self.activation.calculate(input)

    def backward(self, output_error, learning_rate):
        return self.activation.calculate_derivative(self.input) * output_error