import pickle

class Network:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(self, inputs):
        outputs = []

        for input in inputs:
            for layer in self.layers:
                output = layer.forward(input)
                input = output
            outputs.append(output)
        return outputs

    def train(self, inputs, actual_outputs, epochs, learning_rate):
        # sample dimension first
        samples = len(inputs)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = inputs[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # compute loss (for display purpose only)
                err += self.loss.calculate(actual_outputs[j], output)

                # backward propagation
                error = self.loss.calculate_derivative(actual_outputs[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
        
    def save(self, file):
        with open(file, 'wb') as file:
            weights = []
            for layer in self.layers:
                try:
                    weights.append(layer.weights)
                except:
                    pass
            pickle.dump(weights, file)

    def load(self, file):
        with open(file, 'rb') as file:
            weights = pickle.load(file)
            for i, layer in enumerate(self.layers):
                try:
                    layer.weights = weights[i]
                except:
                    pass
