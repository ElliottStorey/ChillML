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
        
    def save(self, file_path):
        # Create a dictionary to store the layer weights and biases
        model_data = {'layers': [], 'loss': self.loss}

        for layer in self.layers:
            layer_data = {}
            # Check if the layer has weights and biases attributes
            if hasattr(layer, 'weights'):
                layer_data['weights'] = layer.weights
            if hasattr(layer, 'biases'):
                layer_data['biases'] = layer.biases

            model_data['layers'].append(layer_data)

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        self.loss = model_data['loss']

        for i, layer_data in enumerate(model_data['layers']):
            layer = self.layers[i]
            # Check if the layer has weights and biases attributes in the loaded model
            if hasattr(layer, 'weights') and 'weights' in layer_data:
                layer.weights = layer_data['weights']
            if hasattr(layer, 'biases') and 'biases' in layer_data:
                layer.biases = layer_data['biases']
