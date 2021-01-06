from neural_net import *


class NeuralNet2(NeuralNet):
    """
    It is the same neural net implementation but with additional second hidden layer
    """
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        super().__init__(sizes, epochs, l_rate)

    def initialize_parameters(self):
        """
        Method overridden
        """
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]    # additional second hidden layer
        output_layer = self.sizes[3]

        params = {
            'W1': np.random.randn(hidden_1, input_layer) / np.sqrt(1. / hidden_1),
            'W2': np.random.randn(hidden_2, hidden_1) / np.sqrt(1. / hidden_2),
            'W3': np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_propagation(self, x_train):
        """
        Method overridden
        """
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = Calculator.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = Calculator.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = Calculator.softmax(params['Z3'])

        return params['A3']

    def backward_propagation(self, y_train, output):
        """
        Method overridden
        """
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * Calculator.softmax_der(params['Z3'])
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * Calculator.sigmoid_der(params['Z2'])
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * Calculator.sigmoid_der(params['Z1'])
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w
