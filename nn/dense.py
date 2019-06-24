from .module import Module
from autograd import Variable


class Dense(Module):
    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()
        self.weights = Variable.random(input_size, output_size)
        self.bias = Variable.random(output_size, sign='+')
        self._parameters = {
            'weights': self.weights,
            'bias': self.bias
        }

    def forward(self, input):
        """
        Evaluation function for dense layer
        :param input: input variable. Must be vector
        :return: Result of summation of weighted inputs and bias vector
        """
        out = input @ self.weights
        return self.bias + out
