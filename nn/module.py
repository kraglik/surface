from typing import List
from autograd import Variable
import itertools


class Module:
    def __init__(self):
        # parameters container is dictionary for sake of serialization simplicity and torch/pytorch interoperability
        self._parameters = dict()

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def __setattr__(self, key, value):
        if isinstance(value, Module):
            self._parameters.update({
                key + parameter_name: parameter
                for parameter_name, parameter
                in value._parameters.items()
            })

        super(Module, self).__setattr__(key, value)

    def forward(self, *inputs) -> Variable:
        """
        Module evaluation function
        :param inputs: Input data for module
        :return: result of application current module to data
        """
        raise NotImplemented()

    def backward(self):
        """
        Optional optimization helpers
        """
        # Doing nothing additional by default
        pass

    def parameters(self) -> List[Variable]:
        """
        :return: Flatten list of model parameters for usage in optimizers
        """
        parameters = []
        for key, parameter in self._parameters.items():
            if isinstance(parameter, Variable): parameters.append(parameter)
            elif isinstance(parameter, Module): parameters.extend(parameter.parameters())
        return parameters
