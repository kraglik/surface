import numpy as np
from .module import Module
from autograd import Variable


class MaxPool2D(Module):

    def __init__(self, sizes=(3, 3)):
        super(MaxPool2D, self).__init__()

        self.sizes = sizes

    def forward(self, inputs: Variable) -> Variable:
        pass



