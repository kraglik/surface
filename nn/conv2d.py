from functools import reduce
import numpy as np
from nn.util import conv2d
from .module import Module
from autograd import Variable
from typing import List


class Conv2D(Module):

    def __init__(self, kernel_sizes, in_channels, out_channels):
        super(Conv2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernels = [Variable.random(*kernel_sizes) for _ in range(out_channels)]
        self.biases = [Variable.random(*kernel_sizes) for _ in range(out_channels)]
        self._parameters.update({
            f"kernel_{i}": kernel for i, kernel in enumerate(self.kernels)
        })
        self._parameters.update({
            f"bias_{i}": bias for i, bias in enumerate(self.biases)
        })

    def forward(self, inputs: List[Variable]) -> List[Variable]:

        result = []

        for kernel in self.kernels:
            kernel_convs = [conv2d(x, kernel) for x in inputs]
            kernel_result = kernel_convs[0]
            for conv in kernel_convs[1:]:
                kernel_result = kernel_result + conv
            result.append(kernel_result)

        return result



