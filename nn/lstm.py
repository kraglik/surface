from .module import Module
from autograd import Variable
import numpy as np
from .dense import Dense
from typing import List
import autograd.functional as F


class LSTM(Module):

    def __init__(self, input_size, output_size):
        super(LSTM, self).__init__()

        self.h_0 = Variable(np.zeros((output_size,)))
        self.c_0 = Variable(np.zeros((output_size,)))

        self.forget_gate = Dense(input_size + output_size, output_size)
        self.input_gate = Dense(input_size + output_size, output_size)
        self.candidate_gate = Dense(input_size + output_size, output_size)
        self.output_gate = Dense(input_size + output_size, output_size)

        self._parameters = {
            'forget_layer': self.forget_gate,
            'input_gate': self.input_gate,
            'candidate_gate': self.candidate_gate,
            'output_gate': self.output_gate
        }

    def forward(self, inputs: List[Variable]) -> List[Variable]:
        result = []

        h, c = self.h_0, self.c_0

        for input in inputs:
            inp = Variable.concat(h, input)

            forget_vector = F.sigmoid(self.forget_gate.forward(inp))
            c = forget_vector * c

            input_filter = F.sigmoid(self.input_gate.forward(inp))
            candidates = F.tanh(self.candidate_gate.forward(inp))

            c = c + (candidates * input_filter)

            h = F.tanh(c) * F.sigmoid(self.output_gate.forward(inp))

            result.append(h)

        return result

