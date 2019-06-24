from autograd import Variable
import autograd.functional as F
import autograd.optimizers as optim
import nn
import numpy as np
import mnist
import pickle
import sys
import random


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(10, 10)
        self._parameters = {
            'lstm': self.lstm
        }

    def forward(self, inputs):
        return self.lstm(inputs)[-1]


if __name__ == '__main__':
    model = Model()

    optimizer = optim.Adam(model=model, lr=0.015)

    sequence = [Variable(np.random.random_sample(size=10)) for x in range(20)]

    for i in range(5000):
        s = sequence[0: random.randint(6, 20)]
        optimizer.fit([s], s[-3].data)

    print(sequence[-3], '\n', model.forward(sequence))
    print(sequence[:12][-3], '\n', model.forward(sequence[:12]))


