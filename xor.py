from autograd import Variable
import autograd.functional as F
import autograd.optimizers as optim
import nn
import numpy as np
import random


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layer_1 = nn.Dense(2, 8)
        self.layer_2 = nn.Dense(8, 8)
        self.layer_3 = nn.Dense(8, 1)

    def forward(self, x) -> Variable:
        x = F.relu(self.layer_1(x))
        x = F.tanh(self.layer_2(x))
        x = F.sigmoid(self.layer_3(x))

        return x


def main():

    model = Model()
    model.optim = optim.Adam(model)

    data = [
        (np.array([0, 0]), np.array(0)),
        (np.array([1, 0]), np.array(1)),
        (np.array([0, 1]), np.array(1)),
        (np.array([1, 1]), np.array(0))
    ]

    for i in range(1000):
        train_data = random.choices(data, k=4)

        for input, target in train_data:
            model.optim.fit([Variable(input)], target)

    for input, target in data:
        prediction = model.forward(Variable(input)).data.tolist()

        print(f"{input.tolist()} -> {prediction} (target: {target.tolist()})")


if __name__ == '__main__':
    main()
