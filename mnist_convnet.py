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

        self.c1 = nn.Conv2D((6, 6), 1, 10)
        self.c2 = nn.Conv2D((6, 6), 10, 15)
        self.c3 = nn.Conv2D((4, 4), 15, 20)
        self.c4 = nn.Conv2D((3, 3), 20, 25)

        self.lstm = nn.LSTM(13 * 13, 13 * 13)

        self.dense = nn.Dense(13 * 13, 10)

        self._parameters = {
            'conv1': self.c1,
            'conv2': self.c2,
            'conv3': self.c3,
            'conv4': self.c4,
            'lstm': self.lstm,
            'dense': self.dense,
        }

    def forward(self, x):

        x = [F.tanh(x) for x in self.c1.forward(x)]
        x = [F.relu(x) for x in self.c2.forward(x)]
        x = [F.tanh(x) for x in self.c3.forward(x)]
        x = [F.relu(x).flatten() for x in self.c4.forward(x)]

        x = self.lstm.forward(x)[-1]
        x = F.softmax(self.dense.forward(x))

        return x


if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})

    train = mnist.train_images()
    train_labels = mnist.train_labels()
    test = mnist.test_images()
    test_labels = mnist.test_labels()

    model = Model()

    model.trainer = optim.Adam(model=model)

    indices = [x for x in range(len(train))]

    for epoch in range(5):
        random.shuffle(indices)

        print(f'--[ EPOCH {epoch + 1} ]--')
        s = 0

        for i in indices:

            print(f'[{"%3.1f" % (100 * s / len(train))} %]', end='\r')
            sys.stdout.flush()

            s += 1

            if i % 100 == 0:
                inputs = [Variable(np.array(train[i], dtype=np.float32) / 255)]
                target = np.zeros(10)
                target[train_labels[i]] = 1
                model.trainer.fit([inputs], target)

                with open('mnist.model', 'wb') as f:
                    pickle.dump(file=f, obj=model)

    print('\n', end='\r')
    sys.stdout.flush()

    errors = 0

    for i in range(len(test)):

        inputs = (np.array(test[i], dtype=np.float32) / 255).flatten()
        output = model.forward(inputs).data.argmax()

        if output != test_labels[i]:
            errors += 1

        if i % 500 == 0:
            print(f"Network output: {output}, target: {test_labels[i]}")

    print('#'*50)
    print(f'Accuracy: {"%.2f" % (100 * (len(test) - errors) / len(test))}%')



