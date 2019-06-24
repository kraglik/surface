from autograd import Variable
from nn.util import conv2d, deconv2d
import numpy as np


if __name__ == '__main__':
    a = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ])

    b = np.array([
        [0.25, 0.25],
        [0.25, 0.25]
    ])

    c = conv2d(Variable(a), Variable(b))
    d = deconv2d(c.data, b)

    A, B = Variable(a), Variable(d)

    C = Variable.concat(A, B)

    D = C.exp()

    E = D.sum()
    E.grad_value = 1.0
    E.backward()

    print(C)
    print(B.grad())

    print(c)
    print(d)
