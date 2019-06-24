from autograd import Variable
import numpy as np


def relu(x: Variable) -> Variable:
    data = x.data.copy()
    np.maximum(data, 0, data)
    w = data.copy()
    w[w > 0] = 1
    return x.unary_op(data, w)


def sigmoid(x: Variable) -> Variable:
    data = (np.exp(-x.data) + 1) ** (-1)
    w = -data + 1
    w[:] *= data[:]
    return x.unary_op(data, w)


def softmax(x: Variable) -> Variable:
    e = x.exp()
    return e / e.sum()


def tanh(x: Variable) -> Variable:
    data = np.tanh(x.data)
    w = 1 - data ** 2
    return x.unary_op(data, w)


def dropout(x: Variable, p=0.5) -> Variable:
    assert 0 <= p <= 1
    dp = np.zeros(x.data.shape, dtype=np.float32)
    dp[:] = 1.0 if np.random.random_sample() > p else 0.0
    return x * Variable(dp)

