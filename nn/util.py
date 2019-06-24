import numpy as np
from numpy.lib.stride_tricks import as_strided
from autograd import Variable


def conv2d(input: Variable, kernel: Variable) -> Variable:

    inp, ker = input.data, kernel.data

    if inp.ndim == 2:

        s = kernel.data.shape + tuple(np.subtract(inp.shape, ker.shape) + 1)

        data = np.einsum('ij,ijkl->kl', kernel.data, as_strided(input.data, shape=s, strides=input.data.strides * 2))

        return input.binary_op(kernel, data,
                               (kernel.data, lambda g, w: deconv2d(g, w)),
                               (kernel.data, lambda g, w: w * g.mean()))


def deconv2d(data, kernel):

    result = np.zeros(shape=tuple(np.add(data.shape, kernel.shape) - 1), dtype=np.float32)

    dx, dy = kernel.shape[0], kernel.shape[1]

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            result[x: x + dx, y: y + dy] += kernel * data[x, y]

    return result

