import numpy as np
from typing import Optional, List, Callable, Union


class Variable:

    def __init__(self, data: Union[float, list, np.ndarray]):
        self.data = data
        if isinstance(self.data, float): self.data = [self.data]
        if isinstance(self.data, list): self.data = np.array(self.data, dtype=np.float32)
        if self.data.dtype != np.float32: self.data = self.data.astype(np.float32)
        self.children: list = []
        self.parents: List['Variable'] = []
        self.grad_value: Optional[np.ndarray] = None
        self.backward_done: bool = False

    def __str__(self):
        return f"Variable (\n{self.data}\n)"

    def unary_op(self, out_data: np.array, w, apply: Callable = lambda grad, w: grad * w) -> 'Variable':
        out = Variable(out_data)
        self.children.append((w, out, apply))
        out.parents = [self]
        return out

    def binary_op(self, other: 'Variable', out_data, self_data, other_data) -> 'Variable':
        if not isinstance(self_data, tuple): self_data = (self_data, lambda grad, w: grad * w)
        if not isinstance(other_data, tuple): other_data = (other_data, lambda grad, w: grad * w)
        out = Variable(out_data)
        self.children.append((self_data[0], out, self_data[1]))
        other.children.append((other_data[0], out, other_data[1]))
        out.parents = [self, other]
        return out

    def __neg__(self) -> 'Variable':
        return self.unary_op(-self.data, -1.0, lambda grad, w: grad * w)

    def __mul__(self, other: 'Variable') -> 'Variable':
        return self.binary_op(other, self.data * other.data, other.data, self.data)

    def __truediv__(self, other: 'Variable') -> 'Variable':
        return self.binary_op(other, self.data / other.data, 1 / other.data, - self.data / (other.data ** 2))

    def __add__(self, other: 'Variable') -> 'Variable':
        return self.binary_op(other, self.data + other.data, 1.0, 1.0)

    def __sub__(self, other: 'Variable') -> 'Variable':
        return self.binary_op(other, self.data - other.data, 1.0, -1.0)

    def sin(self) -> 'Variable':
        return self.unary_op(np.sin(self.data), np.cos(self.data))

    def cos(self) -> 'Variable':
        return self.unary_op(np.cos(self.data), -np.sin(self.data))

    def dot(self, other: 'Variable') -> 'Variable':
        return self.binary_op(other, np.dot(self.data, other.data), other.data, self.data)

    def sum(self) -> 'Variable':
        return self.unary_op(np.sum(self.data), np.ones(self.data.shape))

    def max(self) -> 'Variable':
        pos = self.data.argmax()
        w = np.zeros(self.data.shape); w[pos] = 1.0
        return self.unary_op(self.data.max(), w)

    def min(self) -> 'Variable':
        pos = self.data.argmin()
        w = np.zeros(self.data.shape); w[pos] = 1.0
        return self.unary_op(self.data.min(), w)

    def __pow__(self, power, modulo=None) -> 'Variable':
        return self.unary_op(self.data ** power, (self.data ** (power - 1)) * power)

    def log(self) -> 'Variable':
        return self.unary_op(np.log(self.data), np.power(self.data, -1.0))

    def log2(self) -> 'Variable':
        return self.unary_op(np.log2(self.data), np.log(np.ones(self.data.shape) * 2.0))

    def log10(self) -> 'Variable':
        return self.unary_op(np.log10(self.data), np.log(np.ones(self.data.shape) * 10.0))

    def exp(self) -> 'Variable':
        out_data = np.exp(self.data)
        return self.unary_op(out_data, out_data)

    def flatten(self) -> 'Variable':
        return self.unary_op(self.data.flatten(), 1.0, lambda grad, w: (grad * w).reshape(self.data.shape))

    def __matmul__(self, other: 'Variable') -> 'Variable':
        if len(self.data.shape) == 1 and len(other.data.shape) == 1:
            return self.dot(other)

        f_a = lambda grad, w: grad @ w.transpose()
        f_b = lambda grad, w: w.transpose() @ grad
        w_a = other.data
        w_b = self.data

        if len(self.data.shape) == 1 and len(other.data.shape) == 2:
            w_b = self.data.reshape((self.data.size, 1))
            f_b = lambda grad, w: w @ (grad.reshape((1, grad.size)))

        return self.binary_op(other, np.matmul(self.data, other.data), (w_a, f_a), (w_b, f_b))

    def grad(self) -> np.array:
        if self.grad_value is None: self.grad_value = sum(apply(var.grad(), w) for w, var, apply in self.children)
        return self.grad_value

    def backward(self):
        if self.backward_done:
            return
        if len(self.children) < 1:
            self.grad_value = 1.0
        else:
            self.grad()
        for parent in self.parents:
            parent.backward()
        self.backward_done = True

    def reset_grad(self):
        self.grad_value = None
        self.backward_done = False
        for parent in self.parents:
            if parent.grad_value is not None:
                parent.reset_grad()

    def reset_tree(self):
        parents = self.parents
        self.parents = []
        self.children = []
        for parent in parents:
            parent.reset_tree()

    def reset(self):
        self.reset_grad()
        self.reset_tree()

    @staticmethod
    def random(*shape, **kwargs):
        if 'sign' in kwargs.keys():
            if kwargs.get('sign') == '+':
                random_data = np.random.rand(*shape) * 0.15
            else:
                random_data = np.random.rand(*shape) * -0.15
        else:
            random_data = np.random.rand(*shape) * 0.3 - 0.15
        random_data[random_data == 0] = (np.random.random_sample() * 0.14 + 0.01) * (-1 ** np.random.randint(1, 2))
        return Variable(random_data.astype(np.float32))

    @staticmethod
    def concat(*variables, axis=0) -> 'Variable':

        data = np.concatenate([x.data for x in variables], axis=axis)
        out = Variable(data)
        out.parents = variables

        def concat_grad(i):
            sizes = [np.shape(a.data)[axis] for a in variables[:i + 1]]
            start = sum(sizes[:-1])
            idxs = [slice(None)] * data.ndim
            idxs[axis] = slice(start, start + sizes[-1])
            return lambda g: g[idxs]

        for i, v in enumerate(variables):
            grad_i = concat_grad(i)
            v.children.append((None, out, lambda g, w: grad_i(g)))

        return out

