import numpy as np
from .variable import Variable
from .functional import mse_loss


class Optimizer:
    def __init__(self, model):
        self.model = model
        self.parameters = model.parameters()

    def fit(self, inputs, target):
        raise NotImplemented()


class SGD(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.9, nesterov=False, loss=mse_loss):
        super(SGD, self).__init__(model)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.loss = loss
        self.cache = {parameter: np.zeros(parameter.data.shape) for parameter in self.parameters}

    def fit(self, inputs, target):
        if self.nesterov:
            for parameter in self.parameters:
                parameter.data -= self.cache[parameter] * self.momentum * self.lr
            out = self.model.forward(*inputs)
            error = self.loss(out, Variable(target))
            error.backward()
            for parameter in self.parameters:
                dw = parameter.grad()
                self.cache[parameter] *= self.momentum
                self.cache[parameter] += dw * (1.0 - self.momentum)
                parameter.data -= dw * (1.0 - self.momentum) * self.lr

        else:
            out = self.model.forward(*inputs)
            error = self.loss(out, Variable(target))
            error.backward()
            for parameter in self.parameters:
                dw = parameter.grad()
                self.cache[parameter] *= self.momentum
                self.cache[parameter] += dw * (1.0 - self.momentum)
                parameter.data -= self.cache[parameter] * self.lr

        error.reset()


class Adam(Optimizer):

    def __init__(self, model, loss=mse_loss, lr=0.001, b1=0.9, b2=0.999, eps=10 ** -8):
        super(Adam, self).__init__(model)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.loss = loss

        self.iteration = 0

        self.cache_m = {parameter: np.zeros(parameter.data.shape) for parameter in self.parameters}
        self.cache_v = {parameter: np.zeros(parameter.data.shape) for parameter in self.parameters}

    def fit(self, inputs, target):
        out = self.model.forward(*inputs)
        error = self.loss(out, Variable(target))
        error.backward()

        for parameter in self.parameters:
            g = parameter.grad()
            self.cache_m[parameter] = (1 - self.b1) * g + self.b1 * self.cache_m[parameter]
            self.cache_v[parameter] = (1 - self.b2) * (g ** 2) + self.b2 * self.cache_v[parameter]
            mhat = self.cache_m[parameter] / (1 - self.b1 ** (self.iteration + 1))
            vhat = self.cache_v[parameter] / (1 - self.b2 ** (self.iteration + 1))
            parameter.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

        error.reset()
        self.iteration += 1

