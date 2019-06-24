from autograd import Variable
import numpy as np


def mse_loss(output: Variable, target: Variable):
    assert output.data.size == target.data.size, "Output and target sizes must be equal"
    error = (output - target)
    error = (error * error).sum()
    error = error * Variable(np.array([1.0 / output.data.size]))
    return error


def negative_log_loss(output: Variable, target: Variable):
    assert output.data.size == target.data.size, "Output and target sizes must be equal"
    error = (output - target)

