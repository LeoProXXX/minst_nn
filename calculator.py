import numpy as np


class Calculator:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_der(x):
        return Calculator.sigmoid(x) * (1 - Calculator.sigmoid(x))

    @staticmethod
    def softmax(x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    @staticmethod
    def softmax_der(x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
