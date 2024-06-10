import random
import numpy as np

class neuron():

    def __init__(self, input_values_number, weights=None, biasTrue=None):
        self.input_values_number = input_values_number
        self.input_values = None
        if weights is None:
            self.weights = np.random.uniform(-1,1, size=input_values_number)
        else:
            self.weights = weights
        self.biasTrue = biasTrue
        if biasTrue:
            self.bias = random.uniform(-0.2,0.2)
        else:
            self.bias = 0
        self.output = None

    def add_weights(self):
        return np.dot(self.input_values, self.weights) + self.bias

    def sigmoid(self, calculation):
        return 1 / (1 + np.exp(-calculation))

    def derivSigmoid(self, s):
        return s * (1 - s)

    def activateFunction(self, values):
        self.input_values = np.array(values)
        neuron_calculation = self.add_weights()
        self.output = self.sigmoid(neuron_calculation)


