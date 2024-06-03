from random import choice

import numpy as np


class Network:
    neurons: list
    data: list
    N: int  # neighborhood
    alpha: float  # learning rate
    m: int
    n: int

    def __init__(self, m, n, input_size):
        self.N = 3
        self.m = m
        self.n = n
        self.alpha = 0.1
        #self.neurons = [[np.array([10 * i / (m - 1), 10 * j / (n - 1)]) for i in range(m)] for j in range(n)]
        self.neurons = [[np.random.rand(input_size) for _ in range(m)] for _ in range(n)]

    def Closest(self, point):
        minDist = 1e10
        closest = []
        for j, row in enumerate(self.neurons):
            for i, neuron in enumerate(row):
                neuron = self.neurons[j][i]
                dist = np.linalg.norm(neuron - point)
                if dist < minDist:
                    minDist = dist
                    closest = [i, j, neuron]
        return closest

    def Update(self, t, T):
        N = 1 + int((self.N - 1) * (1 - t / T))
        alpha = self.alpha * (1 - t / T)

        point = choice(self.data)
        closest = self.Closest(point)
        i, j, _ = closest

        left = max(i - N, 0)
        right = min(i + N, self.m - 1) + 1
        bottom = max(j - N, 0)
        top = min(j + N, self.n - 1) + 1

        for y in range(bottom, top):
            for x in range(left, right):
                self.neurons[y][x] += alpha * (point - self.neurons[y][x])

    def Train(self, T, data):
        self.data = np.array(data)
        for epoch in range(T):
            self.Update(epoch, T)

    def Unpack(self):
        unpacked = []
        for row in self.neurons:
            for neuron in row:
                unpacked.append(neuron)

        return unpacked
