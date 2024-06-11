from sklearn.decomposition import PCA
from minisom import MiniSom
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Neuron():
    def __init__(self, weights, classification):
        self.weights = weights
        self.classification = classification
        self.classified = -1

    def chooseNeuronClass(self):
        max_index = self.classification.index(max(self.classification))
        self.classified = max_index

class Kohonen():

    def __init__(self, som_size, input_len, sigma, learning_rate):
        self.input_len = input_len
        self.som = MiniSom(x=som_size[0], y=som_size[1], input_len=input_len, sigma=sigma, learning_rate=learning_rate)
        self.som_size = som_size
        self.neurons = []

    def trainKohonen(self, data, numberOfEpochs):
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)

        self.som.random_weights_init(data)

        weights = self.som.get_weights().reshape(self.som_size[0] * self.som_size[1], self.input_len)
        scaler_weights = MinMaxScaler(feature_range=(-1, 1))
        scaled_weights = scaler_weights.fit_transform(weights).reshape(self.som_size[0], self.som_size[1], self.input_len)
        self.som.weights = scaled_weights

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

        reduced_weights = pca.transform(scaled_weights.reshape(self.som_size[0] * self.som_size[1], self.input_len))

        plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1])
        plt.title("Wizualizacja 2D sieci Kohonena i instancji zbioru treningowego")
        plt.show()

        self.som.train_random(data, numberOfEpochs)

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

        weights_after = self.som.get_weights()
        reduced_weights_after = pca.transform(weights_after.reshape(self.som_size[0] * self.som_size[1], self.input_len))

        plt.scatter(reduced_weights_after[:, 0], reduced_weights_after[:, 1])
        plt.title("Wizualizacja 2D sieci Kohonena i instancji zbioru treningowego")

        plt.show()

    def classificationOfNeurons(self, data, targets):
        bmu_list = []
        for example in data:
            bmu = self.som.winner(example)
            if bmu not in bmu_list:
                bmu_list.append(bmu)
        list_of_neurons = [[0] * 11 for _ in range(len(bmu_list))]
        for i in range(len(data)):
            example = data[i]
            target = targets[i]
            bmu = self.som.winner(example)
            bmu_index = bmu_list.index(bmu)
            list_of_neurons[bmu_index][target[0]] += 1
        for i in range(len(bmu_list)):
            self.neurons.append(Neuron(bmu_list[i], list_of_neurons[i]))
        for neuron in self.neurons:
            neuron.chooseNeuronClass()

    def quantization_error(self, data):
        quantization_error = []
        for example in data:
            bmu = self.som.winner(example)
            quantization_error.append(np.linalg.norm(example - self.som.get_weights()[bmu[0], bmu[1]]))
        return np.mean(quantization_error)

    def testKohonen(self, data):
        predicted = []
        isPredicted = False
        n = 2
        data_target = -1
        for example in data:
            bmu = self.som.winner(example)
            for neuron in self.neurons:
                if neuron.weights == bmu:
                    data_target = neuron.classified
                    isPredicted = True
                    break
            while(isPredicted == False):
                bmu = self.n_closest_neuron(example, n)
                for neuron in self.neurons:
                    if neuron.weights == bmu:
                        data_target = neuron.classified
                        isPredicted = True
                        break
                n = n + 1
            predicted.append(data_target)
            isPredicted = False

        return predicted

    def n_closest_neuron(self, data, n):
        weights = self.som.get_weights()
        distances = np.linalg.norm(weights - data, axis=-1)
        closest = []
        for _ in range(n):
            closest_idx = np.argmin(distances)
            closest.append(np.unravel_index(closest_idx, distances.shape))
            distances[closest[-1]] = np.inf

        return closest[-1]





