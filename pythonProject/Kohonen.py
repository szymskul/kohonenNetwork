from sklearn.decomposition import PCA
from minisom import MiniSom
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Kohonen():

    def __init__(self, som_size, input_len, sigma, learning_rate):
        self.input_len = input_len
        self.som = MiniSom(x=som_size[0], y=som_size[1], input_len=input_len, sigma=sigma, learning_rate=learning_rate)
        self.som_size = som_size

    def trainKohonen(self, data, numberOfEpochs):
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)

        self.som.random_weights_init(data)

        weights = self.som.get_weights().reshape(-1, self.input_len)
        scaler_weights = MinMaxScaler(feature_range=(-1, 1))
        scaled_weights = scaler_weights.fit_transform(weights).reshape(self.som_size[0], self.som_size[1], self.input_len)
        self.som.weights = scaled_weights

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

        reduced_weights = pca.transform(scaled_weights.reshape(-1, self.input_len))

        plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1])
        plt.title("Wizualizacja sieci Kohonena i instancji zbioru treningowego")
        plt.show()

        self.som.train_random(data, numberOfEpochs)

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

        weights_after = self.som.get_weights().reshape(-1, self.input_len)
        reduced_weights_after = pca.transform(weights_after)

        plt.scatter(reduced_weights_after[:, 0], reduced_weights_after[:, 1])
        plt.title("Wizualizacja 2D sieci Kohonena i instancji zbioru treningowego")

        plt.show()

    def quantization_error(self, data):
        quantization_error = []
        for example in data:
            bmu = self.som.winner(example)
            quantization_error.append(np.linalg.norm(example - self.som.get_weights()[bmu[0], bmu[1]]))
        return np.mean(quantization_error)

