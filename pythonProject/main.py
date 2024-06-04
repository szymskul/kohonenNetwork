import ssl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo

ssl._create_default_https_context = ssl._create_unverified_context

wine_set = fetch_ucirepo(id=186)
data = wine_set.data.features

if isinstance(data, pd.DataFrame):
    data = data.values

som_size = (10, 10)
input_len = data.shape[1]
learning_rate = 0.5
sigma = 1.0

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

som = MiniSom(x=som_size[0], y=som_size[1], input_len=input_len, sigma=sigma, learning_rate=learning_rate)
som.random_weights_init(data)

weights = som.get_weights().reshape(-1, input_len)
scaler_weights = MinMaxScaler(feature_range=(-1, 1))
scaled_weights = scaler_weights.fit_transform(weights).reshape(som_size[0], som_size[1], input_len)
som.weights = scaled_weights

fig, ax = plt.subplots()
ax.scatter(reduced_data[:, 0], reduced_data[:, 1])

reduced_weights = pca.transform(scaled_weights.reshape(-1, input_len))

ax.scatter(reduced_weights[:, 0], reduced_weights[:, 1])
plt.show()

num_epochs = 20000
som.train_random(data, num_epochs)

fig, ax = plt.subplots()
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

weights_after = som.get_weights().reshape(-1, input_len)
reduced_weights_after = pca.transform(weights_after)

plt.scatter(reduced_weights_after[:, 0], reduced_weights_after[:, 1])

plt.show()