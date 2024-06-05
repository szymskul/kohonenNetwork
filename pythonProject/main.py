import ssl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from ucimlrepo import fetch_ucirepo

ssl._create_default_https_context = ssl._create_unverified_context

education = fetch_ucirepo(id=320)
X = education.data.features
Y = education.data.targets

binary_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
nominal_columns = ['Mjob', 'Fjob', 'reason', 'guardian']
numeric_columns = [col for col in X.columns if col not in binary_columns + nominal_columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_columns),
        ('bin', OneHotEncoder(drop='if_binary'), binary_columns),
        ('nom', OneHotEncoder(), nominal_columns)
    ]
)

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

if isinstance(X_processed, pd.DataFrame):
    X_processed = X_processed.values

som_size = (10, 10)
input_len = X_processed.shape[1]
learning_rate = 0.5
sigma = 1.0

scaler = MinMaxScaler()
data = scaler.fit_transform(X_processed)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

som = MiniSom(x=som_size[0], y=som_size[1], input_len=input_len, sigma=sigma, learning_rate=learning_rate)
som.random_weights_init(data)

weights = som.get_weights().reshape(-1, input_len)
scaler_weights = MinMaxScaler(feature_range=(-1, 1))
scaled_weights = scaler_weights.fit_transform(weights).reshape(som_size[0], som_size[1], input_len)
som.weights = scaled_weights

plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

reduced_weights = pca.transform(scaled_weights.reshape(-1, input_len))

plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1])
plt.title("Wizualizacja sieci Kohonena i instancji zbioru treningowego")
plt.show()

som.train_random(data, 10000)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

weights_after = som.get_weights().reshape(-1, input_len)
reduced_weights_after = pca.transform(weights_after)

plt.scatter(reduced_weights_after[:, 0], reduced_weights_after[:, 1])
plt.title("Wizualizacja 2D sieci Kohonena i instancji zbioru treningowego")

plt.show()