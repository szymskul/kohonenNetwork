import ssl

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from Kohonen import Network
from plot import *
from random import seed
ssl._create_default_https_context = ssl._create_unverified_context

seed(0)

from ucimlrepo import fetch_ucirepo

# fetch dataset
student_performance = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = student_performance.data.features
y = student_performance.data.targets

scaler = MinMaxScaler(feature_range=(0, 1))
scaler_data = scaler.fit_transform(X)
data_list = [{'x': point[0], 'y': point[1]} for point in scaler_data]


network = Network(5, 5)
network.data = data_list

plot = Plot()
plot.Scatter(data_list)
plot.Scatter(network.Unpack())
plot.Show()

network.Train(10000)

plot.Scatter(data_list)
plot.Scatter(network.Unpack())
plot.Show()