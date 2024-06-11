import ssl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from ucimlrepo import fetch_ucirepo
import Kohonen
import MLP

ssl._create_default_https_context = ssl._create_unverified_context

wine = fetch_ucirepo(id=186)
X = wine.data.features
Y = wine.data.targets

if isinstance(X, pd.DataFrame):
    X = X.values

if isinstance(Y, pd.DataFrame):
    Y = Y.values

Y_processed = []
for i in range(len(Y)):
    Y_processed.append(Y[i])

scaler = MinMaxScaler(feature_range=(-1, 1))

X = scaler.fit_transform(X)

X_processed_kohonen_train = X[0:500]
X_processed_kohonen_test = X[500:600]
Y_processed_kohonen_train = Y[0:500]
Y_processed_kohonen_test = Y[500:600]

som_size = (20, 10)
input_len = X.shape[1]
learning_rate = 0.5
sigma = 1.0
numberOfEpochs = 1000

som = Kohonen.Kohonen(som_size, input_len, sigma, learning_rate)

score = som.quantization_error(X_processed_kohonen_train)
print("Quantization before")
print(score)

som.trainKohonen(X_processed_kohonen_train, numberOfEpochs)

som.classificationOfNeurons(X_processed_kohonen_train, Y_processed_kohonen_train)

predicted = som.testKohonen(X_processed_kohonen_test)

score = som.quantization_error(X_processed_kohonen_train)
print("Quantization after")
print(score)
correct = 0
Y_target = []

for process in Y_processed_kohonen_test:
    Y_target.append(process[0])
for i in range(len(Y_processed_kohonen_test)):
    if Y_target[i] == predicted[i]:
        correct += 1

Y_Set = []
for process in Y_processed:
    Y_Set.append(process[0])

correct = correct/len(Y_processed_kohonen_test) * 100
print(f"Całkowity procent poprawnie rozpoznanych przypadków kohonen {correct}%")

target_values = []
for number in Y_Set:
    if number == 0:
        target_values.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 1:
        target_values.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 2:
        target_values.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 3:
        target_values.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif number == 4:
        target_values.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif number == 5:
        target_values.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif number == 6:
        target_values.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif number == 7:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif number == 8:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif number == 9:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif number == 10:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


target_values = np.array(target_values)
combined_data = list(zip(X, target_values))
train_data = combined_data[0:500]
test_data = combined_data[500:600]

network = MLP.MLP(11, [12,15,12], 11, True)

train_error_list, correct_predictions_list, number_of_epoch = network.train(data_set=train_data, number_of_epochs=1000, momentum=0, learning_Rate=0.4, bias=True , test_data=test_data,shuffle=True, stop_value=None)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, number_of_epoch), train_error_list, linestyle='-', color='b')
plt.title('Błędy treningowe w zależności od liczby epok')
plt.xlabel('Liczba epok')
plt.ylabel('Błąd treningowy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, number_of_epoch), correct_predictions_list, linestyle='-', color='g')
plt.title('Dokładność w zależności od liczby epok')
plt.xlabel('Liczba epok')
plt.ylabel('Dokładność')
plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.show()

network.test(train_data, "test2")