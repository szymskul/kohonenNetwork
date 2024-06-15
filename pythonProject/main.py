import ssl
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo
import Kohonen

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

X_processed_kohonen_train = X[0:2000]
X_processed_kohonen_test = X[2000:2700]
Y_processed_kohonen_train = Y[0:2000]
Y_processed_kohonen_test = Y[2000:2700]

som_size = (20, 10)
input_len = X.shape[1]
learning_rate = 0.5
sigma = 1.0
numberOfEpochs = 1000

som = Kohonen.Kohonen(som_size, input_len, sigma, learning_rate)

score = som.quantization_error(X_processed_kohonen_train)
print("Quantization before")
print(score)

som.classificationOfNeurons(X_processed_kohonen_train, Y_processed_kohonen_train)

predicted_before = som.testKohonen(X_processed_kohonen_test)

correct = 0
medium_level = 0
Y_target = []

for process in Y_processed_kohonen_test:
    Y_target.append(process[0])
for i in range(len(Y_processed_kohonen_test)):
    if Y_target[i] == predicted_before[i]:
        correct += 1
    elif Y_target[i] == predicted_before[i] - 1 or Y_target[i] == predicted_before[i] + 1:
        medium_level += 1

medium_level = medium_level / len(Y_processed_kohonen_test) * 100

correct = correct/len(Y_processed_kohonen_test) * 100
print(f"Całkowity procent poprawnie rozpoznanych przypadków {correct}%")
print(f"Calkowity procent srednio poprawnie rozpoznanych przypadkow {medium_level}%")
print(f"Calkowity procent zle rozpoznanych przypadkow {100 - correct - medium_level}%")

som.trainKohonen(X_processed_kohonen_train, numberOfEpochs)

som.classificationOfNeurons(X_processed_kohonen_train, Y_processed_kohonen_train)

predicted = som.testKohonen(X_processed_kohonen_test)

score = som.quantization_error(X_processed_kohonen_train)
print("Quantization after")
print(score)
correct = 0
medium_level = 0
Y_target = []

for process in Y_processed_kohonen_test:
    Y_target.append(process[0])
for i in range(len(Y_processed_kohonen_test)):
    if Y_target[i] == predicted[i]:
        correct += 1
    elif Y_target[i] == predicted[i] - 1 or Y_target[i] == predicted[i] + 1:
        medium_level += 1

matrix = confusion_matrix(Y_target, predicted)
print(matrix)

medium_level = medium_level / len(Y_processed_kohonen_test) * 100
correct = correct/len(Y_processed_kohonen_test) * 100
print(f"Całkowity procent poprawnie rozpoznanych przypadków {correct}%")
print(f"Calkowity procent srednio poprawnie rozpoznanych przypadkow {medium_level}%")
print(f"Calkowity procent zle rozpoznanych przypadkow {100 - correct - medium_level}%")
