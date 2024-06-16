import ssl

import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo
import Kohonen

ssl._create_default_https_context = ssl._create_unverified_context

print("Co chcesz zrobic?")
print("1. Klasyfikacja za pomoca sieci kohonena wraz z wlasnym klasyfikatorem")
print("2. Redukcja wymiarow za pomoca sieci kohonena i klasyfikacja za pomoca implementacji MLP")
print("3. Klasyfikacja MLP")
choose = int(input("Podaj wybór: "))
if choose == 1:

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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    som_size = (10, 10)
    input_len = X.shape[1]
    learning_rate = 0.5
    sigma = 0.5
    numberOfEpochs = 1000

    som = Kohonen.Kohonen(som_size, input_len, sigma, learning_rate)

    score = som.quantization_error(X_train)
    print("Quantization before")
    print(score)

    som.classificationOfNeurons(X_train, y_train)

    predicted_before = som.testKohonen(X_test)

    correct = 0
    medium_level = 0
    Y_target = []

    for process in y_test:
        Y_target.append(process[0])
    for i in range(len(y_test)):
        if Y_target[i] == predicted_before[i]:
            correct += 1
        elif Y_target[i] == predicted_before[i] - 1 or Y_target[i] == predicted_before[i] + 1:
            medium_level += 1

    medium_level = medium_level / len(y_test) * 100

    correct = correct/len(y_test) * 100
    print(f"Całkowity procent poprawnie rozpoznanych przypadków {correct}%")
    print(f"Calkowity procent srednio poprawnie rozpoznanych przypadkow {medium_level}%")
    print(f"Calkowity procent zle rozpoznanych przypadkow {100 - correct - medium_level}%")

    som.trainKohonen(X_train, numberOfEpochs)

    som.classificationOfNeurons(X_train, y_train)

    predicted = som.testKohonen(X_test)

    score = som.quantization_error(X_train)
    print("Quantization after")
    print(score)
    correct = 0
    medium_level = 0
    Y_target = []

    for process in y_test:
        Y_target.append(process[0])
    for i in range(len(y_test)):
        if Y_target[i] == predicted[i]:
            correct += 1
        elif Y_target[i] == predicted[i] - 1 or Y_target[i] == predicted[i] + 1:
            medium_level += 1

    matrix = confusion_matrix(Y_target, predicted)
    print(matrix)

    print("Accuracy:", accuracy_score(Y_target, predicted))
    print(classification_report(Y_target, predicted, zero_division=1))

    medium_level = medium_level / len(y_test) * 100
    correct = correct/len(y_test) * 100
    print(f"Całkowity procent poprawnie rozpoznanych przypadków {correct}%")
    print(f"Calkowity procent srednio poprawnie rozpoznanych przypadkow {medium_level}%")
    print(f"Calkowity procent zle rozpoznanych przypadkow {100 - correct - medium_level}%")

elif choose == 2:

    wine = fetch_ucirepo(id=186)
    X = wine.data.features
    Y = wine.data.targets['quality'].values.ravel()

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    som_shape = (7, 7)
    som = MiniSom(som_shape[0], som_shape[1], X.shape[1], sigma=0.5, learning_rate=0.5)
    before_quant = som.quantization_error(X_normalized)
    som.random_weights_init(X_normalized)
    print(f"Before training: {before_quant}")
    som.train_random(X_normalized, 100)
    after_quant = som.quantization_error(X_normalized)
    print(f"After training: {after_quant}")
    mapped_data = np.array([som.winner(x) for x in X_normalized])

    X_train, X_test, y_train, y_test = train_test_split(mapped_data, Y, test_size=0.3, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    print("Matrix")
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=1))

    correct = 0
    medium_level = 0

    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct += 1
        elif y_test[i] == y_pred[i] - 1 or y_test[i] == y_pred[i] + 1:
            medium_level += 1

    medium_level = medium_level / len(y_pred) * 100
    correct = correct/len(y_pred) * 100
    print(f"Całkowity procent poprawnie rozpoznanych przypadków {correct}%")
    print(f"Calkowity procent srednio poprawnie rozpoznanych przypadkow {medium_level}%")
    print(f"Calkowity procent zle rozpoznanych przypadkow {100 - correct - medium_level}%")

elif choose == 3:
    wine = fetch_ucirepo(id=186)
    X = wine.data.features
    Y = wine.data.targets['quality'].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(14, 15, 14), max_iter=1000)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    print("Matrix")
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=1))

    correct = 0
    medium_level = 0

    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct += 1
        elif y_test[i] == y_pred[i] - 1 or y_test[i] == y_pred[i] + 1:
            medium_level += 1

    medium_level = medium_level / len(y_pred) * 100
    correct = correct / len(y_pred) * 100
    print(f"Całkowity procent poprawnie rozpoznanych przypadków {correct}%")
    print(f"Calkowity procent srednio poprawnie rozpoznanych przypadkow {medium_level}%")
    print(f"Calkowity procent zle rozpoznanych przypadkow {100 - correct - medium_level}%")