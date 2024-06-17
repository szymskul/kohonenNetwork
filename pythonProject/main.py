import ssl

import numpy as np
import pandas as pd
import pandas as pd
import torch
import ssl
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from minisom import MiniSom
from sklearn.datasets import fetch_openml

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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    som_size = (8, 8)
    input_len = X.shape[1]
    learning_rate = 0.5
    sigma = 0.5
    numberOfEpochs = 1000

    som = Kohonen.Kohonen(som_size, input_len, sigma, learning_rate)

    score = som.quantization_error(X_train)
    print("Quantization before")
    print(score)

    som.classificationOfNeurons(X_train, y_train)

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

    matrix = confusion_matrix(Y_target, predicted, labels=range(11))
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

    som_shape = (8, 8)
    som = MiniSom(som_shape[0], som_shape[1], X.shape[1], sigma=0.5, learning_rate=0.5)
    before_quant = som.quantization_error(X_normalized)
    som.random_weights_init(X_normalized)
    print(f"Before training: {before_quant}")
    som.train_random(X_normalized, 100)
    after_quant = som.quantization_error(X_normalized)
    print(f"After training: {after_quant}")
    mapped_data = np.array([som.winner(x) for x in X_normalized])

    X_train, X_test, y_train, y_test = train_test_split(mapped_data, Y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 11)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    model = MLP()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    test_losses = []
    accuracies = []

    n_epochs = 1000

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            test_loss = criterion(outputs, y_test)
            test_losses.append(test_loss.item())

            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
            accuracies.append(accuracy)

    _, predicted = torch.max(outputs, 1)
    perfect = ((predicted == y_test).sum().item() / len(y_test)) * 100
    medium_level_minus = (predicted == y_test - 1).sum().item() / len(y_test)
    medium_level_plus = (predicted == y_test + 1).sum().item() / len(y_test)
    medium_level = (medium_level_minus + medium_level_plus) * 100

    matrix = confusion_matrix(y_test, predicted, labels=range(11))
    print(matrix)

    print("Accuracy:", accuracy_score(y_test, predicted))
    print(classification_report(y_test, predicted, zero_division=1))

    print(f"Całkowity procent poprawnie rozpoznanych przypadków {perfect}%")
    print(f"Calkowity procent srednio poprawnie rozpoznanych przypadkow {medium_level}%")
    print(f"Calkowity procent zle rozpoznanych przypadkow {100 - perfect - medium_level}%")

    # Wykres błędów i skuteczności w każdej epoce
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0,5)
    plt.legend()
    plt.title('Training and Test Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs + 1), accuracies, label='Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.show()


elif choose == 3:
    wine = fetch_ucirepo(id=186)
    X = wine.data.features
    Y = wine.data.targets['quality'].values.ravel()

    if isinstance(X, pd.DataFrame):
        X = X.values

    if isinstance(Y, pd.Series):
        Y = Y.values

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 11)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    model = MLP()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    test_losses = []
    accuracies = []

    n_epochs = 1000

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Oblicz błąd i skuteczność na zbiorze testowym
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            test_loss = criterion(outputs, y_test)
            test_losses.append(test_loss.item())

            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
            accuracies.append(accuracy)

    _, predicted = torch.max(outputs, 1)
    perfect = ((predicted == y_test).sum().item() / len(y_test)) * 100
    medium_level_minus = (predicted == y_test - 1).sum().item() / len(y_test)
    medium_level_plus = (predicted == y_test + 1).sum().item() / len(y_test)
    medium_level = (medium_level_minus + medium_level_plus) * 100

    matrix = confusion_matrix(y_test, predicted, labels=range(11))
    print(matrix)

    print(classification_report(y_test, predicted, zero_division=1))

    print(f"Całkowity procent poprawnie rozpoznanych przypadków {perfect}%")
    print(f"Calkowity procent srednio poprawnie rozpoznanych przypadkow {medium_level}%")
    print(f"Calkowity procent zle rozpoznanych przypadkow {100 - perfect - medium_level}%")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0,5)
    plt.legend()
    plt.title('Training and Test Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs + 1), accuracies, label='Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.show()