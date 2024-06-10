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

X_processed = preprocessor.fit_transform(X)

if isinstance(X_processed, pd.DataFrame):
    X_processed = X_processed.values

Y_processed = []
for i in range(len(Y)):
    Y_processed.append(Y.iloc[i].tolist())

X_processed_kohonen_train = np.concatenate((X_processed[0:150], X_processed[200:350], X_processed[400:550], X_processed[600:649]), axis=0)
X_processed_kohonen_test = np.concatenate((X_processed[150:200], X_processed[350:400], X_processed[550:600]), axis=0)
Y_processed_kohonen_train = np.concatenate((Y_processed[0:150], Y_processed[200:350], Y_processed[400:550], Y_processed[600:649]), axis=0)
Y_processed_kohonen_test = np.concatenate((Y_processed[150:200], Y_processed[350:400], Y_processed[550:600]), axis=0)

som_size = (20, 10)
input_len = X_processed.shape[1]
learning_rate = 0.5
sigma = 1.0
numberOfEpochs = 1000

som = Kohonen.Kohonen(som_size, input_len, sigma, learning_rate)

score = som.quantization_error(X_processed_kohonen_train)
print("The score")
print(score)

som.trainKohonen(X_processed_kohonen_train, numberOfEpochs)

som.classificationOfNeurons(X_processed_kohonen_train, Y_processed_kohonen_train)

predicted = som.testKohonen(X_processed_kohonen_test)

score = som.quantization_error(X_processed_kohonen_train)
print("The score")
print(score)
correct = 0
Y_target = []

for process in Y_processed_kohonen_test:
    Y_target.append(process[2])
for i in range(len(Y_processed_kohonen_test)):
    if Y_target[i] == predicted[i]:
        correct += 1

Y_Set = []
for process in Y_processed:
    Y_Set.append(process[2])

correct = correct/len(Y_processed_kohonen_test) * 100
print(correct)

target_values = []
for number in Y_Set:
    if number == 0:
        target_values.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 1:
        target_values.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 2:
        target_values.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 3:
        target_values.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 4:
        target_values.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 5:
        target_values.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 6:
        target_values.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 7:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 8:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 9:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 10:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 11:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 12:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif number == 13:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif number == 14:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif number == 15:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif number == 16:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif number == 17:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif number == 18:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif number == 19:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif number == 20:
        target_values.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

target_values = np.array(target_values)
combined_data = list(zip(X_processed, target_values))
train_data = combined_data[0:150] + combined_data[200:350] + combined_data[400:550] + combined_data[600:649]
test_data = combined_data[150:200] + combined_data[350:400] + combined_data[550:600]

network = MLP.MLP(43, [128,64,32], 21, True)

train_error_list, correct_predictions_list, number_of_epoch = network.train(data_set=train_data, number_of_epochs=1000, momentum=0, learning_Rate=0.6, test_data=test_data, bias=True ,shuffle=True, stop_value=None)
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