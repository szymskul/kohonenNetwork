import ssl
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from ucimlrepo import fetch_ucirepo
import Kohonen

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

som_size = (10, 10)
input_len = X_processed.shape[1]
learning_rate = 0.5
sigma = 1.0
numberOfEpochs = 1000

scaler = MinMaxScaler()
data = scaler.fit_transform(X_processed)
positive_data = [0 for _ in range(21)]
for i in range(len(Y_processed)):
    positive_data[Y_processed[i][2]] += 1

som = Kohonen.Kohonen(som_size, input_len, sigma, learning_rate)

som.trainKohonen(data, numberOfEpochs)

som.classificationOfNeurons(data, Y_processed)

predicted = som.testKohonen(data)

'''
score = som.quantization_error(data)
print("The score")
print(score)'''
correct = 0
Y_target = []
for process in Y_processed:
    Y_target.append(process[2])

for i in range(649):
    if Y_target[i] == predicted[i]:
        correct += 1

correct = correct/649

print(correct)