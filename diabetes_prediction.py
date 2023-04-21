import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Cargar los datos y mostrar información básica
diabetes_dataset = pd.read_csv("diabetes.csv")
print(diabetes_dataset.shape)
print(diabetes_dataset.describe())

# Visualizar el balance de clases en el dataset
sns.countplot(x='Outcome', data=diabetes_dataset)
plt.title('Cantidad de pacientes diabéticos y no diabéticos')
plt.xlabel('Estado')
plt.ylabel('Cantidad')
plt.show()

# Visualizar la matriz de correlación
corr = diabetes_dataset.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Separar el dataset en features de entrada (X) y variable target (y)
X = diabetes_dataset.drop('Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Estandarizar los datos de entrada con StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Separar el dataset estandarizado en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, stratify=y, random_state=2)

# Entrenar un clasificador SVM con kernel lineal en el conjunto de entrenamiento
classifier = SVC(kernel='linear', random_state=2)
classifier.fit(X_train, y_train)

# Realizar predicciones en los conjuntos de entrenamiento y prueba, y calcular la exactitud
y_train_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print('Exactitud en el conjunto de entrenamiento:', train_accuracy)

y_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Exactitud en el conjunto de prueba:', test_accuracy)

# Realizar una predicción sobre nuevos datos usando el clasificador y escalador entrenados
new_input = np.array([1, 89, 66, 23, 94, 28.1, 0.167, 21]).reshape(1, -1)
new_input_scaled = scaler.transform(new_input)
new_prediction = classifier.predict(new_input_scaled)

# Mostrar el resultado de la predicción
if new_prediction[0] == 0:
    print('La persona no es diabética')
else:
    print('La persona es diabética')




