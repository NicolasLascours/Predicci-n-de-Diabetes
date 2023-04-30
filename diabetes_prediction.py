#Importo librerías
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


#Lectura del dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

#Muestro los 5 primeros elementos de dataset
diabetes_dataset.head()

#cantidad de filas y columnas
diabetes_dataset.shape

#información estadística casos negativos
diabetes_dataset[diabetes_dataset['Outcome']==0].describe()

#información estadística casos positivos
diabetes_dataset[diabetes_dataset['Outcome']==1].describe()

#cantidad de diabéticos (1) y no diabéticos (0) en el dataset
diabetes_dataset['Outcome'].value_counts()

#Crear diagrama de barras
sns.countplot(x='Outcome', data=diabetes_dataset)
plt.title('Cantidad de pacientes diabéticos y no diabéticos')
plt.xlabel('Estado')
plt.ylabel('Cantidad')
plt.show()

#el % de glucosa en sangre en diabéticos es mayor que los que no lo son.
#A su vez, la personas de mayor edad son más suceptibles a tener diabetes.
diabetes_dataset.groupby('Outcome').mean()

# gráfico de dispersión que indica la relación entre la edad y el nivel de glucosa en sangre de los pacientes diabéticos y no diabéticos
sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=diabetes_dataset)
plt.title('Relación entre edad y nivel de glucosa en sangre')
plt.xlabel('Edad')
plt.ylabel('Nivel de glucosa en sangre')
plt.xticks(range(0, 100, 10))
plt.show()

# diagrama de cajas y bigotes que compara la distribución del nivel de glucosa en sangre entre pacientes diabéticos
#y no diabéticos en el conjunto de datos. 
sns.boxplot(x='Outcome', y='Glucose', data=diabetes_dataset, fliersize=3)
plt.title('Distribución del nivel de glucosa en sangre para pacientes diabéticos y no diabéticos', fontsize=14)
plt.xlabel('Diabético', fontsize=12)
plt.ylabel('Nivel de glucosa', fontsize=12)
plt.figure(figsize=(8,6))
plt.show()

# Me quedo con las columnas relevantes y las traduzco al español
diabetes_df_limpio = diabetes_dataset.loc[:, ['Age', 'Glucose', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Outcome']]
diabetes_df_limpio = diabetes_df_limpio.rename(columns={
    'Age': 'Edad',
    'Glucose': 'Glucosa',
    'Insulin': 'Insulina',
    'BMI': 'IMC',
    'DiabetesPedigreeFunction': 'Historial Familiar de Diabetes',
    'Outcome': 'Resultado'
})


#matriz de correlación entre las variables
corr = diabetes_df_limpio.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

sns.pairplot(data=diabetes_df_limpio, hue='Resultado', corner=True)

#separo los datos y los labels
x = diabetes_df_limpio.drop(columns='Resultado', axis=1)

#estandarización de los datos
scaler = StandardScaler()
scaler.fit(x)
data_estadarizada = scaler.transform(x)
data_estadarizada

#entrenamiento del modelo
x = data_estadarizada
y = diabetes_df_limpio['Resultado']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)

print(x.shape, x_train.shape, x_test.shape)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

#resultado del accurancy sobre los datos del entrenmiento
x_train_prediction = classifier.predict(x_train)
training_data_accurancy = accuracy_score(x_train_prediction, y_train)
training_data_accurancy

#resultado del accurancy sobre los datos de prueba
x_test_prediction = classifier.predict(x_test)
testing_data_accurancy = accuracy_score(x_test_prediction, y_test)
testing_data_accurancy

#Modelo predictivo
input_data = (21,	89,	94,	28.1,	0.167)

# cambio el input_data a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape del array 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# estandarización de input data
std_data = scaler.transform(input_data_reshaped)
std_data

prediction = classifier.predict(std_data)
prediction

if (prediction[0] == 0):
  print('la persona no es diabética')
else:
  print('La persona es diabética ')