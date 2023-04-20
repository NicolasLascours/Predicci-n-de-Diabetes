#Importo librerías
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#lectura del dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

#Muestro los 5 primeros elementos de dataset
print(diabetes_dataset.head())

#cantidad de filas y columnas
print(diabetes_dataset.shape)

#información estadística
print(diabetes_dataset.describe())

#cantidad de diabéticos (1) y no diabéticos (0) en el dataset
print(diabetes_dataset['Outcome'].value_counts())

#Crear diagrama de barras
sns.countplot(x='Outcome', data=diabetes_dataset)
plt.title('Cantidad de pacientes diabéticos y no diabéticos')
plt.xlabel('Estado')
plt.ylabel('Cantidad')
#plt.show()


#el % de glucosa en sangre en diabéticos es mayor que los que no lo son
#a su vez, la personas de mayor edad son más suceptibles a tener diabetes.
print(diabetes_dataset.groupby('Outcome').mean())

# gráfico de dispersión que indica la relación entre la edad y el nivel de glucosa en sangre de los pacientes diabéticos y no diabéticos
sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=diabetes_dataset)
plt.title('Relación entre edad y nivel de glucosa en sangre')
plt.xlabel('Edad')
plt.ylabel('Nivel de glucosa en sangre')
plt.xticks(range(0, 100, 10))
plt.show()

# diagrama de cajas y bigotes que compara la distribución del nivel de glucosa en sangre entre pacientes diabéticos y no diabéticos en el conjunto de datos. 
sns.boxplot(x='Outcome', y='Glucose', data=diabetes_dataset, fliersize=3)
plt.title('Distribución del nivel de glucosa en sangre para pacientes diabéticos y no diabéticos', fontsize=14)
plt.xlabel('Diabético', fontsize=12)
plt.ylabel('Nivel de glucosa', fontsize=12)
plt.figure(figsize=(8,6))
plt.show()

corr = diabetes_dataset.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()



