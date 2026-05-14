# src/ft_engireering.py 

# librerías
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
# ahora, importamos la función CargarDatos del script cargar_datos.py 
from cargar_datos import cargarDatos

# cargar los datos
df = cargarDatos()

# pre-visualización de los datos
df.info()
print(df.head())
print(df.describe())

# Paso 1: división de features/target 
X = df.drop('Pago_atiempo', axis = 1) # features (estas son las variables que nos servirán para predecir)
y = df['Pago_atiempo']                # target (esta es la variable que queremos predecir)

# Paso 2: definir varioables por tipo
num_features = X.select_dtypes('number').columns
cat_features = X.select_dtypes('object').columns

# Paso 3: Crear pipelines de cada ruta
## Ruta 1: numéricas


## Ruta 2: categóricas


# Paso 4: Combinar las 2 rutas en ColumnTransformer


# Paso 5: dividir el dataset en train/test (antes de preprocesar)


# Paso 6: Aplicamos el preprocesamiento 


# Paso 7: Resultados del preprocesamiento


# Paso 8: Construimos una función para "exportar": ft_engineering()
