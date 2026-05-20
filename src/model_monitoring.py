# src/model_monitoring.py

# librerías
import os
import pandas as pd
import requests
# esta es la librería que usaremos para crear la aplicación en Streamlit
import streamlit as st
# librerías para visualización
#import ploty.epress as px
# librerías de machine learning
from sklearn.model_selection import train_test_split
# importar el método para cargar los datos
from cargar_datos import cargarDatos

####################################
## 1. Configuración de la aplicación
#####################################

API_URL = "http://localhost:8000/predict"  # URL de la API FastAPI
DATASET_PATH = "./Base_de_datos.csv"   # Ruta al dataset "transformado"
MONITOR_LOG = "./Base_de_datos.xlsx"   # Ruta al dataset a monitorear (dataset original)


###########################################
## 2. Cargar el dataset y dividir los datos
###########################################

@st.cache_data
def load_data():
    # 2.1 Llamamos a la funcion cargarDatos para cargar el dataset
    df = cargarDatos()

    print(df.head())

    # 2.2 Acá vamos a crear los features y el target
    target = "Pago_atiempo"
    X = df.drop(columns=[target]) # estos son los features que nos ayudarán a predecir el target
    y = df[target] # este es el target que queremos predecir

    # 2.3 En este punto vamos a dividir las X (features) y el target (y) en train/test
    X_ref, X_new, y_ref, y_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2.4 Hacemos que la función devuelva los datos divididos
    return X_ref, X_new, y_ref, y_new

X_ref, X_new, y_ref, y_new = load_data()

print("Datos de referencia (train):")
print(X_ref.head())
print(y_ref.head())
print("\nDatos nuevos (test):")
print(X_new.head())
print(y_new.head())

########################################
## 3. Vamos a crear la interfaz inicial
## la aplicación de Streamlit
########################################

st.title("📊 - Aplicación para el monitoreo de los datos")