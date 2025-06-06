
# app_streamlit_diario.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import zipfile
import os

st.set_page_config(page_title="Predicción de Consumo Diario", layout="wide")

st.title(" Predicción de Consumo Energético Diario (30 días)")

# --- Cargar y descomprimir ZIP si es necesario ---
if not os.path.exists("household_power_consumption.txt"):
    if os.path.exists("household_power_consumption.zip"):
        with zipfile.ZipFile("household_power_consumption.zip", 'r') as zip_ref:
            zip_ref.extractall()

# --- Cargar datos ---
@st.cache_data
def cargar_datos():
    df = pd.read_csv("household_power_consumption.txt", sep=';', low_memory=False, na_values='?')
    df.columns = df.columns.str.strip()
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('DateTime', inplace=True)
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df_daily = df['Global_active_power'].resample('D').mean()
    df_daily.dropna(inplace=True)
    return df_daily

df_daily = cargar_datos()

# --- Mostrar datos históricos ---
st.subheader(" Consumo Diario Histórico")
st.line_chart(df_daily)

# --- Escalar y preparar datos ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_daily.values.reshape(-1, 1))

def crear_secuencias(data, input_steps=30):
    X = []
    for i in range(len(data) - input_steps):
        X.append(data[i:i+input_steps])
    return np.array(X)

X = crear_secuencias(scaled_data)
X_pred = X[-1].reshape((1, 30, 1))

# --- Cargar modelo ---
model = load_model("modelo_diario_30dias.h5")

# --- Predecir ---
pred = model.predict(X_pred)
pred_inv = scaler.inverse_transform(pred.reshape(-1, 1))

# --- Mostrar predicción ---
st.subheader(" Predicción de Consumo para los Próximos 30 Días")
dias_futuros = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=30)
df_pred = pd.DataFrame(pred_inv, index=dias_futuros, columns=["Consumo (kWh)"])
st.line_chart(df_pred)

# --- Mostrar tabla ---
with st.expander(" Ver tabla de predicción"):
    st.dataframe(df_pred.style.format("{:.2f}"))
