# app_streamlit_diario.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import zipfile
import os

st.set_page_config(page_title="Predicción de Consumo Diario", layout="wide")
st.title("⚡ Predicción de Consumo Energético Diario (30 días)")

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
    df_daily = df['Global_active_power'].resample('D').mean().dropna()
    return df_daily

df_daily = cargar_datos()

# --- Mostrar datos históricos ---
st.subheader("📊 Consumo Diario Histórico")
st.line_chart(df_daily)

# --- Cargar scaler y modelo ---
try:
    model = load_model("modelo_diario_30dias.keras")
    scaler = joblib.load("scaler_diario.save")
except Exception as e:
    st.error(f"Error al cargar el modelo o scaler: {e}")
    st.stop()

# --- Preparar datos para predicción ---
last_30 = df_daily[-30:].values.reshape(-1, 1)
last_30_scaled = scaler.transform(last_30)
X_input = last_30_scaled.reshape((1, 30, 1))

# --- Realizar predicción multistep ---
try:
    pred_scaled = model.predict(X_input)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))

    # Crear índice de fechas futuras
    future_dates = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=30)
    df_pred = pd.DataFrame(pred, index=future_dates, columns=["Consumo (kWh)"])
    
    # Mostrar resultados
    st.subheader("🔮 Predicción de Consumo para los Próximos 30 Días")
    st.line_chart(df_pred)

    with st.expander("📋 Ver tabla de predicción"):
        st.dataframe(df_pred.style.format("{:.2f}"))

except Exception as e:
    st.error(f"Error al predecir: {e}")
