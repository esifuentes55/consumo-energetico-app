# app_streamlit_diario.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Configuración inicial
st.set_page_config(page_title="⚡ Predicción de Consumo Diario", layout="wide")
st.title("⚡ Predicción de Consumo Energético Diario (30 días)")

# Verificación de archivos requeridos
# if not os.path.exists("household_power_consumption.txt"):
#     st.error("❌ No se encontró el archivo 'household_power_consumption.txt'.")
#     st.stop()

if not os.path.exists("modelo_diario_30dias.h5"):
    st.error("❌ No se encontró el modelo entrenado 'modelo_diario_30dias.h5'.")
    st.stop()

# --- Cargar datos y procesar ---
@st.cache_data
def cargar_datos():
    url = "https://drive.google.com/uc?id=1HJkvX1rk9dqBuYzfjeBY_xNdQAMdlSHo"
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    
    # Reemplaza este nombre con el real si no es exactamente este
    with z.open("household_power_consumption.txt") as file:
        df = pd.read_csv(file, sep=';', na_values='?', low_memory=False)

    df.columns = df.columns.str.strip()
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('DateTime', inplace=True)
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df_daily = df['Global_active_power'].resample('D').mean()
    df_daily.dropna(inplace=True)
    return df_daily

df_daily = cargar_datos()

# Visualización de historial
st.subheader("📊 Consumo Diario Histórico")
st.line_chart(df_daily)

# Escalado de datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_daily.values.reshape(-1, 1))

# Preparación de entrada para predicción
def crear_secuencia_final(data, steps=30):
    return np.array(data[-steps:]).reshape(1, steps, 1)

X_pred = crear_secuencia_final(scaled_data)

# Cargar modelo
model = load_model("modelo_diario_30dias.h5")

# Realizar predicción
pred = model.predict(X_pred)
pred_inv = scaler.inverse_transform(pred.reshape(-1, 1))

# Mostrar resultado
dias_futuros = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=30)
df_pred = pd.DataFrame(pred_inv, index=dias_futuros, columns=["Consumo (kWh)"])

st.subheader("🔮 Predicción de Consumo para los Próximos 30 Días")
st.line_chart(df_pred)

with st.expander("📋 Ver tabla de predicción"):
    st.dataframe(df_pred.style.format("{:.2f}"))

st.success("✅ Predicción generada exitosamente.")
