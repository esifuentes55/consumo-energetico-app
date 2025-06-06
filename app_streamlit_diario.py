import streamlit as st
import zipfile
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Predicción de Consumo Diario", layout="wide")
st.title("🔋 Predicción de Consumo Energético Diario (30 días)")

# --- Cargar datos desde archivo ZIP ---
@st.cache_data
def cargar_datos():
    with zipfile.ZipFile("household_power_consumption.zip") as z:
        with z.open("household_power_consumption.txt") as file:
            df = pd.read_csv(file, sep=';', low_memory=False)
            df = df[df['Global_active_power'] != '?']
            df['Global_active_power'] = pd.to_numeric(df['Global_active_power'])
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="%d/%m/%Y %H:%M:%S")
            df.set_index('DateTime', inplace=True)
            df_diario = df['Global_active_power'].resample('D').mean()
            return df_diario.dropna()

df_diario = cargar_datos()

# --- Mostrar datos históricos ---
st.subheader("📊 Consumo Diario Histórico")
st.line_chart(df_diario)

# --- Escalar datos y preparar entrada ---
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_diario.values.reshape(-1, 1))
input_data = data_scaled[-30:].reshape((1, 30, 1))

# --- Cargar modelo y predecir ---
try:
    model = load_model("modelo_diario_30dias.keras")
    pred_scaled = model.predict(input_data)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))

    # --- Mostrar predicción ---
    st.subheader("🔮 Predicción de los Próximos 30 Días")
    fechas_futuras = pd.date_range(start=df_diario.index[-1] + pd.Timedelta(days=1), periods=30)
    df_pred = pd.DataFrame(pred, index=fechas_futuras, columns=["Consumo (kWh)"])
    st.line_chart(df_pred)

    with st.expander("🔍 Ver tabla de predicción"):
        st.dataframe(df_pred.style.format("{:.2f}"))

except Exception as e:
    st.error(f"Error al cargar el modelo o hacer predicciones: {e}")
