import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import zipfile
import io

st.set_page_config(page_title="Predicción de Consumo Diario", layout="wide")
st.title("⚡ Predicción de Consumo Energético Diario (30 días)")

# --- Función para cargar y procesar datos desde un ZIP ---
@st.cache_data
def cargar_datos(zip_file, txt_name="household_power_consumption.txt", nrows=None):
    try:
        with zipfile.ZipFile(zip_file) as z:
            with z.open(txt_name) as file:
                df = pd.read_csv(file, sep=';', low_memory=False, nrows=nrows)
                df = df[df['Global_active_power'] != '?']
                df['Global_active_power'] = pd.to_numeric(df['Global_active_power'])
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="%d/%m/%Y %H:%M:%S")
                df.set_index('DateTime', inplace=True)
                df_diario = df['Global_active_power'].resample('D').mean()
                return df_diario.dropna()
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
        return None

# --- Subida del archivo ZIP ---
zip_file = st.file_uploader("📁 Sube el archivo ZIP con el dataset", type="zip")

if zip_file is not None:
    df_daily = cargar_datos(zip_file)

    if df_daily is not None:
        # --- Mostrar datos históricos ---
        st.subheader("📈 Consumo Diario Histórico")
        st.line_chart(df_daily)

        # --- Escalado y preparación ---
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
        try:
            model = load_model("modelo_diario_30dias.h5")
            pred = model.predict(X_pred)
            pred_inv = scaler.inverse_transform(pred.reshape(-1, 1))

            # --- Mostrar predicción ---
            st.subheader("🔮 Predicción de Consumo para los Próximos 30 Días")
            dias_futuros = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=30)
            df_pred = pd.DataFrame(pred_inv, index=dias_futuros, columns=["Consumo (kWh)"])
            st.line_chart(df_pred)

            with st.expander("🔍 Ver tabla de predicción"):
                st.dataframe(df_pred.style.format("{:.2f}"))

        except Exception as e:
            st.error(f"Error al cargar el modelo o predecir: {e}")
else:
    st.info("Por favor sube el archivo ZIP con el dataset.")
