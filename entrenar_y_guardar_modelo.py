# entrenar_y_guardar_modelo.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import zipfile
import joblib

# --- Cargar y procesar datos ---
with zipfile.ZipFile("household_power_consumption.zip") as z:
    with z.open("household_power_consumption.txt") as file:
        df = pd.read_csv(file, sep=";", na_values="?", low_memory=False)
        df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
        df.set_index("DateTime", inplace=True)
        df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
        df_diario = df["Global_active_power"].resample("D").mean().dropna()

# --- Escalar ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_diario.values.reshape(-1, 1))

# --- Crear secuencias ---
def crear_secuencias(data, input_steps=30):
    X, y = [], []
    for i in range(len(data) - input_steps):
        X.append(data[i:i + input_steps])
        y.append(data[i + input_steps])
    return np.array(X), np.array(y)

X, y = crear_secuencias(scaled_data)

# --- Modelo LSTM ---
model = Sequential([
    Input(shape=(30, 1)),
    LSTM(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=20, batch_size=16)

# --- Guardar modelo y scaler ---
model.save("modelo_diario_30dias.keras")
joblib.dump(scaler, "scaler_diario.save")

print("Modelo y scaler guardados correctamente.")
