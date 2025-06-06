import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- Cargar y preprocesar datos ---
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

# --- Escalar ---
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_diario.values.reshape(-1, 1))

# --- Crear secuencias multistep ---
def crear_secuencias_multistep(data, input_steps=30, output_steps=30):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)

X, y = crear_secuencias_multistep(data_scaled)
X = X.reshape((X.shape[0], X.shape[1], 1))

# --- Crear modelo compatible con .h5 ---
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(30))

model.compile(optimizer='adam', loss='mse')

# --- Entrenar modelo ---
stop = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2, callbacks=[stop])

# --- Guardar modelo ---
model.save("modelo_diario_30dias.h5")
print("✅ Modelo guardado como modelo_diario_30dias.h5")
