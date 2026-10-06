import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Output directory create karein
os.makedirs("models", exist_ok=True)

# 2. CPCB dataset load karein
print("⏳ 1. CPCB dataset load ho raha hai...")
df = pd.read_csv("city_hour.csv")

# Representative cities filter karein
TARGET_CITIES = [
    "Delhi",
    "Mumbai",
    "Bengaluru",
    "Kolkata",
    "Hyderabad",
    "Visakhapatnam",
]
df = df[df["City"].isin(TARGET_CITIES)].copy()

# Datetime parsing aur sorting
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values(by=["City", "Datetime"])

# Training features
FEATURE_COLS = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "AQI"]

# 3. Missing sensor readings handle karein
print("🧹 2. Missing sensor gaps linearly interpolate ho rahe hain...")
df[FEATURE_COLS] = df.groupby("City")[FEATURE_COLS].transform(
    lambda group: group.interpolate(method="linear", limit_direction="both")
)
df = df.dropna(subset=FEATURE_COLS)

# 4. Feature Scaling (0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df[FEATURE_COLS].values)

# Scaler ko future inverse-transform ke liye save karein
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Scaler saved at: models/scaler.pkl")


# 5. Sliding Window Sequence Generator (Past 24h -> Next 24h AQI)
def create_sequences(data, input_steps=24, output_steps=24):
  X, y = [], []
  for i in range(len(data) - input_steps - output_steps):
    X.append(data[i : (i + input_steps), :])
    y.append(data[(i + input_steps) : (i + input_steps + output_steps), -1])
  return np.array(X), np.array(y)


print("⚙️ 3. Time-series sequence dataset ready kiya ja raha hai...")
# Training ko fast aur efficient rakhne ke liye recent 35,000 hourly steps
X, y = create_sequences(
    scaled_features[-35000:], input_steps=24, output_steps=24
)

split = int(0.85 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"📊 Training shape: {X_train.shape}, Validation shape: {X_val.shape}")


# 6. LSTM + Attention Model Architecture
def build_model(input_shape):
  inputs = layers.Input(shape=input_shape)

  # Bidirectional LSTM sequence encoder
  lstm_out = layers.Bidirectional(
      layers.LSTM(64, return_sequences=True, dropout=0.2)
  )(inputs)

  # Self-Attention layer
  query = layers.Dense(128)(lstm_out)
  attention = layers.Attention()([query, query])

  # Sequence reduction (Native Keras layer)
  context = layers.GlobalAveragePooling1D()(attention)

  # Dense Projection
  x = layers.Dense(64, activation="relu")(context)
  x = layers.Dropout(0.2)(x)
  outputs = layers.Dense(24)(x)  # Next 24 hours AQI forecast

  model = models.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  return model


model = build_model((24, len(FEATURE_COLS)))
model.summary()

# 7. Model Training
print("\n🚀 4. Training shuru ho rahi hai...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=1,
)

# 8. Model Save
model.save("models/aqi_lstm_attention.keras")
print(
    "\n🎉 Training complete! Model successfully saved at:"
    " models/aqi_lstm_attention.keras"
)
