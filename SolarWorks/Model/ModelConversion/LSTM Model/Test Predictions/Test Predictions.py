import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load trained LSTM model
model = tf.keras.models.load_model("lstm_solar_model.keras")

# Define feature columns (same as training)
features = ["DC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "Hour", "Day", "Month"]

# Create synthetic test data (simulate past 24 hours)
test_data = pd.DataFrame({
    "DC_POWER": np.random.randint(50, 500, size=24),  
    "DAILY_YIELD": np.random.randint(1000, 2000, size=24),  
    "TOTAL_YIELD": np.random.randint(40000, 60000, size=24),  
    "Hour": np.arange(0, 24),  
    "Day": [10] * 24,  
    "Month": [3] * 24  
})

# Load the same scaler used during training
scaler = MinMaxScaler()
test_data_scaled = scaler.fit_transform(test_data)

# Reshape for LSTM (samples, timesteps, features)
X_test = test_data_scaled.reshape((1, 24, len(features)))

# Predict solar power output
predicted_ac_power = model.predict(X_test)

# Print predictions
print("🔮 Predicted Solar Power Output (AC_POWER):", predicted_ac_power.flatten())
