import numpy as np
import onnxruntime as ort
import pandas as pd

# Load dataset
df = pd.read_csv("solar_power_data.csv")

# Convert DATE_TIME to datetime format
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True, errors="coerce")

# Feature Engineering (Extract Hour, Day, Month)
df["Hour"] = df["DATE_TIME"].dt.hour
df["Day"] = df["DATE_TIME"].dt.day
df["Month"] = df["DATE_TIME"].dt.month

# Define features
features = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "Hour", "Day", "Month"]

# Select a sample row and duplicate it
test_sample = df.iloc[0].copy()
test_sample["AC_POWER"] = test_sample["DC_POWER"] * 0.5  # Strongly force failure (Efficiency = 0.5)

# Create test dataset with multiple copies (5 failure cases)
test_data = pd.DataFrame([test_sample] * 5)

# Convert to NumPy array
X_test = test_data[features].values.astype(np.float32)

# Load ONNX model
session = ort.InferenceSession("rf_inverter_failure.onnx")

# Run inference
input_name = session.get_inputs()[0].name
predictions = session.run(None, {input_name: X_test})[0]

print("🔍 Predicted Failures:", predictions)
