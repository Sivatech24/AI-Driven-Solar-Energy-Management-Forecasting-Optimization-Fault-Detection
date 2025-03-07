import onnxruntime as ort
import numpy as np
import pandas as pd

# Load dataset for testing
df = pd.read_csv("solar_power_data.csv")

# Convert DATE_TIME to datetime format
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True, errors='coerce')

# Feature Engineering
df["Hour"] = df["DATE_TIME"].dt.hour
df["Day"] = df["DATE_TIME"].dt.day
df["Month"] = df["DATE_TIME"].dt.month

# Drop missing values
df.dropna(inplace=True)

# Select features for inference
features = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "Hour", "Day", "Month"]
X_test = df[features].iloc[:5].values.astype(np.float32)  # Select first 5 rows for testing

# Load ONNX model
session = ort.InferenceSession("rf_inverter_failure.onnx")

# Perform inference
input_name = session.get_inputs()[0].name
predictions = session.run(None, {input_name: X_test})[0]

# Print predictions
print("🔍 Predicted Failures:", predictions)
