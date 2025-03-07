import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pandas as pd

# Load dataset for feature shape reference
df = pd.read_csv("solar_power_data.csv")
features = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "Hour", "Day", "Month"]

# Load the trained Random Forest model
rf_model = joblib.load("rf_inverter_failure_model.pkl")

# Define the input shape
initial_type = [("input", FloatTensorType([None, len(features)]))]

# Convert the model to ONNX format
onnx_model = convert_sklearn(rf_model, initial_types=initial_type)

# Save the ONNX model
with open("rf_inverter_failure.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Random Forest model successfully converted to ONNX format!")
