import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the trained Random Forest model
rf_model = joblib.load("inverter_failure_model.pkl")

# Define input type
initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]

# Convert the model to ONNX format
onnx_model = convert_sklearn(rf_model, initial_types=initial_type)

# Save the ONNX model
with open("random_forest_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
