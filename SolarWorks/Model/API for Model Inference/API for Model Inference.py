from fastapi import FastAPI
import tensorflow as tf
import joblib
import numpy as np
import onnxruntime as ort

app = FastAPI()

# Load models
solar_model = tf.keras.models.load_model("solar_power_model.h5")
inverter_model = ort.InferenceSession("inverter_failure.onnx")

@app.get("/predict_solar")
def predict_solar(value: float):
    prediction = solar_model.predict(np.array([[value]]))[0][0]
    return {"solar_power_prediction": prediction}

@app.get("/predict_inverter")
def predict_inverter(features: list):
    input_data = np.array(features, dtype=np.float32).reshape(1, -1)
    pred_onx = inverter_model.run(None, {"input": input_data})[0][0]
    return {"inverter_failure_prediction": int(pred_onx)}

