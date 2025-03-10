# AI-Driven Solar Power Prediction

## Overview
This project focuses on forecasting **solar power generation** using **LSTM (Long Short-Term Memory) neural networks** and a **Dense (fully connected) model**. The goal is to improve solar energy efficiency, optimize power output, and predict inverter failures.

## Features
- **Time Series Forecasting**: Uses past energy production data to predict future output.
- **LSTM Model**: Captures temporal dependencies in power generation.
- **Dense Model**: Provides a baseline for comparison.
- **Scalability**: Easily deployable for real-time applications.
- **Preprocessing**: Automated data normalization and feature engineering.

## Dataset
The dataset contains **solar plant energy readings** with the following key features:
- `DATE_TIME` (Timestamp of the reading)
- `DC_POWER` (Generated DC power in kW)
- `AC_POWER` (Generated AC power in kW)
- `DAILY_YIELD` (Cumulative yield per day in kWh)
- `TOTAL_YIELD` (Total cumulative energy generation in kWh)

## Installation
Ensure you have Python **3.8+** installed, then install the required packages:
```sh
pip install numpy pandas tensorflow scikit-learn
```

## Usage
### 1. Train the Model
Run the following command to train the LSTM model:
```python
from solar_power import train_model
model, scaler = train_model("data.csv")
```
This will:
- Load and preprocess the dataset.
- Train the LSTM model.
- Save the trained model as `solar_power_lstm.keras`.

### 2. Predict Power Output
After training, you can use the trained model for predictions:
```python
import numpy as np
from tensorflow.keras.models import load_model

def predict_future_power(model_path, input_data):
    model = load_model(model_path)
    input_data = np.array(input_data).reshape(1, 24, -1)  # Reshape for LSTM
    return model.predict(input_data)

predicted_power = predict_future_power("solar_power_lstm.keras", sample_input)
print("Predicted Power Output:", predicted_power)
```

## Model Architecture
### **LSTM Model**
```python
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```
- **LSTM layers** capture time-series dependencies.
- **Dropout layers** prevent overfitting.
- **Dense layers** refine predictions.

### **Dense Model (Baseline)**
```python
def build_dense_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```

## Performance Optimization
To reduce training time:
1. Use a **smaller batch size** (e.g., `batch_size=16`).
2. Enable **GPU acceleration** with TensorFlow (`tensorflow-gpu`).
3. Apply **XLA optimization**:
    ```python
    import tensorflow as tf
    tf.config.optimizer.set_jit(True)
    ```
4. Use **Early Stopping** to stop training when validation loss stops improving:
    ```python
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ```

## Future Improvements
- **Real-time deployment** using Flask/FastAPI.
- **Integration with IoT sensors** for real-time solar panel monitoring.
- **Hybrid models** combining CNNs and LSTMs for better accuracy.

## License
This project is licensed under the **MIT License**.

---
🚀 **Start optimizing solar energy production today!**

