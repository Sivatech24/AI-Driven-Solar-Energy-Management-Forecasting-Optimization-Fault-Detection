import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

# Default Paths
DEFAULT_CSV_PATH = "solar_power_data.csv"
DEFAULT_MODEL_PATH = "lstm_solar_model.keras"

def load_data(csv_path):
    """Load dataset and preprocess it."""
    print(f"📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert DATE_TIME to datetime
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True, errors="coerce")

    # Feature Engineering
    df["Hour"] = df["DATE_TIME"].dt.hour
    df["Day"] = df["DATE_TIME"].dt.day
    df["Month"] = df["DATE_TIME"].dt.month

    # Select only required features
    selected_features = ["DC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "Hour", "Day", "Month"]
    df = df[selected_features]  # ✅ Ensure exactly 6 features

    print(f"✅ Selected Features: {df.columns.tolist()}")
    return df

def load_model(model_path):
    """Load trained LSTM model."""
    print(f"🤖 Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        exit(1)

def predict_solar_power(model, input_data):
    """Make predictions using the trained model."""
    try:
        # Reshape input to match LSTM expected format: (batch_size, timesteps, features)
        input_data = np.expand_dims(input_data, axis=0)  # Reshape to (1, 6)
        input_data = np.expand_dims(input_data, axis=1)  # Reshape to (1, 1, 6)

        prediction = model.predict(input_data)
        print(f"🔮 Predicted Solar Power Output (AC_POWER): {prediction.flatten()}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")

def main():
    """Command-line interface for loading data and predicting solar power."""
    parser = argparse.ArgumentParser(description="Predict solar power output using an LSTM model.")
    parser.add_argument("--file", type=str, default=DEFAULT_CSV_PATH, help="Path to CSV dataset")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to LSTM model")
    
    args = parser.parse_args()

    # Load dataset and select one sample for prediction
    df = load_data(args.file)
    sample_input = df.iloc[0].values.astype(np.float32)  # Select first row as test input

    # Load model
    model = load_model(args.model)

    # Predict
    predict_solar_power(model, sample_input)

if __name__ == "__main__":
    main()
