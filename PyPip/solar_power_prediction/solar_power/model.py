import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column_name, sequence_length=24):
    """ Preprocess data: normalize and create sequences for LSTM input. """
    
    feature_columns = ['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']

    # Check if all required columns exist
    if not all(col in df.columns for col in feature_columns):
        raise ValueError(f"Missing required columns: {feature_columns}")

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[feature_columns])

    target_column_idx = feature_columns.index(target_column_name)

    # Ensure dataset has enough rows
    if len(df_scaled) <= sequence_length:
        raise ValueError(f"Dataset too small: needs at least {sequence_length + 1} rows, found {len(df_scaled)}")

    sequences, labels = [], []
    for i in range(len(df_scaled) - sequence_length):
        sequences.append(df_scaled[i:i + sequence_length])
        labels.append(df_scaled[i + sequence_length][target_column_idx])  

    return np.array(sequences), np.array(labels), scaler

def build_lstm_model(input_shape):
    """ Build an LSTM model for solar power prediction. """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Removed incorrect `return_sequences=True`
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(data_path, save_path="solar_power_lstm.keras"):
    """ Train the LSTM model with the given dataset. """
    
    # Load dataset with correct date parsing
    df = pd.read_csv(data_path)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%d-%m-%Y %H:%M')

    df.sort_values(by=['DATE_TIME'], inplace=True)

    sequence_length = 24
    target_column_name = "AC_POWER"

    # Ensure dataset has enough rows
    if len(df) <= sequence_length:
        raise ValueError(f"Dataset too small: needs at least {sequence_length + 1} rows, found {len(df)}")

    # Preprocess data
    X, y, scaler = preprocess_data(df, target_column_name, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build and train model
    model = build_lstm_model((sequence_length, X.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Save trained model
    model.save(save_path)
    
    return model, scaler
