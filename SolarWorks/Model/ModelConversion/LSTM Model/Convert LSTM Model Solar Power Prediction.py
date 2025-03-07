import tensorflow as tf

# Load the trained LSTM model
model = tf.keras.models.load_model("lstm_solar_model.keras")

# Verify model structure
model.summary()
