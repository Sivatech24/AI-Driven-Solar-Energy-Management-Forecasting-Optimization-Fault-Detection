import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("solar_power_data.csv")

# Convert DATE_TIME to datetime format
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True, errors='coerce')

# Feature Engineering
df["Hour"] = df["DATE_TIME"].dt.hour
df["Day"] = df["DATE_TIME"].dt.day
df["Month"] = df["DATE_TIME"].dt.month

# Drop missing values
df.dropna(inplace=True)

# Define failure (assume efficiency below 0.85 is a failure)
df["Efficiency"] = df["AC_POWER"] / df["DC_POWER"]
df["Failure"] = (df["Efficiency"] < 0.85).astype(int)

# Features and Target
features = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "Hour", "Day", "Month"]
target = "Failure"

# Split the dataset (include SOURCE_KEY for tracking failures)
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    df[features], df[target], df["SOURCE_KEY"], test_size=0.2, random_state=42
)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "rf_inverter_failure_model.pkl")
print("✅ Model saved as 'rf_inverter_failure_model.pkl'")
