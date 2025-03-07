import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load dataset
generation_data = pd.read_csv("Plant_1_Generation_Data.csv")

# Convert DATE_TIME to datetime format
generation_data["DATE_TIME"] = pd.to_datetime(generation_data["DATE_TIME"], dayfirst=True, errors='coerce')

# Feature Engineering
generation_data["Hour"] = generation_data["DATE_TIME"].dt.hour
generation_data["Day"] = generation_data["DATE_TIME"].dt.day
generation_data["Month"] = generation_data["DATE_TIME"].dt.month

# Drop missing values
generation_data.dropna(inplace=True)

# Define failure (assume efficiency below 0.85 is a failure)
generation_data["Efficiency"] = generation_data["AC_POWER"] / generation_data["DC_POWER"]
generation_data["Failure"] = (generation_data["Efficiency"] < 0.85).astype(int)

# Features and Target
features = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "Hour", "Day", "Month"]
target = "Failure"

# Split dataset (Include SOURCE_KEY for tracking failures)
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    generation_data[features], generation_data[target], generation_data["SOURCE_KEY"], 
    test_size=0.2, random_state=42
)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, rf_predictions)
print(f"Inverter Failure Detection Accuracy: {accuracy:.2f}")
print(classification_report(y_test, rf_predictions))

# Identify failed inverters
failed_inverters = id_test[y_test.values == 1]  # Ensure correct filtering
print("\n🔴 Failed Inverters (IDs):")
print(failed_inverters.unique())


# Count failures per inverter ID
failed_inverter_counts = failed_inverters.value_counts()

# Plot bar graph
plt.figure(figsize=(12, 6))
failed_inverter_counts.plot(kind="bar", color="red", edgecolor="black")
plt.xlabel("Inverter ID (SOURCE_KEY)")
plt.ylabel("Number of Failures")
plt.title("🔴 Inverter Failures Per Inverter ID")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
