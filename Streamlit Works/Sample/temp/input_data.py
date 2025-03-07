import streamlit as st
import pandas as pd
import time

st.title("📂 Upload & Process Data")

# File Upload
uploaded_gen = st.file_uploader("Upload Generation Data CSV", type=["csv"], key="gen")
uploaded_weather = st.file_uploader("Upload Weather Sensor Data CSV", type=["csv"], key="weather")

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return None

# Load Data
gen_data = load_data(uploaded_gen)
weather_data = load_data(uploaded_weather)

default_gen_data = pd.read_csv('https://github.com/Sivatech24/Streamlit/raw/refs/heads/main/Plant_1_Generation_Data.csv')
default_weather_data = pd.read_csv('https://github.com/Sivatech24/Streamlit/raw/refs/heads/main/Plant_1_Weather_Sensor_Data.csv')

if gen_data is None:
    gen_data = default_gen_data
if weather_data is None:
    weather_data = default_weather_data

st.write("✅ Data Loaded Successfully!")

# Progress Bar & Log
progress_bar = st.progress(0)
status_text = st.empty()

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)
    status_text.text(f"Processing... {i + 1}%")

st.success("🎉 Data Processing Complete!")
