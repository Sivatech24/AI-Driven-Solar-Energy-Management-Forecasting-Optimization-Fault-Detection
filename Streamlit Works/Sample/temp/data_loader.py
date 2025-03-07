import streamlit as st
import pandas as pd

# File Upload for Generation and Weather Data
uploaded_gen = st.file_uploader("Upload Generation Data CSV", type=["csv"], key="gen")
uploaded_weather = st.file_uploader("Upload Weather Sensor Data CSV", type=["csv"], key="weather")

def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return None

# Load Data (Keep Separate)
gen_data = load_data(uploaded_gen)
weather_data = load_data(uploaded_weather)

# Default datasets (if no file is uploaded)
default_gen_data = pd.read_csv('https://github.com/Sivatech24/Streamlit/raw/refs/heads/main/Plant_1_Generation_Data.csv')
default_weather_data = pd.read_csv('https://github.com/Sivatech24/Streamlit/raw/refs/heads/main/Plant_1_Weather_Sensor_Data.csv')

if gen_data is None:
    gen_data = default_gen_data

if weather_data is None:
    weather_data = default_weather_data
