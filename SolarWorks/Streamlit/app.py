import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')

st.title("Solar Plant Data Analysis and Forecasting")

# File Upload
uploaded_gen = st.file_uploader("Upload Generation Data CSV", type=["csv"], key="gen")
uploaded_weather = st.file_uploader("Upload Weather Sensor Data CSV", type=["csv"], key="weather")

def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return None

# Load Data
gen_data = load_data(uploaded_gen)
weather_data = load_data(uploaded_weather)

default_gen_data = pd.read_csv('Plant_1_Generation_Data.csv')
default_weather_data = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')

if gen_data is None:
    gen_data = default_gen_data
if weather_data is None:
    weather_data = default_weather_data

# Data Preview
st.subheader("Generation Data Preview")
st.dataframe(gen_data.head())

st.subheader("Weather Data Preview")
st.dataframe(weather_data.head())

# Convert DateTime columns
gen_data['DATE_TIME'] = pd.to_datetime(gen_data['DATE_TIME'], format='%d-%m-%Y %H:%M')
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

# Data Description
st.subheader("Statistical Summary - Generation Data")
st.write(gen_data.describe())

st.subheader("Statistical Summary - Weather Data")
st.write(weather_data.describe())

# Correlation Heatmap
st.subheader("Correlation Heatmap - Generation Data")
numeric_gen_data = gen_data.select_dtypes(include=['float64', 'int64'])
gen_correlation = numeric_gen_data.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(gen_correlation, annot=True, ax=ax)
st.pyplot(fig)

st.subheader("Correlation Heatmap - Weather Data")
numeric_weather_data = weather_data.select_dtypes(include=['float64', 'int64'])
weather_correlation = numeric_weather_data.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(weather_correlation, annot=True, ax=ax)
st.pyplot(fig)

# Daily Yield & AC/DC Power
df_gen = gen_data.set_index('DATE_TIME').resample('D').sum().reset_index()
st.subheader("Daily Yield & AC/DC Power")
fig, ax = plt.subplots(figsize=(12, 5))
df_gen.plot(x='DATE_TIME', y=['DAILY_YIELD', 'TOTAL_YIELD'], ax=ax)
st.pyplot(fig)

# Forecasting
st.subheader("Time Series Forecasting with ARIMA")
df_daily_gen = gen_data.groupby('DATE_TIME').sum().reset_index()
df_daily_gen.set_index('DATE_TIME', inplace=True)
train_gen, test_gen = train_test_split(df_daily_gen[['DAILY_YIELD']], test_size=0.2, shuffle=False)

arima_model = ARIMA(train_gen['DAILY_YIELD'], order=(5,1,0))
arima_fit = arima_model.fit()
forecast_arima = arima_fit.forecast(steps=len(test_gen))
test_gen['Forecast_ARIMA'] = forecast_arima

fig, ax = plt.subplots(figsize=(12, 5))
train_gen['DAILY_YIELD'].plot(ax=ax, label='Train')
test_gen['DAILY_YIELD'].plot(ax=ax, label='Test')
test_gen['Forecast_ARIMA'].plot(ax=ax, label='ARIMA Forecast')
plt.legend()
st.pyplot(fig)

st.subheader("SARIMA Model Forecast")
sarima_model = SARIMAX(train_gen['DAILY_YIELD'], order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.forecast(steps=len(test_gen))
test_gen['Forecast_SARIMA'] = sarima_forecast

fig, ax = plt.subplots(figsize=(12, 5))
train_gen['DAILY_YIELD'].plot(ax=ax, label='Train')
test_gen['DAILY_YIELD'].plot(ax=ax, label='Test')
test_gen['Forecast_SARIMA'].plot(ax=ax, label='SARIMA Forecast')
plt.legend()
st.pyplot(fig)

# DC Power Efficiency
gen_data['DC_POWER_CONVERTED'] = gen_data['DC_POWER'] * 0.98
fig, ax = plt.subplots(figsize=(12, 5))
gen_data.plot(x='DATE_TIME', y='DC_POWER_CONVERTED', ax=ax, title="DC Power Converted")
st.pyplot(fig)

# DC Power and Daily Yield
df_day = gen_data.copy()
df_day['time'] = df_day['DATE_TIME'].dt.time
df_day['day'] = df_day['DATE_TIME'].dt.date

df_day_grouped = df_day.groupby(['time', 'day'])['DC_POWER'].mean().unstack()
fig, ax = plt.subplots(figsize=(15, 10))
df_day_grouped.plot(ax=ax)
st.pyplot(fig)

# Temperature Analysis
st.subheader("Module & Ambient Temperature")
fig, ax = plt.subplots(figsize=(12, 5))
weather_data.plot(x='DATE_TIME', y=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE'], ax=ax)
st.pyplot(fig)

# ARIMA Model Summary
st.subheader("ARIMA Model Summary")
st.text(arima_fit.summary())
