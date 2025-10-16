import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🌧️ Rainfall Prediction App")

# Load model and scaler
with open("random_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

feature_names = ['Temperature', 'Humidity', 'Wind Speed', 'Pressure']

st.header("Manual Input")
temperature = st.number_input("Temperature", value=25.0)
humidity = st.number_input("Humidity", value=60.0)
wind_speed = st.number_input("Wind Speed", value=10.0)
pressure = st.number_input("Pressure", value=1013.0)

if st.button("Predict Rainfall (Manual Input)"):
    sample = pd.DataFrame([{
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind Speed': wind_speed,
        'Pressure': pressure
    }])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    result = "Rain Tomorrow" if prediction == 1 else "No Rain Tomorrow"
    st.success(f"🌦️ Prediction: **{result}**")

