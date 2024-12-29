import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load the trained model
model_path = "price_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Verify the model is a pipeline
if not hasattr(model, "predict"):
    st.error("The loaded model is not a valid scikit-learn pipeline. Please check the model file.")
    st.stop()

# App title
st.title("Malaysia House Price Prediction App")

# Input form
st.header("Enter House Details:")
location = st.selectbox("Location", ["Penang", "Maleka", "Putrajaya", "Langkawi"])
house_type = st.selectbox("House Type", ["Terrace", "Apartment", "Bungalow"])
size = st.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
age = st.number_input("House Age (years)", min_value=0, max_value=100, value=10)

# Prediction button
if st.button("Predict House Price"):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame(
        [[location, size, bedrooms, bathrooms, house_type, age]],
        columns=["Location", "Size (sq ft)", "Bedrooms", "Bathrooms", "House Type", "Age (years)"]
    )

    try:
        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"Estimated House Price: MYR {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
