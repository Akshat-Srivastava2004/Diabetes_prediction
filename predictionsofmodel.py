# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
import streamlit as st

# Load the saved model and scaler
loaded_model = pickle.load(open('C:/Users/aksha/Downloads/trained_model.sav', 'rb'))
scaler = pickle.load(open('C:/Users/aksha/Downloads/scaler.sav', 'rb'))  # Load the scaler

# Function for Prediction
def diabetes_prediction(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)  # Ensure correct type

    # Reshape the array as we're predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Scale the input data
    scaled_data = scaler.transform(input_data_reshaped)

    # Make prediction
    prediction = loaded_model.predict(scaled_data)

    # Interpret prediction
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # Title
    st.title('Diabetes Prediction Web App')

    # Input data from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Code for Prediction
    diagnosis = ''

    # Prediction button
    if st.button('Diabetes Test Result'):
        # Convert user input to float and pass to prediction
        input_values = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        input_values = [float(x) for x in input_values]
        diagnosis = diabetes_prediction(input_values)

    st.success(diagnosis)

if __name__ == '__main__':
    main()
