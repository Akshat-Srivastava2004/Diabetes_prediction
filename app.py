# -*- coding: utf-8 -*-
"""
Flask App for Diabetes Prediction
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the app

# Load the saved model and scaler
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))  # Load the scaler

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

# Flask Routes
@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Diabetes Prediction API!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()
        input_values = [
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]
        # Convert input data to float
        input_values = [float(x) for x in input_values]
        # Get prediction
        result = diabetes_prediction(input_values)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
