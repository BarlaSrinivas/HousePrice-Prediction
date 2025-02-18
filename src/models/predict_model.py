# -*- coding: utf-8 -*-
"""
Predict House Prices using Trained Model

This script loads a trained XGBoost model and makes predictions on new data.
"""

import joblib
import numpy as np


def load_model(model_path="../../models/trained_model.joblib"):
    return joblib.load(model_path)


def predict_price(model, features):
    # Ensure features are in the correct order and format
    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    input_data = np.array([features[name] for name in feature_names]).reshape(1, -1)

    prediction = model.predict(input_data)
    return prediction[0]


if __name__ == "__main__":
    # Load the trained model
    model = load_model()

    # Example usage
    sample_house = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    predicted_price = predict_price(model, sample_house)
    print(f"Predicted house price: ${predicted_price:.2f}")
