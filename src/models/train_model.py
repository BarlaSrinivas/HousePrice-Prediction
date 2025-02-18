# -*- coding: utf-8 -*-
"""
Train House Price Prediction Model

This script trains an XGBoost regression model using the California Housing dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import os
import joblib


def load_data():
    dataset = fetch_california_housing()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["price"] = dataset.target
    return df


def train_model(df):
    X = df.drop(columns=["price"], axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_score = metrics.r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2 Score): {r2_score:.2f}")

    return y_pred


def save_plots(y_test, y_pred):

    # Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.savefig("../../reports/figures/actual_vs_predicted.png")
    plt.close()

    # Distribution of Errors
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True)
    plt.xlabel("Prediction Error")
    plt.title("Distribution of Prediction Errors")
    plt.savefig("../../reports/figures/error_distribution.png")
    plt.close()

    print("Plots saved in 'figures' folder")


if __name__ == "__main__":
    df = load_data()
    model, X_test, y_test = train_model(df)
    y_pred = evaluate_model(model, X_test, y_test)
    save_plots(y_test, y_pred)

    # Save the trained model
    joblib.dump(model, "../../models/trained_model.joblib")
    print("Model saved as 'trained_model.joblib'")
