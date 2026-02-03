# Implement a linear regression model using scikit-learn for the simulated dataset -
# simulated_data_multiple_linear_regression_for_ML.csv  -
# to predict the “disease_score_fluct” from multiple clinical parameters.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Step 1: Load the dataset
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
    df = pd.read_csv(file_path)

    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns)
    print(df.head())

    # Step 2: Separate features (clinical parameters) and target
    # Assuming last column is disease_score_fluct
    X = df.iloc[:, 0:5]      # multiple clinical parameters
    y = df.iloc[:, 6]        # disease_score_fluct

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999
    )

    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)

    # Step 4: Feature standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Initialize Linear Regression model
    model = LinearRegression()

    # Step 6: Train the model
    model.fit(X_train_scaled, y_train)

    # Step 7: Predict on test data
    y_pred = model.predict(X_test_scaled)

    # Step 8: Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

if __name__ == "__main__":
    main()
