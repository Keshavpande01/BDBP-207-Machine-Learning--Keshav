# Implement a linear regression model using scikit-learn for the simulated dataset
# - simulated_data_multiple_linear_regression_for_ML.csv  - to predict the “disease_score” from multiple clinical parameters

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def main():

    # Step 1: Load dataset
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
    df = pd.read_csv(file_path)

    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns)
    print(df.head())

    # Step 2: Separate features and target
    X = df.iloc[:, 0:5]   # clinical parameters
    y = df.iloc[:, 5]     # disease_score
    print(X,y)

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    # Step 4: Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Initialize Linear Regression model
    model = LinearRegression()

    # Step 6: Train the model
    model.fit(X_train_scaled, y_train)

    # Step 7: Predictions
    y_pred = model.predict(X_test_scaled)

    # Step 8: Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

if __name__ == "__main__":
    main()



# def main():
#     # step 1 :First load the data file
#     file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
#     df = pd.read_csv(file_path)
#     print(df)
#
#     # Display the basic information of dataset we are using :
#     print("The Shape of Dataset : ",df.shape)
#     print("The totoal size of Dataset",df.size)
#     print(df.columns)
#     print(df.head())
#
#     # Features: all columns except the targets
#     X = df.drop(columns=["disease_score", "disease_score_fluct"])
#
#      Method 2 : Initialize the variable and target
#     y_score = df["disease_score"]
#     y_fluct = df["disease_score_fluct"]
#
#     X_train, X_test, y_score_train, y_score_test = train_test_split(
#         X, y_score, test_size=0.2, random_state=42)
#
#     _, _, y_fluct_train, y_fluct_test = train_test_split(
#         X, y_fluct, test_size=0.2, random_state=42
#     )
#     # Initialize
#     lr_score = LinearRegression()
#
#     # train model
#     lr_score.fit(X_train, y_score_train)
#
#     # Predictions
#     y_score_pred = lr_score.predict(X_test)
#
#     # Evaluation
#     mse_score = mean_squared_error(y_score_test, y_score_pred)
#     r2_score_value = r2_score(y_score_test, y_score_pred)
#
#     print("Disease Score Model")
#     print("MSE:", mse_score)
#     print("R2 Score:", r2_score_value)
#
# if __name__ == '__main__':
#     main()
