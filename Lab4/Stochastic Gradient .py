import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------
# Load Dataset
# -----------------------
def load_data():

    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

    print("Dataset Shape:", df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


# -----------------------
# EDA
# -----------------------
def perform_eda(df):

    print("\nStatistical Summary:")
    print(df.describe())

    numeric_df = df.select_dtypes(include=['int64','float64'])

    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    numeric_df.hist(figsize=(10,8))
    plt.tight_layout()
    plt.show()


# -----------------------
# Standardization
# -----------------------
def standardize(X):

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std


# -----------------------
# Hypothesis
# -----------------------
def hypothesis(x, theta):

    return np.dot(x, theta)


# -----------------------
# Stochastic Gradient Descent
# -----------------------
def stochastic_gradient_descent(X, y, alpha=0.001, epochs=100):

    m, n = X.shape

    theta = np.zeros(n)

    for epoch in range(epochs):

        for i in range(m):

            xi = X[i]
            yi = y[i]

            prediction = hypothesis(xi, theta)

            error = prediction - yi

            gradient = error * xi

            theta = theta - alpha * gradient

        print(f"Epoch {epoch+1} completed",{error})

    return theta


# -----------------------
# Prediction
# -----------------------
def predict(X, theta):

    return np.dot(X, theta)


# -----------------------
# R2 Score
# -----------------------
def r2_score(y_true, y_pred):

    mean_y = np.mean(y_true)

    ss_total = np.sum((y_true - mean_y)**2)

    ss_res = np.sum((y_true - y_pred)**2)

    return 1 - (ss_res/ss_total)


# -----------------------
# Main
# -----------------------
def main():

    df = load_data()

    perform_eda(df)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Feature scaling
    X = standardize(X)

    # Add intercept
    X = np.c_[np.ones(X.shape[0]), X]

    theta = stochastic_gradient_descent(X, y, alpha=0.001, epochs=100)

    print("\nFinal Theta:", theta)

    y_pred = predict(X, theta)

    r2 = r2_score(y, y_pred)

    print("\nR2 Score:", r2)


if __name__ == "__main__":
    main()