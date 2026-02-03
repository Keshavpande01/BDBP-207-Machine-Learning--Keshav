import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def load_data(filename):
    data = pd.read_csv(filename)

    # Features: first 5 columns
    X = data.iloc[:, 0:5].values.tolist()

    # Target: disease_score_fluct (column index 6)
    y = data.iloc[:,6].values.tolist()

    return X, y

def split_data(X, y, test_size=0.3, random_state=999):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def compute_mean_and_std(X):
    n_samples = len(X)
    n_features = len(X[0])

    mean = [0.0] * n_features
    std = [0.0] * n_features

    for j in range(n_features):
        mean[j] = sum(X[i][j] for i in range(n_samples)) / n_samples

    for j in range(n_features):
        std[j] = (
            sum((X[i][j] - mean[j]) ** 2 for i in range(n_samples)) / n_samples
        ) ** 0.5

    return mean, std
def normalize(X, mean, std):
    X_norm = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            row.append((X[i][j] - mean[j]) / std[j])
        X_norm.append(row)
    return X_norm
def gradient_descent(X, y, learning_rate=0.001, iterations=10000):
    n_samples = len(X)
    n_features = len(X[0])

    # Initialize parameters
    weights = [0.0] * n_features
    bias = 0.0

    for it in range(iterations):

        # Predictions
        y_pred = []
        for i in range(n_samples):
            pred = sum(weights[j] * X[i][j] for j in range(n_features)) + bias
            y_pred.append(pred)

        # Gradients
        dw = [0.0] * n_features
        db = 0.0

        for i in range(n_samples):
            error = y[i] - y_pred[i]
            for j in range(n_features):
                dw[j] += (-2 / n_samples) * X[i][j] * error
            db += (-2 / n_samples) * error

        # Update parameters
        for j in range(n_features):
            weights[j] -= learning_rate * dw[j]
        bias -= learning_rate * db

        # Print loss occasionally
        if it % 1000 == 0:
            loss = sum((y[i] - y_pred[i]) ** 2 for i in range(n_samples)) / n_samples
            print(f"Iteration {it}, Loss: {loss:.4f}")

    return weights, bias
def predict(X, weights, bias):
    predictions = []
    for i in range(len(X)):
        pred = sum(weights[j] * X[i][j] for j in range(len(weights))) + bias
        predictions.append(pred)
    return predictions

def mean_squared_error(y_true, y_pred):
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)


def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    ss_tot = sum((y_true[i] - mean_y) ** 2 for i in range(len(y_true)))
    return 1 - (ss_res / ss_tot)

def main():
    # Load data
    X, y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Compute normalization parameters (training data only)
    mean, std = compute_mean_and_std(X_train)

    # Normalize
    X_train_norm = normalize(X_train, mean, std)
    X_test_norm = normalize(X_test, mean, std)

    # Train model using gradient descent
    weights, bias = gradient_descent(
        X_train_norm,
        y_train,
        learning_rate=0.001,
        iterations=10000
    )

    # Test predictions
    y_test_pred = predict(X_test_norm, weights, bias)

    # Evaluation
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print("\nTest MSE:", mse)
    print("Test R²:", r2)

if __name__ == "__main__":
    main()