import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------
def load_data(filename):
    data = pd.read_csv(filename)

    # Features: first 5 columns
    X = data.iloc[:, 0:5].values.tolist()

    # Target: disease_score_fluct (column index 6)
    y = data.iloc[:,6].values.tolist()

    return X, y


# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------
def split_data(X, y, test_size=0.3, random_state=999):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


# --------------------------------------------------
# 3. Compute Mean and Std (Training Data Only)
# --------------------------------------------------
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


# --------------------------------------------------
# 4. Normalize Features
# --------------------------------------------------
def normalize(X, mean, std):
    X_norm = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            row.append((X[i][j] - mean[j]) / std[j])
        X_norm.append(row)
    return X_norm


# --------------------------------------------------
# 5. Main Training Pipeline
# --------------------------------------------------
def main():
    # Load dataset
    X, y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Compute normalization parameters on training data
    mean, std = compute_mean_and_std(X_train)

    # Normalize data
    X_train_norm = normalize(X_train, mean, std)
    X_test_norm = normalize(X_test, mean, std)

    # Convert to NumPy arrays
    X_train_norm = np.array(X_train_norm)
    X_test_norm = np.array(X_test_norm)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Initialize model parameters
    n_features = X_train_norm.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    learning_rate = 0.01
    iterations = 100000

    # --------------------------------------------------
    # 6. Gradient Descent Training
    # --------------------------------------------------
    for i in range(iterations):
        y_pred = np.dot(X_train_norm, weights) + bias

        dw = (-2 / len(y_train)) * np.dot(X_train_norm.T, (y_train - y_pred))
        db = (-2 / len(y_train)) * np.sum(y_train - y_pred)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        if i % 1000 == 0:
            loss = np.mean((y_train - y_pred) ** 2)
            print(f"Iteration {i}, Training Loss: {loss:.4f}")

    # --------------------------------------------------
    # 7. Model Evaluation
    # --------------------------------------------------
    y_test_pred = np.dot(X_test_norm, weights) + bias

    mse = np.mean((y_test - y_test_pred) ** 2)
    print("\nTest MSE:", mse)

    ss_res = np.sum((y_test - y_test_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print("Test R²:", r2)


# --------------------------------------------------
# Run Script
# --------------------------------------------------
if __name__ == "__main__":
    main()
