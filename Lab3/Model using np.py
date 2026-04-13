import pandas as pd
import numpy as np


def cost_function(X, y, theta):
    m = len(y)
    y_pred = X @ theta   # Hypothesis function
    return np.sum((y_pred - y) ** 2) / (2 * m)


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)

    for i in range(iterations):

        y_pred = X @ theta
        gradient = (1/m) * (X.T @ (y_pred - y))

        theta = theta - alpha * gradient

        loss = np.mean((y - y_pred) ** 2)
        print(f"Iteration {i+1}, Loss: {loss:.4f}")

    return theta


def r2_score(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)


def main():

    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

    X = df.iloc[:, :-1].values
    y = df.iloc[:,-1].values

    # Feature scaling
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Add bias column
    m = X.shape[0]
    X = np.c_[np.ones(m), X]

    theta = np.zeros(X.shape[1])

    alpha = 0.01
    iterations = 1000

    theta = gradient_descent(X, y, theta, alpha, iterations)

    print("\nFinal Cost:", cost_function(X, y, theta))
    print("Final Theta:", theta)

    y_pred = X @ theta
    print("Final R2:", r2_score(y, y_pred))


if __name__ == "__main__":
    main()

