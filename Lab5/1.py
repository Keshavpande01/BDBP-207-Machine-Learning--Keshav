
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data= pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data.loc[:, 'age':'Gender'].values
y = data['disease_score'].values.reshape(-1, 1)


X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

def hypothesis(X, theta):
    return X @ theta

def stochastic_gradient_descent(X, y, learning_rate, epochs):
    m, n = X.shape
    theta = np.random.randn(n, 1) * 0.01
    costs = []

    for epoch in range(epochs):
        for i in range(m):
            idx = np.random.randint(0, m)
            X_i = X[idx:idx+1]
            y_i = y[idx:idx+1]

            prediction = hypothesis(X_i, theta)
            error = prediction - y_i

            gradient = X_i.T @ error
            theta = theta - learning_rate * gradient

        # Cost after each epoch
        y_pred = hypothesis(X, theta)
        cost = np.mean((y_pred - y) ** 2)
        costs.append(cost)

    return theta, costs

def main():
    theta, costs = stochastic_gradient_descent(
        X_train, y_train,
        learning_rate=0.001,
        epochs=1000
    )

    # Predictions
    y_pred = hypothesis(X_test, theta)

    # Metrics
    mse = np.mean((y_pred - y_test) ** 2)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R² Score:", r2)


    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Ground Truth (y)")
    plt.ylabel("Predicted Values (ŷ)")
    plt.title("Actual vs Predicted Values")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("SGD Convergence Curve")
    plt.show()

if __name__ == "__main__":
    main()