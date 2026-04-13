import pandas as pd
import numpy as np


# ----------------------------
# Hypothesis Function
# ----------------------------
def hypothesis(x, theta):

    return np.dot(x, theta)


# ----------------------------
# Stochastic Gradient Descent
# ----------------------------
def stochastic_gradient_descent(X, y, alpha=0.01, epochs=100):

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

        print(f"Epoch {epoch+1} completed")

    return theta


# ----------------------------
# Prediction
# ----------------------------
def predict(X, theta):

    return np.dot(X, theta)


# ----------------------------
# R2 Score
# ----------------------------
def r2_score(y_true, y_pred):

    mean_y = np.mean(y_true)

    ss_total = np.sum((y_true - mean_y)**2)

    ss_res = np.sum((y_true - y_pred)**2)

    return 1 - (ss_res/ss_total)


# ----------------------------
# Main
# ----------------------------
def main():

    data = pd.read_csv("data.csv")

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Add intercept term
    X = np.c_[np.ones(X.shape[0]), X]

    theta = stochastic_gradient_descent(X, y, alpha=0.01, epochs=100)

    print("\nFinal Theta:", theta)

    y_pred = predict(X, theta)

    r2 = r2_score(y, y_pred)

    print("\nR2 Score:", r2)


if __name__ == "__main__":
    main()