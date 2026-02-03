import pandas as pd
import numpy as np

def hypothesis_fun(X, theta, row_idx):
    total = 0
    for j in range(len(theta)):
        total += theta[j] * X[row_idx][j]
    return total

def cost_function(X, y, theta):
    m = len(y)
    total_cost = 0

    for i in range(m):
        h = hypothesis_fun(X, theta, i)
        total_cost += (h - y[i]) ** 2

    return total_cost / (2 * m)

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)

    for _ in range(iterations):
        new_theta = theta.copy()

        for j in range(len(theta)):
            gradient_sum = 0

            for i in range(m):
                h = hypothesis_fun(X, theta, i)
                gradient_sum += (h - y[i]) * X[i][j]

            new_theta[j] = theta[j] - (alpha / m) * gradient_sum

        theta = new_theta

    return theta


def main():
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Add bias term (x₀ = 1)
    m = X.shape[0]
    X = np.c_[np.ones(m), X]

    # Initialize theta
    theta = np.zeros(X.shape[1])

    alpha = 0.000001   # small because data is unnormalized
    iterations = 10

    theta = gradient_descent(X, y, theta, alpha, iterations)

    print("Final Cost:", cost_function(X, y, theta))
    print("Final Theta:", theta)


if __name__ == "__main__":
    main()


