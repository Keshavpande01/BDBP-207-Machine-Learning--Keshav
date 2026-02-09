# Implement normal equations method from scratch and compare your results on a simulated dataset (disease score fluctuation as target)
# and the admissions dataset (https://www.kaggle.com/code/erkanhatipoglu/linear-regression-using-the-normal-equation ).
# You can compare the results with scikit-learn and your own gradient descent implementation.

### Closed-form solution (for simulated datasets) -

import pandas as pd
import numpy as np

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data.drop(columns=['disease_score', 'disease_score_fluct']).values
y = data['disease_score_fluct'].values
X = np.c_[np.ones(X.shape[0]), X]     # Adding a bias term to X
y = y.reshape(-1, 1)     # Normalise y
theta = np.zeros(X.shape[1])
num_of_iterations = 1000
learning_rate = 0.01

def hypothesis_func(X, theta):
    return np.dot(X, theta)

def cost_func(X, theta, y):
    error = np.dot(X, theta) - y
    cost = 0.5 * np.dot(error.T, error)
    return cost.item()     # .item() - converts 1 by 1 matrix into scalar form

def compute_gradient(X, error):
    return np.dot(X.T, error)

def compute_theta(X, y):
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

def main():
    theta_closed_form = compute_theta(X, y)      # Compute theta using closed-form solution (normal equation)

    from sklearn.linear_model import LinearRegression    # Comparison with scikit-learn
    from sklearn.metrics import mean_squared_error, r2_score
    model = LinearRegression()
    model.fit(X[:, 1:], y)     # We use X without the intercept term for sklearn

    y_pred_closed_form = hypothesis_func(X, theta_closed_form)
    y_pred_sklearn = model.predict(X[:, 1:])

    mse_closed_form = mean_squared_error(y, y_pred_closed_form)
    mse_sklearn = mean_squared_error(y, y_pred_sklearn)

    r2_closed_form = r2_score(y, y_pred_closed_form)
    r2_sklearn = r2_score(y, y_pred_sklearn)

    print(f"Closed-form coefficients: {theta_closed_form.ravel()}")
    print(f"Scikit-learn coefficients: {model.coef_.ravel()}")

    print(f"Closed-form MSE: {mse_closed_form}")
    print(f"Scikit-learn MSE: {mse_sklearn}")

    print(f"Closed-form R²: {r2_closed_form}")
    print(f"Scikit-learn R²: {r2_sklearn}")

if __name__ == "__main__":
    main()

