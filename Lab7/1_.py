import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

def main():

    # Load SONAR dataset
    data = pd.read_csv("sonar.csv", header=None)

    # Features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Convert labels (Rock, Metal) → (0,1)
    y = y.map({'R':0, 'M':1})

    # Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # 10-fold cross validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=kfold)

    print("Accuracy for each fold:", scores)
    print("Average accuracy:", scores.mean())


if __name__ == "__main__":
    main()



# ______________________________________________________________

import numpy as np
import pandas as pd


# -----------------------------
# Load Dataset
# -----------------------------
def load_data(filename):

    data = pd.read_csv(filename, header=None)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Convert Rock/Metal to 0/1
    y = np.where(y == 'M', 1, 0)

    return X, y


# -----------------------------
# Standardize Data
# -----------------------------
def standardize(X_train, X_test):

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


# -----------------------------
# Add Intercept
# -----------------------------
def add_intercept(X):

    ones = np.ones((X.shape[0],1))
    X = np.hstack((ones, X))

    return X


# -----------------------------
# Sigmoid
# -----------------------------
def sigmoid(z):

    return 1 / (1 + np.exp(-z))


# -----------------------------
# Logistic Regression Training
# -----------------------------
def train_logistic(X, y, lr=0.01, iterations=1000):

    m, n = X.shape

    theta = np.zeros(n)

    for _ in range(iterations):

        z = np.dot(X, theta)
        h = sigmoid(z)

        gradient = np.dot(X.T, (h - y)) / m

        theta = theta - lr * gradient

    return theta


# -----------------------------
# Prediction
# -----------------------------
def predict(X, theta):

    z = np.dot(X, theta)
    probs = sigmoid(z)

    return (probs >= 0.5).astype(int)


# -----------------------------
# Accuracy
# -----------------------------
def accuracy(y_true, y_pred):

    return np.mean(y_true == y_pred)


# -----------------------------
# 10-Fold Cross Validation
# -----------------------------
def k_fold_cv(X, y, k=10):

    n = len(X)

    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_size = n // k

    scores = []

    for i in range(k):

        start = i * fold_size
        end = start + fold_size

        test_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        # Standardize
        X_train, X_test = standardize(X_train, X_test)

        # Add intercept
        X_train = add_intercept(X_train)
        X_test = add_intercept(X_test)

        # Train logistic regression
        theta = train_logistic(X_train, y_train)

        # Predict
        y_pred = predict(X_test, theta)

        acc = accuracy(y_test, y_pred)

        print("Fold", i+1, "Accuracy:", acc)

        scores.append(acc)

    print("Average Accuracy:", np.mean(scores))


# -----------------------------
# Main Function
# -----------------------------
def main():

    X, y = load_data("sonar.csv")

    print("Dataset Shape:", X.shape)

    k_fold_cv(X, y, 10)


if __name__ == "__main__":
    main()