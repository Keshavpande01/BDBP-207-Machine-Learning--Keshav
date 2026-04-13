import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:,0:6].values
    y = data.iloc[:,6].values
    return X,y

# -----------------------------
# Add Intercept
# -----------------------------
def add_intercept(X):
    x0 = np.ones((X.shape[0],1))
    X = np.hstack((x0,X))
    return X

# -----------------------------
# Hypothesis function
# -----------------------------
def hypothesis(X,thetas):
    predictions = []
    for i in range(len(X)):
        h = 0
        for j in range(len(thetas)):
            h += thetas[j] * X[i][j]
        predictions.append(h)
    return predictions

# -----------------------------
# Gradient Descent
# -----------------------------
def gradient_descent(X,y,alpha=0.01,iterations=1000):
    m,n = X.shape
    thetas = [0]*n
    for iteration in range(iterations):
        preds = hypothesis(X,thetas)
        new_thetas = thetas.copy()
        for j in range(n):
            gradient = 0
            for i in range(m):
                gradient += (preds[i] - y[i]) * X[i][j]
            gradient = gradient / m
            new_thetas[j] = thetas[j] - alpha * gradient
        thetas = new_thetas
    return thetas

# -----------------------------
# Prediction
# -----------------------------
def predict(X,thetas):
    preds = []
    for i in range(len(X)):
        h = 0
        for j in range(len(thetas)):
            h += thetas[j] * X[i][j]
        preds.append(h)
    return preds


# -----------------------------
# R2 Score
# -----------------------------
def r2_score(y_true,y_pred):
    y_mean = sum(y_true)/len(y_true)
    ss_total = 0
    ss_res = 0
    for i in range(len(y_true)):
        ss_total += (y_true[i] - y_mean)**2
        ss_res += (y_true[i] - y_pred[i])**2
    r2 = 1 - (ss_res/ss_total)
    return r2


# -----------------------------
# K Fold Cross Validation
# -----------------------------
def k_fold_cv(X,y,k=5):
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    scores = []
    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size
        test_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start],indices[end:]))

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        # Standardize
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Add intercept
        X_train = add_intercept(X_train)
        X_test = add_intercept(X_test)

        # Train model
        thetas = gradient_descent(X_train,y_train,alpha=0.01,iterations=1000)

        # Prediction
        y_pred = predict(X_test,thetas)

        # R2
        score = r2_score(y_test,y_pred)

        scores.append(score)

        print("Fold",fold+1,"R2 =",score)

    print("Average R2 =",sum(scores)/len(scores))


# -----------------------------
# Main Function
# -----------------------------
def main():

    X,y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")
    print("Dataset shape:",X.shape)
    k_fold_cv(X,y,k=15)

if __name__ == "__main__":
    main()