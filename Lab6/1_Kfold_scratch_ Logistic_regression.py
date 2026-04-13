import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# -----------------------------
# Load Dataset
# -----------------------------
def load_data(filename):
    # Convert target to Numeric
    data = pd.read_csv(filename)
    X = data.iloc[:,0:-1].values
    y = data.iloc[:,-1].values
    y = np.where(y == 'M', 1, 0)
    return X,y

# -----------------------------
# Add Intercept
# -----------------------------
def add_intercept(X):
    x0 = np.ones((X.shape[0],1))
    X = np.hstack((x0,X))
    return X


# -----------------------------
# Sigmoid Function
# -----------------------------
def sigmoid(z):
    return 1/(1 + np.exp(-z))


# -----------------------------
# Hypothesis Function
# -----------------------------
def hypothesis(X,thetas):
    preds = []
    for i in range(len(X)):
        z = 0
        for j in range(len(thetas)):
            z += thetas[j] * X[i][j]
        preds.append(sigmoid(z))
    return preds


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

            gradient = gradient/m

            new_thetas[j] = thetas[j] - alpha*gradient

        thetas = new_thetas

    return thetas


# -----------------------------
# Prediction
# -----------------------------
def predict(X,thetas):

    preds = []

    for i in range(len(X)):

        z = 0

        for j in range(len(thetas)):
            z += thetas[j] * X[i][j]

        prob = sigmoid(z)

        if prob >= 0.5:
            preds.append(1)
        else:
            preds.append(0)

    return preds


# -----------------------------
# Accuracy
# -----------------------------
def accuracy_score(y_true,y_pred):

    correct = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    acc = correct/len(y_true)

    return acc


# -----------------------------
# K-Fold Cross Validation
# -----------------------------
def k_fold_cv(X,y,k=5):

    n = len(X)

    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_size = n//k

    scores = []

    for fold in range(k):

        start = fold*fold_size
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
        thetas = gradient_descent(X_train,y_train)

        # Predict
        y_pred = predict(X_test,thetas)

        acc = accuracy_score(y_test,y_pred)

        scores.append(acc)

        print("Fold",fold+1,"Accuracy =",acc)

    print("Average Accuracy =",sum(scores)/len(scores))


# -----------------------------
# Main Function
# -----------------------------
def main():

    X,y = load_data("sonar.csv")

    print("Dataset shape:",X.shape)

    k_fold_cv(X,y,k=10)


if __name__ == "__main__":
    main()