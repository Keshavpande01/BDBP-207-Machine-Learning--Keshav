import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_logistic(X, y, lr=0.01, iters=1000, penalty=None, lam=0.1):

    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(iters):

        z = np.dot(X, theta)
        h = sigmoid(z)

        gradient = (1/m) * np.dot(X.T, (h - y))

        if penalty == "l2":
            gradient += lam * theta

        if penalty == "l1":
            gradient += lam * np.sign(theta)

        theta -= lr * gradient

    return theta


def predict(X, theta):

    probs = sigmoid(np.dot(X, theta))
    return (probs >= 0.5).astype(int)


def accuracy(y_true, y_pred):

    return np.mean(y_true == y_pred)


def main():

    data = pd.read_csv("data.csv")

    data = data.drop(['id','Unnamed: 32'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

    X = data.drop('diagnosis', axis=1).values
    y = data['diagnosis'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3,random_state=999
    )

    penalties = [None, "l1", "l2"]

    for p in penalties:

        theta = train_logistic(X_train, y_train, penalty=p)

        y_pred = predict(X_test, theta)

        print("Penalty:", p, "Accuracy:", accuracy(y_test, y_pred))


if __name__ == "__main__":
    main()