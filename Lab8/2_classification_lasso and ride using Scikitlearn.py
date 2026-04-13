
import pandas as pd
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():

    # Load dataset
    data = pd.read_csv("data.csv")

    # Drop unnecessary columns
    data = data.drop(['id', 'Unnamed: 32'], axis=1)

    # Encode target variable
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    # Features and target
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999
    )

    # Ridge Classifier (L2)
    ridge = RidgeClassifier()
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    # Lasso Classifier (L1)
    lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)

    # Print accuracy
    print("Ridge Accuracy:", accuracy_score(y_test, ridge_pred))
    print("Lasso Accuracy:", accuracy_score(y_test, lasso_pred))


if __name__ == "__main__":
    main()