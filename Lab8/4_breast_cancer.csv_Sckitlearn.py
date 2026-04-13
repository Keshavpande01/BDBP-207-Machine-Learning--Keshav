import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():

    data = pd.read_csv("breast-cancer.csv")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Ordinal Encoding
    oe = OrdinalEncoder()
    X_ord = oe.fit_transform(X)

    # One-Hot Encoding
    ohe = OneHotEncoder(sparse_output=False)
    X_hot = ohe.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_hot, y, test_size=0.3, random_state=42
    )

    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()