import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():

    data = pd.read_csv("breast-cancer.csv", header=None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    y = LabelEncoder().fit_transform(y)

    X = OneHotEncoder(sparse_output=False).fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3,random_state=42
    )

    model = DecisionTreeClassifier()

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test,pred))


if __name__ == "__main__":
    main()