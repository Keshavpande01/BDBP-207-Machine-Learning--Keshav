import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -------------------------
# Load Dataset
# -------------------------
def load_data():

    df = pd.read_csv("Caravan.csv")

    print("Dataset Shape:", df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


# -------------------------
# EDA
# -------------------------
def perform_eda(df):

    print("\nStatistical Summary:")
    print(df.describe())

    numeric_df = df.select_dtypes(include=['int64','float64'])

    plt.figure(figsize=(12,10))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    numeric_df.hist(figsize=(12,10))
    plt.tight_layout()
    plt.show()


# -------------------------
# Prepare Data
# -------------------------
def prepare_data(df):

    y = df["Purchase"]

    X = df.drop("Purchase", axis=1)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# -------------------------
# Logistic Regression
# -------------------------
# def logistic_model(X,y):
#
#     model = LogisticRegression(max_iter=2000)
#
#     kf = KFold(n_splits=10, shuffle=True, random_state=42)
#
#     scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
#
#     print("\nLogistic Regression Results")
#     print("Mean Accuracy:", scores.mean())
#     print("Std Dev:", scores.std())

def logistic_model(X, y):

    model = LogisticRegression(max_iter=2000)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    scores = []

    fold = 1

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        scores.append(acc)

        print(f"Fold {fold} Accuracy: {acc}")

        fold += 1

    print("\nMean Accuracy:", np.mean(scores))
    print("Std Dev:", np.std(scores))



# -------------------------
# Random Forest
# -------------------------
def random_forest_model(X,y):

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

    print("\nRandom Forest Results")
    print("Mean Accuracy:", scores.mean())
    print("Std Dev:", scores.std())


# -------------------------
# Main
# -------------------------
def main():

    df = load_data()

    perform_eda(df)

    X,y = prepare_data(df)

    logistic_model(X,y)

    random_forest_model(X,y)


if __name__ == "__main__":
    main()