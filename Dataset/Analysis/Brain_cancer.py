import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -------------------------
# Load Dataset
# -------------------------
def load_data():

    df = pd.read_csv("BrainCancer.csv")

    print("Dataset Shape:", df.shape)

    print("\nFirst Rows:")
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

    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True)
    plt.title("Correlation Heatmap")
    plt.show()

    numeric_df.hist(figsize=(10,8))
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,5))
    sns.boxplot(data=numeric_df)
    plt.xticks(rotation=90)
    plt.title("Outlier Detection")
    plt.show()


# -------------------------
# Data Preparation
# -------------------------
from sklearn.preprocessing import LabelEncoder, StandardScaler

def prepare_data(df):

    # drop index column
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    # categorical columns
    categorical_cols = ['sex','diagnosis','loc','stereo']

    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # target
    y = df["status"]

    # features
    X = df.drop("status", axis=1)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns


# -------------------------
# Logistic Regression
# -------------------------
def logistic_model(X,y):

    model = LogisticRegression(max_iter=1000)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

    print("\nLogistic Regression Results")
    print("Mean Accuracy:", scores.mean())
    print("Std Dev:", scores.std())

    return model


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

    return model


# -------------------------
# Confusion Matrix
# -------------------------
def plot_confusion_matrix(model, X, y):

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3,random_state=42
    )

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title("Confusion Matrix")

    plt.show()


# -------------------------
# Feature Importance
# -------------------------
def feature_importance(model, X, y, feature_names):

    model.fit(X,y)

    importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature":feature_names,
        "Importance":importance
    }).sort_values(by="Importance",ascending=False)

    print("\nFeature Importance:")
    print(imp_df)

    plt.figure(figsize=(8,5))
    sns.barplot(x="Importance",y="Feature",data=imp_df)
    plt.title("Feature Importance")
    plt.show()


# -------------------------
# Main
# -------------------------
def main():

    df = load_data()

    perform_eda(df)

    X,y,feature_names = prepare_data(df)

    log_model = logistic_model(X,y)

    rf_model = random_forest_model(X,y)

    plot_confusion_matrix(log_model, X, y)

    feature_importance(rf_model, X, y, feature_names)


if __name__ == "__main__":
    main()