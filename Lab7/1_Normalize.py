import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main():

    # Load SONAR dataset
    data = pd.read_csv("sonar.csv", header=None)

    # Features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Create pipeline (Scaler + Logistic Regression)
    pipeline = Pipeline([
        ('model', LogisticRegression(max_iter=1000))
    ])

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
