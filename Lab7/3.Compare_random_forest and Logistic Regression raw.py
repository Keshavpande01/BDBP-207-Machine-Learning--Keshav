import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ── Load Dataset ──
data = pd.read_csv("sonar.csv", header=None)

# Features and Target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert labels (R/M → 0/1)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ── Models to Compare ──
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
}

# ── 10-Fold Cross Validation ──
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# ── Compare Models ──
for name, clf in models.items():
    # Scaling is important for Logistic Regression, optional for Random Forest
    if name == "Logistic Regression":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", clf)
        ])
    else:
        pipeline = Pipeline([
            ("model", clf)
        ])

    scores = cross_val_score(pipeline, X, y, cv=kfold, scoring="accuracy")

    print(f"\n{name}:")
    for i, acc in enumerate(scores,1 ):
        print(f"  Fold {i+1}: Accuracy = {acc:.4f}")
    print(f"  Mean Accuracy: {scores.mean():.4f}")
    print(f"  Std Dev: {scores.std():.4f}")