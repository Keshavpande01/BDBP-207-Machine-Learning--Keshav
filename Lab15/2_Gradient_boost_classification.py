import pandas as pd
from ISLP import load_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load Weekly dataset
weekly = load_data("Weekly")

# Features and target
X = weekly.drop(columns=["Direction", "Today"])
y = LabelEncoder().fit_transform(weekly["Direction"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
gb_clf = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train
gb_clf.fit(X_train, y_train)

# Predict
y_pred = gb_clf.predict(X_test)

# Evaluation
print("\n===== Classification Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))