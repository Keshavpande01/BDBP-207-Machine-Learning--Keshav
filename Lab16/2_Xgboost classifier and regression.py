import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, classification_report

# ------------------ CLASSIFICATION ------------------
iris = load_iris()
X_cls, y_cls = iris.data, iris.target

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
)

clf = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,           # row sampling
    colsample_bytree=0.8,    # feature sampling
    eval_metric='mlogloss',
    random_state=42,
)

clf.fit(X_train_cls, y_train_cls)

y_pred_cls = clf.predict(X_test_cls)

print("XGBoost Classifier Accuracy (Iris):", accuracy_score(y_test_cls, y_pred_cls) * 100, "%")
print(classification_report(y_test_cls, y_pred_cls))


# ------------------ REGRESSION ------------------
calif_housing = fetch_california_housing()
X_reg, y_reg = calif_housing.data, calif_housing.target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

reg = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42

)

reg.fit(X_train_reg, y_train_reg)

y_pred_reg = reg.predict(X_test_reg)

print("R2 Score (XGBoost Regressor):", r2_score(y_test_reg, y_pred_reg))
print("MSE (California Housing):", round(mean_squared_error(y_test_reg, y_pred_reg), 2))