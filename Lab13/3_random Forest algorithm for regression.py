### Implement Random Forest algorithm for regression using scikit-learn. Use diabetes dataset from scikit-learn.

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

diabetes = load_diabetes()    # loading the diabetes dataset
X, y = diabetes.data, diabetes.target    # X: features, y: target
feature_names = diabetes.feature_names    # column names
df = pd.DataFrame(X, columns=feature_names)    # converting into dataframe

# EDA already done in exercise-1

# Split the dataset into 70% train set & 30% test set -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling -
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model -
rand_forest_reg = RandomForestRegressor(n_estimators=100, max_depth=3, max_features='sqrt', bootstrap=True, random_state=999, n_jobs=-1)
rand_forest_reg.fit(X_train_scaled, y_train)
y_pred = rand_forest_reg.predict(X_test_scaled)
print("Predicted y-values: ")
print(y_pred)

# Evaluate the model -
r2 = r2_score(y_test, y_pred)
print(f"r2 score: {r2}")
mse = mean_squared_error(y_test, y_pred)
print(f"M.S.E.: {mse}")