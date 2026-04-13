import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data loading and cleaning
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

# Drop ID column
X = data.iloc[:, 0:5].values   # features
y = data.iloc[:, 5].values    # target]


# Train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor( random_state=42)


# Model training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

#Feature importance
importance = model.feature_importances_

for i, v in enumerate(importance):
    print(f"Feature {i}: {v}")

# Tree visualization
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
plot_tree(model, filled=True)
plt.show()