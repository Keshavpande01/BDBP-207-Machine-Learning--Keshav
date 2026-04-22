import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split ,cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("Boston.csv")

# Exploratory Data Analysis
print("Overview of the Dataset") # overview of the dataset
print(df.info())
print("Statistical summary of the Dataset") # Statistical summary of the dataset
print(df.describe())
print("Unique values in the Dataset")  # Display the unique values
print(df.nunique())
print("Columns of the Dataset:") # give features column names
print(df.columns)
print("Duplicate values in the Dataset")
print(df.duplicated())
print("Null values in the Dataset")
print(df.isnull().sum())

df.hist(bins = 30, figsize = (10,6)) # feature
plt.show()
df.boxplot(figsize=(10,6))  # outlier detection (Boxplot)
plt.show()
pd.plotting.scatter_matrix(df, figsize = (10,6))  # scatter matrix - to   visualize the pairwise relationships of the returns
plt.show()

# Drop the unnecessary column -
df = df.drop(columns = ["unnamed: 0"])
print(df)

# Define Features (X) and target variable (y) -
X = df.drop(columns = ["medv"])
y=df["medv"]

# split the data into 70% train set and 30 % test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# n estimators : the number of boosting stages that will be performed , max_depth : limits the number of nodes in the tree
# learning_rate : how much the contribution of each tree will shrink , loss = "squared-error" : loss function to optimize ( here , mean squared error )
gb_reg = GradientBoostingClassifier(n_estimators=100 , max_depth=3, learning_rate=0.1, subsample=0.5, random_state=0.1, loss="squared_epsilon_error",)
gb_reg.fit(X_train, y_train) # Train the model

# Evaluating the model's performance -
y_pred = gb_reg.predict(X_val)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# K-fold cross validation -
kf = KFold(n_splits=5, shuffle=True)
cv_kf = cross_val_score(gb_reg, X_train, y_train, cv=kf ,scoring='r2')
print("R2 scores for fold ", cv_kf.std())
print("Mean R2 score: ", cv_kf.mean())

# Feature importance visualization -
plt.figure(figsize=(10,6))
plt.barh(X.columns, gb_reg.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Gradient Boosting Regression")
plt.show()