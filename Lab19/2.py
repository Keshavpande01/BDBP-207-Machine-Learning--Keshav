### For the heart.csv dataset, build a logistic regression classifier to predict the risk of heart disease.  Vary the threshold to generate multiple confusion matrices.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from Q1 import compute_confusion_matrix, calculate_accuracy, calculate_precision, calculate_sensitivity, calculate_specificity, calculate_F1_score, roc_curve, compute_auc, plot_roc_curve

df = pd.read_csv("Heart.csv")    # load the data

# EDA -
print(df.head())
print(df.info())    # data types
print(df.describe())    # summary statistics
print(df.shape)   # dimensions
print(df.columns)   # column names
print(df.isnull().sum()/len(df) * 100)   # missing values %

# Handle missing values
print(df["Ca"].skew())   # skewness > 0  (do median imputation)
df["Ca"]=df["Ca"].fillna(df["Ca"].median())
# Since "Thal" is categorical , we need to apply mode imputation -
df["Thal"]=df["Thal"].fillna(df["Thal"].mode()[0])
print(df.isnull().sum()/len(df) * 100)    # Just to check if the imputation has been done correctly

print(df.nunique())
print(df.duplicated().sum())   # Check duplicates

df = df.drop("Unnamed: 0", axis=1)    # Irrelevant column - not needed
print(df.columns)

# Plots -
df.hist(bins=30, figsize=(10,6))
plt.show()     # distribution of each feature
df.boxplot(figsize=(10,6))
plt.show()   # outlier detection
pd.plotting.scatter_matrix(df,figsize=(10,6))
plt.show()    # pairwise feature relationships

# Data pre-processing -
# One-hot encoding -
categorical_cols = ["ChestPain", "Thal"]   # features to be encoded
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))    # Applying the encoder to the categorical columns
encoded_features.columns  = encoder.get_feature_names_out(categorical_cols)    # renaming the columns
df = df.drop(columns=categorical_cols)    # dropping the original columns
df = pd.concat([df, encoded_features], axis=1)   # merging the encoded columns into the dataframe

# Label encoding of the target variable -
df["AHD"] = LabelEncoder().fit_transform(df["AHD"])

# Splitting the dataset (70% - train set, 30% - test set) -
X = df.drop(columns=["AHD"])   # Features
y = df["AHD"]   # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the data (data normalization) -
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)    # fit on training data
X_test_scaled = scaler.transform(X_test)    # transform test data

# Training the model -
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
predicted_probs = model.predict_proba(X_test_scaled)[:, 1]    # Get probabilities of class 1
thresholds = [0.7, 0.5, 0.4, 0.2]
confusion_matrices = compute_confusion_matrix(y_test, predicted_probs, thresholds)
for threshold, (TP, FP, TN, FN) in confusion_matrices.items():
    print(f"Threshold: {threshold}")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Accuracy: {calculate_accuracy(TP, TN, FP, FN)}")
    print(f"Precision: {calculate_precision(TP, FP)}")
    print(f"Sensitivity: {calculate_sensitivity(TP, FN)}")
    print(f"Specificity: {calculate_specificity(TN, FP)}")
    print(f"F1 Score: {calculate_F1_score(calculate_precision(TP, FP), calculate_sensitivity(TP, FN))}")
    print("-" * 50)

fpr, tpr = roc_curve(y_test, predicted_probs, thresholds)
plot_roc_curve(fpr, tpr)
auc = compute_auc(fpr, tpr)
print(f"AUC: {auc}")

