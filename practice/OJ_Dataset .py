import numpy as np
import pandas as pd
from ISLP import load_data

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ================== LOAD DATA ==================
df = load_data("OJ")

# Target encoding
df['Purchase'] = df['Purchase'].map({'CH': 0, 'MM': 1})

# Binary encoding
df['Store7'] = df['Store7'].map({'No': 0, 'Yes': 1})

# One-hot encoding
df = pd.get_dummies(df, columns=['STORE'], drop_first=True)

# ================== FEATURES & TARGET ==================
X = df.drop(columns=['Purchase'])
y = df['Purchase']

# ================== TRAIN-TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=800, random_state=42
)

# ================== PIPELINE ==================
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# ================== PARAMETER GRID ==================
param_grid = [
    {
        'svm__kernel': ['linear'],
        'svm__C': [0.01, 0.1, 1, 10, 100]
    },
    {
        'svm__kernel': ['rbf'],
        'svm__C': [0.01, 0.1, 1, 10, 100],
        'svm__gamma': ['scale', 0.001, 0.01, 0.1, 1]
    }
]

# ================== GRID SEARCH ==================
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# ================== BEST MODEL ==================
best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# ================== TEST PERFORMANCE ==================
y_pred = best_model.predict(X_test)

print("\nTuned SVM Results:")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================== BASELINE MODELS ==================

# Linear SVM
linear_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear', C=1))
])
linear_svm.fit(X_train, y_train)

print("\nLinear SVM:")
print("Test Accuracy:", accuracy_score(y_test, linear_svm.predict(X_test)))

# RBF SVM (default)
rbf_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])
rbf_svm.fit(X_train, y_train)

print("\nRBF SVM (Default):")
print("Test Accuracy:", accuracy_score(y_test, rbf_svm.predict(X_test)))

#
# import numpy as np
# import pandas as pd
# from ISLP import load_data
#
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report
#
# from xgboost import XGBClassifier
#
# # ================== LOAD DATA ==================
# df = load_data("OJ")
#
# # Target encoding
# df['Purchase'] = df['Purchase'].map({'CH': 0, 'MM': 1}).astype(int)
#
# # Binary encoding
# df['Store7'] = df['Store7'].map({'No': 0, 'Yes': 1}).astype(int)
#
# # One-hot encoding
# df = pd.get_dummies(df, columns=['STORE'], drop_first=True)
#
# # ================== FEATURE ENGINEERING ==================
# df['PriceDiff'] = df['PriceMM'] - df['PriceCH']
# df['DiscDiff'] = df['DiscMM'] - df['DiscCH']
# df['PctDiscDiff'] = df['PctDiscMM'] - df['PctDiscCH']
# df['PriceRatio'] = df['PriceMM'] / (df['PriceCH'] + 1e-5)
#
# df['Loyalty_PriceDiff'] = df['LoyalCH'] * df['PriceDiff']
# df['DiscRatio'] = df['DiscMM'] / (df['DiscCH'] + 1e-5)
#
# # ================== FEATURES & TARGET ==================
#
# X = df.drop(columns=['Purchase'])
# X = X.astype(float)
# y = df['Purchase']
#
# # ================== TRAIN TEST SPLIT ==================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=800, random_state=42 ,stratify=y
# )
#
# # ================== XGBOOST MODEL ==================
# xgb = XGBClassifier(
#     eval_metric='logloss',
#     random_state=42,
# )
#
# # ================== HYPERPARAMETER TUNING ==================
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.8, 1],
#     'colsample_bytree': [0.8, 1]
# }
#
# grid = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid.fit(X_train, y_train)
#
# # ================== RESULTS ==================
# best_xgb = grid.best_estimator_
#
# print("Best Parameters:", grid.best_params_)
# print("Best CV Score:", grid.best_score_)
#
# y_pred = best_xgb.predict(X_test)
#
# print("\nXGBoost Results:")
# print("Test Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
#
#
# # ================== FEATURE IMPORTANCE ==================
# import matplotlib.pyplot as plt
#
# importance = best_xgb.feature_importances_
# features = X.columns
#
# feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)
#
# print("\nTop Features:\n", feat_imp.head(10))
#
# feat_imp.head(10).plot(kind='bar')
# plt.title("Top Feature Importance")
# plt.show()
#
# # ================== FINAL SUMMARY REPORT ==================
#
# # ================== DATA UNDERSTANDING ==================
#
# print("\n" + "="*50)
# print("        DATA & MODEL INSIGHTS")
# print("="*50)
#
# # Class distribution
# print("\n1. Class Distribution:")
# print(f"  Citrus Hill (CH - 0): {sum(y==0)} samples")
# print(f"  Minute Maid (MM - 1): {sum(y==1)} samples")
#
# # Feature importance insights
# top_features = feat_imp.head(5)
#
# print("\n2. Top Influential Factors:")
# for i, (feature, importance) in enumerate(top_features.items(), start=1):
#     print(f"  {i}. {feature} (importance: {importance:.4f})")
#
# # Behavioral insights
# print("\n3. Key Observations:")
#
# print("  • Brand loyalty (LoyalCH) is the strongest factor influencing purchase decisions.")
# print("  • Customers with high loyalty tend to stick to their preferred brand.")
#
# print("  • Discount-based features (DiscRatio, DiscDiff) significantly affect switching behavior.")
# print("  • Customers respond more to relative discounts rather than absolute prices.")
#
# print("  • Price difference (PriceDiff) impacts decisions, but less than loyalty and discounts.")
#
# print("  • Promotional strategies (SpecialCH) influence customer choices at stores.")
#
# # Model interpretation
# print("\n4. Model Understanding:")
#
# print("  • The model captures both behavioral (loyalty) and economic (price/discount) factors.")
# print("  • Higher recall for MM indicates improved detection of switching customers.")
# print("  • Balanced precision and recall suggests a well-generalized model.")
#
# # Conclusion
# print("\n5. Conclusion:")
#
# print("  The purchasing decision is primarily driven by brand loyalty,")
# print("  but discounts and promotions play a key role in influencing brand switching.")
# print("  The model successfully captures real-world consumer behavior patterns.")
#
# print("="*50)