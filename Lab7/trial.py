# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
#
# def main():
#
#     data = pd.read_csv("sonar.csv", header=None)
#
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1].map({'R':0,'M':1})
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_scaled, y)
#
#     # Feature importance
#     importance = model.coef_[0]
#
#     feature_importance = pd.DataFrame({
#         "Feature": X.columns,
#         "Coefficient": importance
#     })
#
#     feature_importance["Absolute Importance"] = abs(feature_importance["Coefficient"])
#
#     feature_importance = feature_importance.sort_values(
#         by="Absolute Importance", ascending=False
#     )
#
#     # print(feature_importance)
#     # import matplotlib.pyplot as plt
#     #
#     # feature_importance.sort_values(
#     #     by="Absolute Importance"
#     # ).tail(10).plot(
#     #     x="Feature",
#     #     y="Absolute Importance",
#     #     kind="barh"
#     # )
#     #
#     # plt.title("Top Contributing Features")
#     # plt.show()
#     top_features = feature_importance.head(10)
#
#     print(top_features)
#
# if __name__ == "__main__":
#     main()
#
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
#
#
# def main():
#
#     # Load dataset
#     data = pd.read_csv("sonar.csv", header=None)
#
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1].map({'R':0,'M':1})
#
#     # Scale data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Train logistic regression to get feature importance
#     model = LogisticRegression(max_iter=2000)
#     model.fit(X_scaled, y)
#
#     # Extract coefficients
#     importance = np.abs(model.coef_[0])
#
#     # Select top 15 features
#     top_features = np.argsort(importance)[-15:]
#
#     print("Top contributing features:", top_features)
#
#     # Reduce dataset to important features
#     X_selected = X.iloc[:, top_features]
#
#     # Pipeline
#     pipeline = Pipeline([
#         ("scaler", StandardScaler()),
#         ("model", LogisticRegression(max_iter=2000))
#     ])
#
#     # Stratified 10-fold CV
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#
#     scores = cross_val_score(pipeline, X_selected, y, cv=cv)
#
#     print("Accuracy per fold:", scores)
#     print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
#
#
# if __name__ == "__main__":
#     main()
#

