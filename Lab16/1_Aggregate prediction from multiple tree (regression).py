import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def aggregate_predictions(X, y, n_trees=40, max_depth=6, test_size=0.3, random_state=42):
    np.random.seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train trees on bootstrap samples and collect predictions
    predictions = []
    for i in range(n_trees):
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state + i)
        tree.fit(X_train[indices], y_train[indices])
        predictions.append(tree.predict(X_test))

    # Average predictions
    final_pred = np.array(predictions).mean(axis=0)

    # Evaluate
    print(f"Ensemble R²:  {r2_score(y_test, final_pred):.4f}")
    print(f"Ensemble MSE: {mean_squared_error(y_test, final_pred):.4f}")

    # Compare with single tree
    single_tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    single_tree.fit(X_train, y_train)
    single_pred = single_tree.predict(X_test)

    print(f"\nSingle Tree R²:  {r2_score(y_test, single_pred):.4f}")
    print(f"Single Tree MSE: {mean_squared_error(y_test, single_pred):.4f}")


def main():
    X, y = fetch_california_housing(return_X_y=True)
    aggregate_predictions(X, y)


if __name__ == "__main__":
    main()