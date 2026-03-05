import pandas as pd
import numpy as np

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data.loc[:, 'age':'blood_sugar']


def robust_scale_data(X):
    # Step 1: Convert to numpy for manual operations
    X = np.array(X)

    # Step 2: Compute median
    median = np.median(X, axis=0)

    # Step 3: Compute Q1 and Q3
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)

    # Step 4: Compute IQR
    IQR = Q3 - Q1

    # Step 5: Avoid division by zero
    IQR[IQR == 0] = 1

    # Step 6: Scale
    X_robust = (X - median) / IQR

    return X_robust


def main():
    robust_scaled_data = robust_scale_data(X)
    print("Robust Scaled data:")
    print(robust_scaled_data)


if __name__ == "__main__":
    main()