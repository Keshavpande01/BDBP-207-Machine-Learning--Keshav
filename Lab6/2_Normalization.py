# Data normalization - scale the values between 0 and 1. Implement code from scratch.

import pandas as pd
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data.loc[:, 'age':'blood_sugar']

def normalize_data(X):
    min_X = X.min()
    max_X= X.max()
    new_X = (X - min_X) / (max_X - min_X)
    return new_X

def main():
    normalized_X = normalize_data(X)
    print("Normalised features: ")
    print(normalized_X)

if __name__ == "__main__":
    main()