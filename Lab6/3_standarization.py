# Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.

import pandas as pd

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data.loc[:, 'age':'blood_sugar']

def standardize_data(X):
    z_score = (X - X.mean()) / X.std()
    return z_score

def main():
    standardized_data = standardize_data(X)
    print("Standardized data: ")
    print(standardized_data)

if __name__ == "__main__":
    main()