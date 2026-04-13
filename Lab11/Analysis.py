
#----------To check if a dataset has categorical data, you can inspect the data types of columns.-----------------
import pandas as pd

def main():

    data = pd.read_csv("breast-cancer_2.csv")

    categorical_cols = []

    for col in data.columns:
        if data[col].dtype == "object":
            categorical_cols.append(col)

    print("Categorical columns:", categorical_cols)

    for col in categorical_cols:

        unique_vals = list(set(data[col]))
        mapping = {}

        for i,val in enumerate(unique_vals):
            mapping[val] = i

        data[col] = [mapping[v] for v in data[col]]

    print("\nEncoded Dataset:")
    print(data.head())
    print(data.shape)
    print(data.isnull().sum())

if __name__ == "__main__":
    main()