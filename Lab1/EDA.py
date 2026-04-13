import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # Load dataset
    df = pd.read_csv("data.csv")

    # 1. Dataset overview
    print("Shape of dataset:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns)

    # 2. Dataset information
    print("\nDataset Info:")
    print(df.info())

    # 3. Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # 4. Duplicate rows
    print("\nDuplicate rows:", df.duplicated().sum())

    # 5. Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())

    # 6. Correlation matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

    # 7. Feature distributions
    df.hist(figsize=(12,10))
    plt.tight_layout()
    plt.show()

    # 8. Boxplot for outliers
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.title("Outlier Detection")
    plt.show()


if __name__ == "__main__":
    main()