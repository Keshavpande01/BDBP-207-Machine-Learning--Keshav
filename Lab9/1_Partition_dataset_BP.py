import pandas as pd

def partition_data(data, threshold):

    left = data[data["BP"] <= threshold]
    right = data[data["BP"] > threshold]

    print("\nThreshold:", threshold)

    print("Left Partition (BP <=", threshold, ")")
    print(left)

    print("\nRight Partition (BP >", threshold, ")")
    print(right)


def main():

    # Load dataset
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

    print("Original Dataset")
    print(data.head())

    # Threshold = 80
    partition_data(data, 80)

    # Threshold = 78
    partition_data(data, 78)

    # Threshold = 82
    partition_data(data, 82)


if __name__ == "__main__":
    main()
