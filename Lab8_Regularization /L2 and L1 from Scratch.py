import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filename):
    data= pd.read_csv(filename)
    X_df = data.iloc[:,0:6]
    y_df = data.iloc[:,6]
    X = X_df.values
    y = y_df.values
    return X,y


def split_data(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=999)
    return X_train,X_test,y_train,y_test

def standardize_data(X_train,X_test):
    scaler= StandardScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled

def x0_intercept(X_train_scaled):
    x0=np.ones((X_train_scaled.shape[0],1))
    X_train_scale_int=np.hstack((x0,X_train_scaled))
    len_X = len(X_train_scale_int[0])
    return X_train_scale_int,len_X

def initialise_thetas(len_X):
    return [0.0] * len_X

def hypothesis_func(X, thetas):
    predictions = []
    for i in range(len(X)):
        h = 0
        for j in range(len(thetas)):
            h = np.dot(X[i],thetas)
        predictions.append(h)
    return predictions

def cost_func(X, thetas, y, lambda_reg=0.1, reg_type=None):
    m = len(y)
    predictions = hypothesis_func(X, thetas)

    total_error = 0
    for i in range(m):
        total_error += (predictions[i] - y[i])**2

    cost = total_error / (2 * m)

    if reg_type == "l2":
        reg = sum(thetas[j]**2 for j in range(1, len(thetas)))
        cost += (lambda_reg / (2 * m)) * reg

    elif reg_type == "l1":
        reg = sum(abs(thetas[j]) for j in range(1, len(thetas)))
        cost += (lambda_reg / (2 * m)) * reg

    return cost

def gradient_descent(X, thetas, y,
                     alpha=0.01, iterations=1000,
                     lambda_reg=0.1, reg_type=None):

    m = len(y)

    for iteration in range(iterations):
        old_thetas = thetas.copy()
        predictions = hypothesis_func(X, thetas)

        for j in range(len(thetas)):
            gradient = 0

            for i in range(m):
                gradient += (predictions[i] - y[i]) * X[i][j]

            gradient = gradient / m


            if j != 0:
                if reg_type == "l2":
                    gradient += (lambda_reg / m) * thetas[j]

                elif reg_type == "l1":
                    if thetas[j] > 0:
                        gradient += (lambda_reg / m)
                    elif thetas[j] < 0:
                        gradient -= (lambda_reg / m)

            thetas[j] -= alpha * gradient

        # Convergence check
        change = sum(abs(thetas[j] - old_thetas[j]) for j in range(len(thetas)))

        if change < 1e-6:
            break

    return thetas

def run_model(X_train, y_train, X_test, y_test, reg_type=None):
    X_train_int = np.hstack((np.ones((X_train.shape[0],1)), X_train))
    X_test_int = np.hstack((np.ones((X_test.shape[0],1)), X_test))

    thetas = [0] * X_train_int.shape[1]

    thetas = gradient_descent(
        X_train_int,
        thetas,
        y_train,
        alpha=0.01,
        iterations=1000,
        lambda_reg=0.1,
        reg_type=reg_type
    )

    y_pred = hypothesis_func(X_test_int, thetas)

    r2 = r2_score(y_test, y_pred)

    return thetas, r2

def predict_y_test(X_test_scaled,thetas):
    x0=np.ones((X_test_scaled.shape[0],1))
    X_test_scale_int=np.hstack((x0,X_test_scaled))

    y_test_pred=[]
    for i in range(len(X_test_scale_int)):
        X_test_row=X_test_scale_int[i]
        h=0
        for j in range(len(thetas)):
            h+= float(thetas[j] * X_test_row[j])
        y_test_pred.append(h)
    return y_test_pred

def r2_score(y_test,y_test_pred):
    y_test_list=y_test
    y_test_mean=sum(y_test_list)/len(y_test_list)

    sum_of_square_t=0
    for i in range(len(y_test_list)):   #total sum
        sum_of_square_t +=((y_test_list[i] - y_test_mean)**2)

    sum_of_square_r=0
    for i in range(len(y_test_list)):   #residual sum
        sum_of_square_r+=((y_test_list[i] - y_test_pred[i])**2)


    r2= 1 - (sum_of_square_r/sum_of_square_t)
    return r2
def main():
    X, y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    print("\n--- NO REGULARIZATION ---")
    t0, r0 = run_model(X_train_scaled, y_train, X_test_scaled, y_test, reg_type=None)
    print("Thetas:", t0)
    print("R2:", r0)

    print("\n--- L2 (RIDGE) ---")
    t2, r2 = run_model(X_train_scaled, y_train, X_test_scaled, y_test, reg_type="l2")
    print("Thetas:", t2)
    print("R2:", r2)

    print("\n--- L1 (LASSO) ---")
    t1, r1 = run_model(X_train_scaled, y_train, X_test_scaled, y_test, reg_type="l1")
    print("Thetas:", t1)
    print("R2:", r1)

if __name__== "__main__":
    main()