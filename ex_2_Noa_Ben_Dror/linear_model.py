import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def fit_linear_regression(X, y):
    """
    The function returns two sets of values: the first is a numpy array of the
    coefficients vector ‘w‘ and the second is a numpy array of the singular values of X.
    :param X: numpy array with m rows and d columns
    :param y: numpy array with m rows
    :return: x and singular values
    """
    w = np.linalg.pinv(X).dot(y)
    singular_values = np.linalg.svd(X, compute_uv=False)
    return w, singular_values


def predict(X, w):
    """
    :param X: numpy array with m rows and d columns
    :param w: coefficients vector
    :return: a numpy array with the predicted values by the model
    """
    return X.dot(w)


def mse(true_y, pred_y):
    """
    calculates the mean squared error between 2 vectors
    :param true_y: a response vector
    :param pred_y: a prediction vector
    :return: MSE over the received samples
    """
    return np.mean((true_y - pred_y)**2)


def load_data(file_path):
    """
    loads data from path, and filters it
    :param file_path: path of csv file
    :return: responses vector and filtered design matrix
    """
    # drop irrelevant features
    data = pd.read_csv(file_path)
    for feat in ["id", "lat", "long"]:
        data = data.drop(feat, 1)

    # drop rows with missing values
    data.dropna(how="any", inplace=True)

    # drop irrelevant values
    for feat in ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                 "sqft_above", "yr_built", "zipcode", "sqft_living15", "sqft_lot15"]:
        data = data[data[feat] > 0]

    for feat in ["floors", "sqft_basement", "yr_renovated"]:
        data = data[data[feat] >= 0]

    # make sure date is a number
    data["date"] = data["date"].str.replace("T000000", "")

    # make sure categorical features have relevant categorical values
    data = data[data["waterfront"].isin(range(2))]
    data = data[data["view"].isin(range(5))]
    data = data[data["condition"].isin(range(1,6))]
    data = data[data["grade"].isin(range(1, 14))]

    # make dummies
    data = pd.get_dummies(data, columns=["zipcode"])

    return data["price"], data.drop("price", 1)


def plot_singular_values(singular_values):
    """
    plots singular values
    :param singular_values: a collection of singular values
    :return:
    """
    des_singular_values = np.sort(singular_values)[::-1]
    plt.plot(range(0, len(des_singular_values)), des_singular_values)
    plt.title("Singular values of X")
    plt.xlabel("Singular values")
    plt.ylabel("The value")
    plt.show()


def feature_evaluation(X, y):
    """
    plots for every non-categorical feature, a graph (scatter plot) of the feature values
    and the response values. It then also computes and shows on the graph the Pearson
    Correlation between the feature and the response.
    :param X: design matrix
    :param y: response vector
    :return:
    """

    X = X[[c for c in X.columns if not c.startswith("zipcode")]]
    for column in X:
        col = X[column].values.astype("float32")
        pearson_corr = np.cov(col, y, rowvar=False) / (np.std(col) * np.std(y))
        p = pearson_corr[0][1]
        plt.scatter(col, y)
        plt.title("Q17 - Pearson correlation between "
                  + str(column) + " and price\nwith Correlation "
                                  "of:" + str(p))
        plt.xlabel(column)
        plt.ylabel("price")
        plt.show()


# load the data
y, X = load_data("C:\\Users\\noabe\\OneDrive\\מסמכים\\לימודים\\שנה ג\\מדמח\\IML\\EXS\\EX2\\kc_house_data.csv")

# show correlation between feature and response
feature_evaluation(X, y)

# add intercept column
X.insert(0, "intercept", 1, True)

# make sure there are only numbers
X = X.values.astype("float")

# get w, and singular values, and plot singular values
w_after_fitting, singular_values = fit_linear_regression(X, y)
plot_singular_values(singular_values)

# split the data to train-set and test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
mse_vals = []

# calculate the MSE for each percentage of the trained data
for p in range(1, 101):
    percentage = round(y_train.shape[0] * (p / 100))
    w = fit_linear_regression(X_train[1:percentage], y_train[1:percentage])[0]
    prediction = predict(X_test, w)
    mse_vals.append(mse(y_test, prediction))
plt.plot(mse_vals)
plt.title("Q16 - MSE of model by\n"
          "increasing percentage of data")
plt.xlabel("Percentage of data")
plt.ylabel("Mean square error")
plt.show()