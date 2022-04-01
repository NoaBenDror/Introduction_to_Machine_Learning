import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def calc_score(prediction, y):
    dictionary = {}
    error = np.mean(prediction != y)
    accuracy = 1 - error
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for i in range(len(y)):
        if y[i] == -1 and prediction[i] == -1:
            TN += 1
        if y[i] == 1 and prediction[i] == 1:
            TP += 1
        if y[i] == 1 and prediction[i] == -1:
            FN += 1
        if y[i] == -1 and prediction[i] == 1:
            FP += 1

    if (FP + TN) != 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 1

    if (TP + FN) != 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 1

    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 1

    if (FP + TN) != 0:
        specificity = TN / (FP + TN)
    else:
        specificity = 1

    dictionary['num_samples'] = len(y)
    dictionary['error'] = error
    dictionary['accuracy'] = accuracy
    dictionary['FPR'] = FPR
    dictionary['TPR'] = TPR
    dictionary['precision'] = precision
    dictionary['specificity'] = specificity

    return dictionary


class Perceptron:

    def __init__(self):
        self.model = None  # the variables that define hypothesis chosen

    def fit(self, X, y):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        w = np.zeros(X.shape[1])  # initialization of w
        while np.any(np.sign(X.dot(w)) - y):
            for i in range(len(X)):
                if (np.dot(w, X[i]) * y[i]) <= 0:
                    w = w + X[i] * y[i]
                    break  # we found the first misclassification
        self.model = w

    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))  # an entry with value 1 to each vector x
        return np.sign(X.dot(self.model))

    def score(self, X, y):
        return calc_score(self.predict(X), y)


class LDA:

    def __init__(self):
        self.mu_1 = np.array([])
        self.mu_minus_1 = np.array([])
        self.inv_cov = np.array([])
        self.ln_pr_1 = 0
        self.ln_pr_minus_1 = 0

    def fit(self, X, y):
        pr_y_1 = np.sum(y == 1) / len(y)
        self.ln_pr_1 = np.log(pr_y_1)
        self.ln_pr_minus_1 = np.log(1 - pr_y_1)
        X_1 = X[np.where(y == 1)]
        X_minus_1 = X[np.where(y == -1)]
        self.mu_1 = np.sum(X_1, axis=0) / np.sum(y == 1)
        self.mu_minus_1 = np.sum(X_minus_1, axis=0) / np.sum(y == -1)
        cov_1 = (X_1 - self.mu_1).T @ (X_1 - self.mu_1)
        cov_minus_1 = (X_minus_1 - self.mu_minus_1).T @ (X_minus_1 - self.mu_minus_1)
        self.inv_cov = np.linalg.inv((cov_1 + cov_minus_1) / len(y))

    def predict(self, X):
        delta_1 = np.dot(X, self.inv_cov).dot(self.mu_1) -\
                  0.5 * np.dot(self.mu_1.T, self.inv_cov).dot(self.mu_1) + self.ln_pr_1
        delta_minus_1 = np.dot(X, self.inv_cov).dot(self.mu_minus_1) -\
                        0.5 * np.dot(self.mu_minus_1.T, self.inv_cov).dot(self.mu_minus_1) + self.ln_pr_minus_1

        return np.where(delta_1 > delta_minus_1, 1, -1)

    def score(self, X, y):
        return calc_score(self.predict(X), y)


class SVM:

    def __init__(self):
        self.model = SVC(C=1e10, kernel='linear')  # the variables that define hypothesis chosen

    def fit(self, X, y):
         self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return calc_score(self.predict(X), y)


class Logistic:

    def __init__(self):
        self.model = LogisticRegression(solver='liblinear')  # the variables that define hypothesis chosen

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return calc_score(self.predict(X), y)


class DecisionTree:

    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return calc_score(self.predict(X), y)





