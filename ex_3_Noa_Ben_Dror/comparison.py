from models import *
import matplotlib.pyplot as plt


def f(X):
    array = np.array([0.3, -0.5])
    return np.sign(np.dot(X, array) + 0.1)


def draw_points(m):
    X = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=m)
    y = f(X)
    while np.all(y == 1) or np.all(y == -1):
        X = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=m)
        y = f(X)
    return X, y


def Q9():
    M = np.array([5, 10, 15, 25, 70])
    for m in M:
        X, y = draw_points(m)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        points = [-5, 5]
        y_one = X[np.where(y == 1)]
        y_minus_one = X[np.where(y == -1)]
        plt.scatter(y_one[:, 0], y_one[:, 1], c="blue", label="class 1")
        plt.scatter(y_minus_one[:, 0], y_minus_one[:, 1], c="orange", label="class -1")

        #  true hypothesis
        plt.plot(points, [-5 * 0.3 / 0.5 + 0.1 / 0.5, 5 * 0.3 / 0.5 + 0.1 / 0.5], label='true hypothesis')

        # perceptron
        perceptron = Perceptron()
        perceptron.fit(X, y)
        plt.plot(points, [5 * perceptron.model[0] / perceptron.model[1] - perceptron.model[2] / perceptron.model[1],
                          -5 * perceptron.model[0] / perceptron.model[1] - perceptron.model[2] / perceptron.model[1]],
                          label='Perceptron')


        # SVM
        svm = SVM()
        svm.fit(X, y)
        plt.plot(points, [5 * svm.model.coef_[0, 0] / svm.model.coef_[0, 1] - svm.model.intercept_ / svm.model.coef_[0,1],
                          -5 * svm.model.coef_[0, 0] / svm.model.coef_[0, 1] - svm.model.intercept_ / svm.model.coef_[0, 1]],
                           label='SVM')

        plt.title("Data Points and separating hyperplanes: " + str(m) + " points")
        plt.legend()
        plt.show()


def Q10():
    M = np.array([5, 10, 15, 25, 70])
    k = 10000
    n = 500
    perceptron_mean_acc = []
    SVM_mean_acc = []
    LDA_mean_acc = []

    for m in M:
        perceptron_acc = 0
        SVM_acc = 0
        LDA_acc = 0

        for i in range(n):
            X_train, y_train = draw_points(m)
            X_test, y_test = draw_points(k)

            perceptron = Perceptron()
            perceptron.fit(X_train, y_train)

            svm = SVM()
            svm.fit(X_train, y_train)

            lda = LDA()
            lda.fit(X_train, y_train)

            perceptron_acc += perceptron.score(X_test, y_test)['accuracy']
            SVM_acc += svm.score(X_test, y_test)['accuracy']
            LDA_acc += lda.score(X_test, y_test)['accuracy']

        perceptron_mean_acc.append(perceptron_acc / n)
        SVM_mean_acc.append(SVM_acc / n)
        LDA_mean_acc.append(LDA_acc / n)

    plt.plot(M, perceptron_mean_acc, label='Perceptron')
    plt.plot(M, SVM_mean_acc, label='SVM')
    plt.plot(M, LDA_mean_acc, label='LDA')
    plt.legend()
    plt.xlabel('size of training data')
    plt.ylabel('accuracy')
    plt.title("Accuracy per size of training data")
    plt.show()

Q9()
Q10()


