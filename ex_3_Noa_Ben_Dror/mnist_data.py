from sklearn.neighbors import KNeighborsClassifier

from models import *
import time
import matplotlib.pyplot as plt


data_path = "C:\\Users\\noabe\\OneDrive\\מסמכים\\לימודים\\שנה ג\\מדמח\\IML\\EXS\\EX3"
train_data = np.loadtxt(data_path + "\\mnist_train.csv", delimiter=',')
test_data = np.loadtxt(data_path + "\\mnist_test.csv", delimiter=',')
x_train = train_data[:, 1:]
y_train = train_data[:, 0]
x_test = test_data[:, 1:]
y_test = test_data[:, 0]

train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]


def Q12():
    zeros = x_test[np.where(y_test == 0)]
    ones = x_test[np.where(y_test == 1)]
    num_list = [zeros, ones]
    for num in num_list:
        plt.imshow(num[0].reshape(28, 28))
        plt.show()
        plt.imshow(num[1].reshape(28, 28))
        plt.show()
        plt.imshow(num[2].reshape(28, 28))
        plt.show()


def rearrange_data(X):
    return np.reshape(X, -1)


def Q14():
    M = np.array([50, 100, 300, 500])
    n = 50
    logistical_reg_mean_acc = []
    soft_svm_mean_acc = []
    dec_tree_mean_acc = []
    knn_mean_acc = []
    for m in M:
        logistical_reg_acc = 0
        soft_svm_acc = 0
        dec_tree_acc = 0
        knn_acc = 0
        logistical_reg_train_time = 0
        soft_svm_train_time = 0
        dec_tree_train_time = 0
        knn_train_time = 0
        logistical_reg_test_time = 0
        soft_svm_test_time = 0
        dec_tree_test_time = 0
        knn_test_time = 0
        for i in range(n):
            rand_train = np.random.randint(0, x_train.shape[0], m)
            X_train = x_train[rand_train]
            Y_train = y_train[rand_train]
            while np.all(Y_train == 1) or np.all(Y_train == -1):
                rand_train = np.random.randint(0, x_train.shape[0], m)
                X_train = x_train[rand_train]
                Y_train = y_train[rand_train]

            start = time.time()
            logistical_reg = LogisticRegression()
            logistical_reg.fit(X_train, Y_train)
            end = time.time()
            logistical_reg_train_time += end - start

            start = time.time()
            soft_svm = SVC(C=1, kernel='linear')
            soft_svm.fit(X_train, Y_train)
            end = time.time()
            soft_svm_train_time += end - start

            start = time.time()
            dec_tree = DecisionTreeClassifier()
            dec_tree.fit(X_train, Y_train)
            end = time.time()
            dec_tree_train_time += end - start

            start = time.time()
            knn = KNeighborsClassifier()
            knn.fit(X_train, Y_train)
            end = time.time()
            knn_train_time += end - start

            start = time.time()
            logistical_reg_acc += logistical_reg.score(x_test, y_test)
            end = time.time()
            logistical_reg_test_time += end - start

            start = time.time()
            soft_svm_acc += soft_svm.score(x_test, y_test)
            end = time.time()
            soft_svm_test_time += end - start

            start = time.time()
            dec_tree_acc += dec_tree.score(x_test, y_test)
            end = time.time()
            dec_tree_test_time += end - start

            start = time.time()
            knn_acc += knn.score(x_test, y_test)
            end = time.time()
            knn_test_time += end - start

        logistical_reg_mean_acc.append(logistical_reg_acc / n)
        soft_svm_mean_acc.append(soft_svm_acc / n)
        dec_tree_mean_acc.append(dec_tree_acc / n)
        knn_mean_acc.append(knn_acc / n)

        print("logistical regression time: " + str(logistical_reg_train_time / n +
                                                         logistical_reg_test_time / n) + " m: " + str(m))
        print("soft svm time: " + str(soft_svm_train_time / n +
                                            soft_svm_test_time / n) + " on m: " + str(m))
        print("decision tree time: " + str(dec_tree_train_time / n +
                                                 dec_tree_test_time / n) + " on m: " + str(m))
        print("knn time: " + str(knn_train_time / n
                                       + knn_test_time / n) + " on m: " + str(m))
        print()

    plt.plot(M, logistical_reg_mean_acc, label='logistical regression')
    plt.plot(M, soft_svm_mean_acc, label='soft-SVM')
    plt.plot(M, dec_tree_mean_acc, label='decision tree')
    plt.plot(M, knn_mean_acc, label='k-nearest neighbors')
    plt.title('accuracy per size of training data')
    plt.xlabel('size of training data')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


Q14()