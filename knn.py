import numpy as np

class Knn:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.train_size = len(y_train)
        self.test_size = len(y_test)


    def distance(self, n):
        dis = np.array([])

        test_array = np.array(self.x_test[n])

        for j in range(0, self.train_size):
            train_array = np.array(self.x_train[j])
            d = 0.0
            d = np.linalg.norm(test_array - train_array)
            dis = np.append(dis, np.array([d]))
        return dis

    def neighbor(self, k, dis):
        indexarr = np.argsort(dis)
        total = 0.0

        for i in range(k):
            total += (k-i) * self.y_train[indexarr[i]]

        return round(total/(k*(k+1)/2))