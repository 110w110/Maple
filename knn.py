import numpy as np


class Knn:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.train_size = len(y_train)
        self.test_size = len(y_test)
        # self.craft_train = np.zeros((train_size, 56))
        # self.craft_test = np.zeros((test_size, 56))

    # 주석은 784개의 input을 사용한 코드입니다.
    # n번째 test 값에서 각 train 에 떨어진 거리를 계산하고 리스트로 반환
    def distance(self, n):
        dis = np.array([])

        test_array = np.array(self.x_test[n])

        for j in range(0, self.train_size):
            train_array = np.array(self.x_train[j])
            d = 0.0
            d = np.linalg.norm(test_array - train_array)
            dis = np.append(dis, np.array([d]))
        return dis

    # 한 개의 test 값에서 제일 가까운 k개를 찾고 예측 일자 반환
    def neighbor(self, k, dis):
        indexarr = np.argsort(dis)
        # print(indexarr)
        # 0.0으로 초기화된 원소 60개짜리 배열
        # numList = [0.0] * 60

        # k_list = []
        # total = 0.0
        total = 0.0

        # for i in indexarr[0:k]:
        for i in range(k):
            total += (k-i) * self.y_train[indexarr[i]]
            print(self.y_train[indexarr[i]],end=' ')
            # print(total)
            # numList[self.y_train[i]] += 1/dis[i]
            # total += 1/dis[i]
            # numList[self.y_train[i]] = k-i
        print()
        # for i in indexarr[0:k]:
            # print((1/dis[i]) / total, ' * ', self.y_train[i])
            # total2 += (1/dis[i]) / total * self.y_train[i]

        # index = numList.index(max(numList))
        # print(index, numList[index], numList)
        # print(total/(k*(k+1)/2))
        return round(total/(k*(k+1)/2))

    """

    # 아래는 handcraft feature 입니다.
    def handcraft(self):
        # self.train = self.train
        # self.craft_train = np.zeros((60000, 56))
        # craft_test = np.zeros((60000, 56))

        for i in range(0, 60000):
            if i % 10000 == 0:
                print(i)
            tr_count = 0
            te_count = 0
            for j in range(0, 28):
                for k in range(0, 28):
                    # print(self.train)
                    if self.train[i][j*28+k] > 0:
                        tr_count = tr_count + 1
                    if i < 10000 and self.test[i][j*28+k] > 0:
                        te_count = te_count + 1
                self.craft_train[i][j] = tr_count
                if i < 10000:
                    self.craft_test[i][j] = te_count
            tr_count = 0
            te_count = 0
            for j in range(0, 28):
                for k in range(0, 28):
                    if self.train[i][k*28+j] > 0:
                        tr_count = tr_count + 1
                    if i < 10000 and self.test[i][k*28+j] > 0:
                        te_count = te_count + 1
                self.craft_train[i][28 + j] = tr_count
                if i < 10000:
                    self.craft_test[i][28 + j] = te_count

    def distance(self, a):
        dis = np.array([])

        te = np.array(self.craft_test[a])

        for j in range(0, 60000):
            tr = np.array(self.craft_train[j])
            d = 0.0
            d = np.linalg.norm(te - tr)
            dis = np.append(dis, np.array([d]))
        return dis

    # 한 개의 test 값에서 제일 가까운 k개를 찾고 가장 많은 품종을 반환합니다.
    def neighbor(self, k, dis):
        indexarr = np.argsort(dis)
        indexarr[0:k-1]
        numList = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

        k_list = []

        for i in indexarr[0:k]:
            numList[self.y_train[i]] = numList[self.y_train[i]] + 1/dis[i]
            k_list.append(self.y_train[i])

        index = numList.index(max(numList))
        return index
    """