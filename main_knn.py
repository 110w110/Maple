from input_data import *
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from knn import Knn

def normalize(arr):
    normalized = [[] for _ in range(len(arr))]
    for x in range(len(arr[0])):
        mx = -sys.maxsize
        mn = sys.maxsize
        for y in range(len(arr)):
            mx = max(mx, arr[y][x])
            mn = min(mn, arr[y][x])
        for y in range(len(arr)):
            normalized[y].append((arr[y][x] - mn) / (mx-mn))
    # print(*normalized,sep='\n')
    return normalized
# x = np.array([0,1,2,3,4])
# y = x * 2 + 1
file = load_xls('./data/maple.xlsx')
start_date = get_single_column(file, 'sheet1', 'B2', 'B99')
# print(len(start_date)) 97
forsythia_flowering = get_single_column(file, 'sheet6', 'D2', 'D99')
# print(len(forsythia_flowering)) # 97
for i in range(len(start_date)):
    # TODO:날짜 파싱
    start_date[i] = (int(start_date[i].strftime("%Y%m%d")[4:6])-10)*31 + int(start_date[i].strftime("%Y%m%d")[6:])
    forsythia_flowering[i] = (int(forsythia_flowering[i].strftime("%Y%m%d")[4:6])-3)*31 + int(forsythia_flowering[i].strftime("%Y%m%d")[6:])
    # start_date[i] = float(start_date[i].strftime("%Y%m%d")[6:])

# print(len(start_date)) 97
min_temper = get_single_column(file, 'sheet2', 'D14', 'D919')
max_temper = get_single_column(file, 'sheet2', 'E14', 'E919')
# print(len(min_temper)) 905
rain_weight = get_single_column(file, 'sheet3', 'B2', 'B99')
sun_summer = get_singleline_data(file, 'sheet7', 'B6', 'CU6')
sun_fall = get_singleline_data(file, 'sheet7', 'B7', 'CU7')
print(len(sun_fall))
# rain_time = get_single_column(file, 'sheet3', 'C2', 'C35')
for i in range(393, -1, -1):
    if 0 <= i % 12 < 5 or 10 <= i % 12:
        del (min_temper[i])
        del (max_temper[i])
        # del (rain_weight[i])
# print(len(min_temper)) 676
# TODO
#   세로로 해야함;
# min_temper = normalize(min_temper)
# max_temper = normalize(max_temper)
# sun_summer = normalize(sun_summer)
# forsythia_flowering = normalize(forsythia_flowering)

# np.array(min_temper[i]).astype(float)
# np.array(max_temper[i]).astype(float)
# np.array(sun_summer[i]).astype(float)
x = []
for i in range(97):
    v = []
    # for j in range(5):
    #     v.append(min_temper[i*5+j])
    # for j in range(5):
    #     v.append(max_temper[i*5+j])
    v.append(rain_weight[i])
    v.append(forsythia_flowering[i])
    v.append(sun_summer[i])
    v.append(sun_fall[i])
    x.append(v)
# print(*x,sep='\n')
x = np.array(x)
y = np.array(start_date)
print(x)
print(len(x))
# x = x / x.max()
# y = y / y.max()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.17, shuffle=True)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=1004)
# print('x_test : ', x_test)
# print('y_test : ', y_test)
# print('x_train : ', x_train)
# print('y_train : ', y_train)

train_size = len(y_train)
test_size = len(y_test)

k1 = Knn(x_train, x_test, y_train, y_test)
k = 2
acc = 0
for i in range(test_size):
    result = k1.neighbor(k, k1.distance(i))
    print(i, "th data result ", result, ", label ", y_test[i], sep='', end=' ')

    # print(result, "  label ", end=' ')
    # print(y_test[i], end=' ')
    #if result == y_test[i]:
    #    count = count + 1

    # acc += abs(result - y_test[i])
    acc += 1 - abs(result - y_test[i])/result
    print(", Accuracy : %.2f" % ((1-abs(result-y_test[i])/result)*100), "%", sep='')
print("\nTotal Accuracy : %.4f" % (acc/test_size*100), "%", sep='')
