from input_data import *
import numpy as np
from sklearn.model_selection import train_test_split
from knn import Knn
import func as fc


file = load_xls('./data/maple.xlsx')
start_date = get_single_column(file, 'sheet1', 'B2', 'B99')
forsythia_flowering = get_single_column(file, 'sheet6', 'D2', 'D99')
for i in range(len(start_date)):
    # TODO:날짜 파싱
    start_date[i] = (int(start_date[i].strftime("%Y%m%d")[4:6])-10)*31 + int(start_date[i].strftime("%Y%m%d")[6:])
    forsythia_flowering[i] = (int(forsythia_flowering[i].strftime("%Y%m%d")[4:6])-3)*31 + int(forsythia_flowering[i].strftime("%Y%m%d")[6:])

min_temper = get_single_column(file, 'sheet2', 'D14', 'D919')
max_temper = get_single_column(file, 'sheet2', 'E14', 'E919')
rain_weight = get_single_column(file, 'sheet3', 'B2', 'B99')
sun_summer = get_singleline_data(file, 'sheet7', 'B6', 'CU6')
sun_fall = get_singleline_data(file, 'sheet7', 'B7', 'CU7')

for i in range(393, -1, -1):
    if 0 <= i % 12 < 5 or 10 <= i % 12:
        del (min_temper[i])
        del (max_temper[i])

x = []
for i in range(97):
    v = []
    for j in range(5):
        v.append(min_temper[i*5+j])
    for j in range(5):
        v.append(max_temper[i*5+j])
    v.append(rain_weight[i])
    v.append(forsythia_flowering[i])
    v.append(sun_summer[i])
    v.append(sun_fall[i])
    x.append(v)

x = np.array(fc.normalize(x))
y = np.array(start_date)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

train_size = len(y_train)
test_size = len(y_test)

k1 = Knn(x_train, x_test, y_train, y_test)
k = 3
acc = 0

sum_of_result = 0
print("Data\tPrediction\tReal\tDifference")
for i in range(test_size):
    result = k1.neighbor(k, k1.distance(i))
    print(i + 1, "th \t", fc.itd(result), "    \t", fc.itd(y_test[i]), sep='', end='')
    print(" \t", abs(result - y_test[i]), " day", sep='')
    sum_of_result += abs(result - y_test[i])
print("="*35)
print("Average : %.2f days" % (sum_of_result/test_size))
