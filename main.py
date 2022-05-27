import time

import keras.initializers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Activation
from tensorflow.keras import optimizers
from input_data import *
import numpy as np
from sklearn.model_selection import train_test_split
from knn import Knn

keras.initializers.he_normal(seed=None)
# x = np.array([0,1,2,3,4])
# y = x * 2 + 1
file = load_xls('./data/maple.xlsx')
start_date = get_single_column(file, 'sheet1', 'B2', 'B35')
forsythia_flowering = get_single_column(file, 'sheet6', 'D2', 'D35')
for i in range(len(start_date)):
    # TODO:날짜 파싱
    start_date[i] = (int(start_date[i].strftime("%Y%m%d")[4:6])-10)*31 + int(start_date[i].strftime("%Y%m%d")[6:])
    forsythia_flowering[i] = (int(forsythia_flowering[i].strftime("%Y%m%d")[4:6])-3)*31 + int(forsythia_flowering[i].strftime("%Y%m%d")[6:])
    # start_date[i] = float(start_date[i].strftime("%Y%m%d")[6:])
min_temper = get_single_column(file, 'sheet2', 'D14', 'D408')
max_temper = get_single_column(file, 'sheet2', 'E14', 'E408')
#rain_weight = get_single_column(file, 'sheet3', 'B2', 'B35')
rain_weight = get_single_column(file, 'sheet4', 'C2', 'C409')
# rain_time = get_single_column(file, 'sheet3', 'C2', 'C35')
for i in range(393, -1, -1):
    if 0 <= i % 12 < 7 or 9 <= i % 12:
        del (min_temper[i])
        del (max_temper[i])
        del (rain_weight[i])
        # del (rain_weight[i])
np.array(min_temper[i]).astype(float)
np.array(max_temper[i]).astype(float)
np.array(rain_weight[i]).astype(float)
x = []
for i in range(33):
    v = []
    for j in range(2):
        v.append(min_temper[i*2+j])
    for j in range(2):
        v.append(max_temper[i*2+j])
    for j in range(2):
        v.append(rain_weight[i*2+j])
    v.append(forsythia_flowering[i])
    x.append(v)

x = np.array(x)
y = np.array(start_date)

# x = x / x.max()
# y = y / y.max()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=1004)
# print('x_test : ', x_test)
# print('y_test : ', y_test)
# print('x_train : ', x_train)
# print('y_train : ', y_train)

train_size = len(y_train)
test_size = len(y_test)
print(train_size, test_size)

# # TODO: 정규화
# mean = x_train.mean(axis=0)
# x_train -= mean
# std = x_train.std(axis=0)
# x_train /= std
#
# x_test -= mean
# x_test /= std

k1 = Knn(x_train, x_test, y_train, y_test)
k = 3
acc = 0
for i in range(test_size):

    print(i, "th data result ", end=' ')

    result = k1.neighbor(k, k1.distance(i))

    print(result, "  label ", end=' ')
    print(y_test[i])
    #if result == y_test[i]:
    #    count = count + 1

    acc += abs(result - y_test[i])
print("\nAccuracy: %.4f" % acc)
time.sleep(10000)

model = Sequential()

# TODO: batch_normalization
model.add(Dense(21, kernel_initializer='he_normal'))
# model.add(layers.Dense(3))
model.add(BatchNormalization())
model.add(Activation('relu'))

# TODO: hidden layer 추가?
#model.add(Dense(units=15,activation='softmax'))

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = optimizers.SGD(lr=0.01,decay=1e-7)
#sgd = optimizers.SGD(lr=0.01)

#  TODO: 일반 사용
#model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

# TODO: sgd 직접 설정
model.compile(optimizer=sgd,loss='mse',metrics=['accuracy'])

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.compile(loss='mean_squared_error', optimizer='sgd')

# TODO: validation split / Test 에서 val 구하면 안돼
model.fit(x_train, y_train, validation_split=0.1, epochs=1000)
# model.fit(x_train, y_train, epochs=1000)
print(model.summary())

test_loss, test_acc = model.evaluate(x_test, y_test)

print('test_loss : ',test_loss)
print('test_acc : ',test_acc)
print()

predictions = model.predict(x_test)

print('y_test = ',y_test[:10])
print('predictions = ', np.argmax(predictions[:10],axis=1))

# for i in range(33):
#     model.fit(x[i*10:i*10+10], y[i], epochs=1000, verbose=1)

# predict = model.predict(x[0][32])
# print("predict",*predict)
# print("real",y[32])

# print(hist.history['loss'])
# print(hist.history['acc'])
# print(hist.history['val_loss'])
