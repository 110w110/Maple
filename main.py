import time

import keras.initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Activation
from tensorflow.keras import optimizers
from input_data import *
import numpy as np
from sklearn.model_selection import train_test_split

keras.initializers.he_normal(seed=None)

file = load_xls('./data/maple.xlsx')
start_date = get_single_column(file, 'sheet1', 'B2', 'B99')
forsythia_flowering = get_single_column(file, 'sheet6', 'D2', 'D99')
for i in range(len(start_date)):
    # TODO:날짜 파싱
    start_date[i] = (int(start_date[i].strftime("%Y%m%d")[4:6])-10)*31 + int(start_date[i].strftime("%Y%m%d")[6:])
    forsythia_flowering[i] = (int(forsythia_flowering[i].strftime("%Y%m%d")[4:6])-3)*31 + int(forsythia_flowering[i].strftime("%Y%m%d")[6:])

min_temper = get_single_column(file, 'sheet2', 'D14', 'D919')
max_temper = get_single_column(file, 'sheet2', 'E14', 'E919')

rain_weight = get_single_column(file, 'sheet4', 'C2', 'C1597')
sun_summer = get_singleline_data(file, 'sheet7', 'B6', 'CU6')
sun_fall = get_singleline_data(file, 'sheet7', 'B7', 'CU7')

for i in range(393, -1, -1):
    if 0 <= i % 12 < 7 or 9 <= i % 12:
        del (min_temper[i])
        del (max_temper[i])
        del (rain_weight[i])

x = []
for i in range(97):
    v = []
    for j in range(2):
        v.append(min_temper[i*2+j])
    for j in range(2):
        v.append(max_temper[i*2+j])
    for j in range(2):
        v.append(rain_weight[i*2+j])
    v.append(forsythia_flowering[i])

    v.append(sun_summer[i])
    v.append(sun_fall[i])
    x.append(v)

x = np.array(x)
y = np.array(start_date)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=1004)
# print('x_train : ', x_train)
# print('y_train : ', y_train)

model = Sequential()

# TODO: batch_normalization
model.add(Dense(32, kernel_initializer='he_normal'))
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

# TODO: validation split
model.fit(x_train, y_train, validation_split=0.2, epochs=1000)
print(model.summary())

test_loss, test_acc = model.evaluate(x_test, y_test)

print('test_loss : ',test_loss)
print('test_acc : ',test_acc)
print()

predictions = model.predict(x_test)

# print('y_test = ',y_test[:10])
# print('predictions = ', np.argmax(predictions[:10],axis=1))

predictions=np.argmax(predictions[:10],axis=1)

def itd(num):
    month = 10
    if num > 31:
        month += 1
        num -= 31
    return str(month)+"/"+str(num)

print("Data\tPrediction\tReal\tDifference")
for i in range(len(y_test)):
    print(i + 1, "th \t", itd(predictions[i]), "    \t", itd(y_test[i]), sep='', end='')
    print(" \t", abs(predictions[i] - y_test[i]), " day", sep='')
