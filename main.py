import time

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from input_data import *
import numpy as np



# x = np.array([0,1,2,3,4])
# y = x * 2 + 1
file = load_xls('./data/maple.xlsx')
start_date = get_single_column(file, '단풍시기', 'B2', 'B35')
for i in range(len(start_date)):
    start_date[i] = float(start_date[i].strftime("%Y%m%d"))
min_temper = get_single_column(file, '기온', 'D14', 'D408')
max_temper = get_single_column(file, '기온', 'E14', 'E408')
# rain_weight = get_single_column(file, '강수량', 'C2', 'C408')
# rain_time = get_single_column(file, '장마', 'C2', 'C35')
for i in range(393, -1, -1):
    if 0 <= i % 12 < 5 or 10 <= i % 12:
        del (min_temper[i])
        del (max_temper[i])
        # del(rain_weight[i])
np.array(min_temper[i]).astype(float)
np.array(max_temper[i]).astype(float)
x = []
for i in range(33):
    v = []
    for j in range(5):
        v.append(min_temper[i*5+j])
    for j in range(5):
        v.append(max_temper[i*5+j])
    x.append(v)

x = np.array(x)
y = np.array(start_date,)

model = models.Sequential()
# model.add(layers.Dense(10, input_shape=(1,)))
# model.add(layers.Dense(1))

# model.add(layers.Dense(1, input_dim=1))

model.add(layers.Dense(units=33, activation='relu'))
model.add(layers.Dense(units=99, activation='relu'))
model.add(layers.Dense(units=33, activation='sigmoid'))


model.compile("SGD", "mse")
model.fit(x,y, epochs=1000)
# for i in range(33):
#     model.fit(x[i*10:i*10+10], y[i], epochs=1000, verbose=1)

# predict = model.predict(x[0][32])
# print("predict",*predict)
# print("real",y[32])


# print(hist.history['loss'])
# print(hist.history['acc'])
# print(hist.history['val_loss'])



# model = models.Sequential()
# model.add(layers.Dense(32, activation='relu', input_shape=(23,)))
# model.add(layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse')
#
# model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)
#
# print(model)

# output = None
#
# hidden_layer_weights = tf.Variable(tf.random.normal([4,3]))
# out_weights = tf.Variable(tf.random.normal([3,2]))
#
# # Weights and biases
# weights = [
#     tf.Variable(hidden_layer_weights),
#     tf.Variable(out_weights)]
# biases = [
#     tf.Variable(tf.zeros(3)),
#     tf.Variable(tf.zeros(2))]
#
# # Input TEST
# pre = [15.0, 30.0, 50.0, 100.0]
# temp = [10.0, 20.0, 30.0, 40.0]
# dust = [1.0, 2.0, 20.0, 35.0]
#
# features = tf.Variable([pre, temp, dust])
#
# # TODO: 레이어 추가 필요
# hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
# hidden_layer = tf.nn.relu(hidden_layer)
# logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
#
# print(logits)



# # tf Graph input
# x = tf.placeholder("float", [None, 28, 28, 1])
# y = tf.placeholder("float", [None, n_classes])
#
# x_flat = tf.reshape(x, [-1, n_input])


# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
#     .minimize(cost)
#
# # Initializing the variables
# init = tf.global_variables_initializer()
#
# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
#     # Training cycle
#     for epoch in range(training_epochs):
#         total_batch = int(mnist.train.num_examples / batch_size)
#         # Loop over all batches
#         for i in range(total_batch):
#             batch_x, batch_y = mnist.train.next_batch(batch_size)
#             # Run optimization op (backprop) and cost op (to get loss value)
#             sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
#
