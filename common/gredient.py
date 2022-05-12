import numpy as np
np.random.seed(0)
import tensorflow as tf




# 버전 안맞음
#
# with tf.device('/gpu:0'):
#     x = tf.placeholder(tf.float32)
#     y = tf.placeholder(tf.float32)
#     z = tf.placeholder(tf.float32)
#
#     a = x * y
#     b = a + z
#     c = tf.reduce_sum(b)
# grad_x, grad_y, grad_z = tf.gradients(c, [x, y, z])
# print(grad_x, grad_y,  grad_z)
#
# # 실제 데이터 넣고 실행
# # 모든 데이터 돌릴거니 randn 필요한가?
# with tf.Session() as sess:
#     values = {
#         x: np.random.randn(N, D),
#         y: np.random.randn(N, D),
#         z: np.random.randn(N, D),
#     }
#     out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values)
#     c_val, grad_x_val, grad_y_val, grad_z_val = out
#     print(out)


# 텐서플로우 쓰면 필요없나?
# # 일단 투 레이어 먼저 테스트
# class TwoLayerNet:
#     def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
#         # 가중치 초기화
#         self.params = dict()
#
#         self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#
#         self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)
#
#
#     # x == 입력 데이터, t : 정답 레이블
#     def numerical_gradient(self, x, t):
#         h = 1e-4
#         grad = np.zeros_like(x)
#
#         for idx in range(x.size):
#             tmp_val = x[idx]
#             # f(x+h)
#             x[idx] = tmp_val + h
#             fxh1 = f(x)
#
#             # f(x-h)
#             x[idx] = tmp_val - h
#             fxh2 = f(x)
#
#             grad[idx] = (fxh1 - fxh2) / (2*h)
#             x[idx] = tmp_val # 값 복원
#
#     def predict(self, x):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']
#
#         a1 = np.dot(x, w1) + b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1, w2) + b2
#         y = softmax(a2)
#
#     def loss(self, x, t):
#         return cross_entropy_error(self.predict(x), t)
#
#     def accuracy(self, x, t):
#         y = self.predict(x)
#
#         # axis == 1 하면 안되는것 같기도
#         y = np.argmax(y, axis=1)
#         t = np.argmax(t, axis=1)
#
#         accuracy = np.sum(y == t) / float(x.shape[0])
#         return accuracy
#
#     # 이거 대신 gradient 이용 ?
#     def numerical_gradient(self, x, t):
#         # lambda w ?
#         loss_w = lambda w: self.loss(x,t)
#
#         grads = dict()
#         grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
#
#         return grads
#
#     # def learn (batch) 는 안써도 될듯 input 적어서