import tensorflow as tf

output = None

hidden_layer_weights = tf.Variable(tf.random.normal([4,3]))
out_weights = tf.Variable(tf.random.normal([3,2]))

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input TEST
pre = [15.0, 30.0, 50.0, 100.0]
temp = [10.0, 20.0, 30.0, 40.0]
dust = [1.0, 2.0, 20.0, 35.0]

features = tf.Variable([pre, temp, dust])

# TODO: 레이어 추가 필요
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

print(logits)
