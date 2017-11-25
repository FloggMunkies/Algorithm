import tensorflow as tf

optimizer = tf.train.GradientDescentOptimizer(0.01)

sess = tf.Session()

# Variables
W = tf.Variable([-3], dtype=tf.float32)
b = tf.Variable([.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Nodes
linear_model = W * x + b
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

init = tf.global_variables_initializer()
sess.run(init)

train = optimizer.minimize(loss)

for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))