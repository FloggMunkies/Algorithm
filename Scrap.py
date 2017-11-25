import tensorflow as tf

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_node = a + b

print(sess.run(add_node, {a: 2, b: 10}))
print(sess.run(add_node, {a: [5, 4], b: [2, 8]}))
