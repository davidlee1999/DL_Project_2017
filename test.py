import tensorflow as tf
a = tf.Variable(3, dtype = tf.int32)
b = tf.Variable(5, dtype = tf.int32)
c = tf.Variable(0, dtype = tf.int32)
c = tf.add(a,3)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(c))
