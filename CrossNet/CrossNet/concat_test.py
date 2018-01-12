import tensorflow as tf
import numpy as np

vec_one = tf.placeholder(dtype = tf.float32, shape = [3, 1])
vec_two = tf.placeholder(dtype = tf.float32, shape = [3, 1])
const_one = np.matrix([1., 1., 1.])
const_one = const_one.reshape([3, 1])

concated = tf.concat((vec_one, vec_two), axis = 0)

with tf.Session() as sess:
    print(sess.run(concated, feed_dict = {vec_one:const_one, vec_two:const_one}))
