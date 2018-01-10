from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
CLASSES_NUM = 10
CHARS_NUM = 1

RECORD_DIR = "./data"
TRAIN_FILE_ONE = "train_one.tfrecords"
VALID_FILE_ONE = "valid_one.tfrecords"
TRAIN_FILE_TWO = "train_two.tfrecords"
VALID_FILE_TWO = "valid_two.tfrecords"
TRAIN_DIR_ONE = "./data/train_data_one"
TRAIN_DIR_TWO = "./data/train_data_two"
VALID_DIR_ONE = "./data/valid_data_one"
VALID_DIR_TWO = "./data/valid_data_two"

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label_raw': tf.FixedLenFeature([], tf.string),
      })
  image = tf.decode_raw(features['image_raw'], tf.int16)
  image.set_shape([IMAGE_HEIGHT * IMAGE_WIDTH])
  # image = tf.cast(image, tf.float32) * (1.  / 255) - 0.5
  image = tf.cast(image, tf.float32) * (1. / 255)
  reshape_image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
  label = tf.decode_raw(features['label_raw'], tf.uint8)
  label.set_shape([CHARS_NUM * CLASSES_NUM])
  reshape_label = tf.reshape(label, [CHARS_NUM, CLASSES_NUM])
  return tf.cast(reshape_image, tf.float32), tf.cast(reshape_label, tf.float32)


def inputs(train, batch_size, file_source):

  if file_source == "train_one":
    filename = os.path.join(RECORD_DIR, TRAIN_FILE_ONE)
  elif file_source == "train_two":
    filename = os.path.join(RECORD_DIR, TRAIN_FILE_TWO)
  elif file_source == "valid_one":
    filename = os.path.join(RECORD_DIR, VALID_FILE_ONE)
  elif file_source == "valid_two":
    filename = os.path.join(RECORD_DIR, VALID_FILE_TWO)
  else:
    raise("invalid file name for input!")

  with tf.name_scope(file_source):
    filename_queue = tf.train.string_input_producer([filename])
    image, label = read_and_decode(filename_queue)
    if train:
        images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                       batch_size=batch_size,
                                                       num_threads=6,
                                                       capacity=2000 + 3 * batch_size,
                                                       min_after_dequeue=2000)
    else:
        images, sparse_labels = tf.train.batch([image, label],
                                               batch_size=batch_size,
                                               num_threads=6,
                                               capacity=2000 + 3 * batch_size)

    return images, sparse_labels


def _fc_matrix(name, shape):
  """generate the matrix of a fully connected layer."""
  with tf.device("/cpu:0"):
    initializer = tf.truncated_normal_initializer()
    var = tf.get_variable(name, shape, initializer = initializer, dtype = tf.float32)
  return var


def _fc_bias(name, shape):
  """generate the bias vector of a fully connected layer."""
  with tf.device("/cpu:0"):
    initializer = tf.zeros_initializer()
    var = tf.get_variable(name, shape, initializer = initializer, dtype = tf.float32)
  return var

### DNN building function.
def build_net(image_one, image_two):
  
  with tf.variable_scope("one_1") as scope:
    matrix = _fc_matrix("matrix", [256, 784])
    bias = _fc_bias("bias", [256, 1])
    pre_activation = tf.add(tf.matmul(matrix, image_one), bias)
    fc_one_1 = tf.nn.relu(pre_activation, name = scope.name)

  with tf.variable_scope("two_1") as scope:
    matrix = _fc_matrix("matrix", [256, 784])
    bias = _fc_bias("bias", [256, 1])
    pre_activation = tf.add(tf.matmul(matrix, image_two), bias)
    fc_two_1 = tf.nn.relu(pre_activation, name = scope.name)

  with tf.variable_scope("one_2") as scope:
    matrix = _fc_matrix("matrix", [96, 256])
    bias = _fc_bias("bias", [96, 1])
    pre_activation = tf.add(tf.matmul(matrix, fc_one_1), bias)
    fc_one_2 = tf.nn.relu(pre_activation, name = scope.name)

  with tf.variable_scope("two_2") as scope:
    matrix = _fc_matrix("matrix", [96, 256])
    bias = _fc_bias("bias", [96, 1])
    pre_activation = tf.add(tf.matmul(matrix, fc_two_1), bias)
    fc_two_2 = tf.nn.relu(pre_activation, name = scope.name)

  with tf.variable_scope("one_3") as scope:
    matrix = _fc_matrix("matrix", [64, 96])
    bias = _fc_bias("bias", [64, 1])
    pre_activation = tf.add(tf.matmul(matrix, fc_one_2), bias)
    fc_one_3 = tf.nn.relu(pre_activation, name = scope.name)

  with tf.variable_scope("two_3") as scope:
    matrix = _fc_matrix("matrix", [64, 96])
    bias = _fc_bias("bias", [64, 1])
    pre_activation = tf.add(tf.matmul(matrix, fc_two_2), bias)
    fc_two_3 = tf.nn.relu(pre_activation, name = scope.name)

  with tf.variable_scope("one_exclu_4") as scope:
    matrix = _fc_matrix("matrix", [16, 64])
    bias = _fc_bias("bias", [16, 1])
    pre_activation = tf.add(tf.matmul(matrix, fc_one_3), bias)
    fc_one_exclu_4 = tf.nn.relu(pre_activation, name = scope.name)

  with tf.variable_scope("two_exclu_4") as scope:
    matrix = _fc_matrix("matrix", [16, 64])
    bias = _fc_bias("bias", [16, 1])
    pre_activation = tf.add(tf.matmul(matrix, fc_two_3), bias)
    fc_two_exclu_4 = tf.nn.relu(pre_activation, name = scope.name)

  with tf.variable_scope("common_4") as scope:
    matrix = _fc_matrix("matrix", [32, 128])
    bias = _fc_bias("bias", [32, 1])
    concatinated = tf.concat((fc_one_3, fc_two_3), 0, name = "concat")
    pre_activation = tf.add(tf.matmul(matrix, concatinated), bias)
    common_4 = tf.nn.relu(pre_activation, name = scope.name)

  with tf.variable_scope("one_5") as scope:
    matrix = _fc_matrix("matrix", [10, 48])
    bias = _fc_bias("bias", [10, 1])
    concatinated = tf.concat((fc_one_exclu_4, common_4), 0, name = "concat")
    fc_one_5 = tf.add(tf.matmul(matrix, concatinated), bias)

  with tf.variable_scope("two_5") as scope:
    matrix = _fc_matrix("matrix", [10, 48])
    bias = _fc_bias("bias", [10, 1])
    concatinated = tf.concat((fc_two_exclu_4, common_4), 0, name = "concat")
    fc_two_5 = tf.add(tf.matmul(matrix, concatinated), bias)

  return fc_one_5, fc_two_5


### Calculate loss by cross-entropy.
### IMPORTANT NOTE.
### Before using tf.nn.softmax_cross_entropy_with_logits
### change the shape from
### [ img 1   img 2   img 3  ...   img n]
### [   x       x       x             x ]
### [  ...     ...     ...           ...]
### [   x       x       x             x ]
### back to
### [ img 1 : x x x x x x x x ]
### [ img 2 : x x x x x x x x ]
### [ ...                  ...]
### [ img n : x x x x x x x x ]
### to match the dimension.
def loss(logits_one, logits_two, label_one, label_two):
  logits_one_transposed = tf.transpose(logits_one)
  logits_two_transposed = tf.transpose(logits_two)
  label_one_transposed = tf.transpose(label_one)
  label_two_transposed = tf.transpose(label_two)
  with tf.variable_scope("one_cross_entropy") as scope:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = label_one_transposed, 
                    logits = logits_one_transposed, name='corss_entropy_per_example')
    loss_one = tf.reduce_mean(cross_entropy, name='cross_entropy')

  with tf.variable_scope("two_cross_entropy") as scope:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = label_two_transposed, 
                     logits = logits_two_transposed, name='corss_entropy_per_example')
    loss_two = tf.reduce_mean(cross_entropy, name='cross_entropy')

  return loss_one, loss_two


### Attach the optimizers to two nets.
def training(loss_one, loss_two):
  # optimizer_one = tf.train.AdamOptimizer(1e-4)
  # optimizer_two = tf.train.AdamOptimizer(1e-4)
  optimizer_one = tf.train.GradientDescentOptimizer(1e-4)
  optimizer_two = tf.train.GradientDescentOptimizer(1e-4)
  train_op_one = optimizer_one.minimize(loss_one)
  train_op_two = optimizer_two.minimize(loss_two)
  return train_op_one, train_op_two


def evaluation(logits, labels):
  correct_prediction = tf.equal(tf.argmax(logits,2), tf.argmax(labels,2))
  correct_batch = tf.reduce_mean(tf.cast(correct_prediction, tf.int32), 1)
  return tf.reduce_sum(tf.cast(correct_batch, tf.float32))


def output(logits):
  return tf.argmax(logits, 2)

