from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import argparse
import sys

import tensorflow as tf
import model as captcha

FLAGS = None

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

BATCH_SIZE = 128
TRAIN_DIR_ONE = "./captcha_train_one"
CHECKPOINT_ONE = "./captcha_train_one/captcha"
TRAIN_DIR_TWO = "./captcha_train_two"
CHECKPOINT_TWO = "./captcha_train_two/captcha"

def run_train():
  """Train CAPTCHA for a number of steps."""

  with tf.Graph().as_default():

    ### Read images and labels in from tfrecords file.
    image_one, label_one = captcha.inputs(True, FLAGS.batch_size, "train_one")
    image_two, label_two = captcha.inputs(True, FLAGS.batch_size, "train_two")


    ### Change the shape from
    ### [ img 1 : x x x x x x x x ]
    ### [ img 2 : x x x x x x x x ]
    ### [ ...                  ...]
    ### [ img n : x x x x x x x x ]
    ### to
    ### [ img 1   img 2   img 3  ...   img n]
    ### [   x       x       x             x ]
    ### [  ...     ...     ...           ...]
    ### [   x       x       x             x ]
    image_one = tf.reshape(image_one, [-1, IMAGE_HEIGHT * IMAGE_WIDTH])
    image_two = tf.reshape(image_two, [-1, IMAGE_HEIGHT * IMAGE_WIDTH])
    image_one = tf.transpose(image_one)
    image_two = tf.transpose(image_two)
    label_one = tf.reshape(label_one, [-1, 10])
    label_two = tf.reshape(label_two, [-1, 10])
    label_one = tf.transpose(label_one)
    label_two = tf.transpose(label_two)

    ### Build Net
    ### logits_one and logits_two are the outputs before softmax.
    logits_one, logits_two = captcha.build_net(image_one, image_two)

    ### Use softmax and cross-entropy to calculate loss.
    loss_one, loss_two = captcha.loss(logits_one, logits_two, label_one, label_two)

    ### Attach the optimizers to two nets.
    train_op_one, train_op_two = captcha.training(loss_one, loss_two)

    saver = tf.train.Saver(tf.global_variables())

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init_op)

    #eval_one = sess.run([captcha.evaluation(logits_one, label_one)])
    #eval_two = sess.run([captcha.evaluation(logits_two, label_two)])
    #print('>> Step %d run_train: accr_one = %.2f, accr_two = %.2f (%.3f sec)' % (step, eval_one,
    #                                                 eval_two, duration))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()

        ### Run a batch to train.
        _, loss_value_one = sess.run([train_op_one, loss_one])
        _, loss_value_two = sess.run([train_op_two, loss_two])

        duration = time.time() - start_time

        ### Output the status after 100 batchs.
        if step % 100 == 0:
          print('>> Step %d run_train: loss_one = %.2f, loss_two = %.2f (%.3f sec)' % (step, loss_value_one,
                                                     loss_value_two, duration))
          eval_one = sess.run(captcha.correct_rate(logits_one, label_one))
          eval_two = sess.run(captcha.correct_rate(logits_two, label_two))
          print('>> Step %d run_train: accr_one = %.2f, accr_two = %.2f (%.3f sec)' % (step, eval_one,
                                                     eval_two, duration))
        ### Save a checkpoint after 5000 batchs.
        if step % 5000 == 0:
          print('>> %s Saving in %s' % (datetime.now(), CHECKPOINT_ONE))
          print('>> %s Saving in %s' % (datetime.now(), CHECKPOINT_TWO))
          saver.save(sess, CHECKPOINT_ONE, global_step=step)
          saver.save(sess, CHECKPOINT_TWO, global_step=step)

        if step == 60000:
          coord.request_stop()
          coord.join(threads)
          sess.close()
          return 0
        step += 1
    except Exception as e:
      print('>> %s Saving in %s' % (datetime.now(), CHECKPOINT_ONE))
      print('>> %s Saving in %s' % (datetime.now(), CHECKPOINT_TWO))
      saver.save(sess, CHECKPOINT_ONE, global_step=step)
      saver.save(sess, CHECKPOINT_TWO, global_step=step)
      coord.request_stop(e)
    finally:
      coord.request_stop()
    coord.join(threads)
    sess.close()


def main(_):
  if tf.gfile.Exists(TRAIN_DIR_ONE):
    tf.gfile.DeleteRecursively(TRAIN_DIR_ONE)
  tf.gfile.MakeDirs(TRAIN_DIR_ONE)
  if tf.gfile.Exists(TRAIN_DIR_TWO):
    tf.gfile.DeleteRecursively(TRAIN_DIR_TWO)
  tf.gfile.MakeDirs(TRAIN_DIR_TWO)
  run_train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./captcha_train',
      help='Directory where to write event logs.'
  )
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='./captcha_train/captcha',
      help='Directory where to write checkpoint.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
