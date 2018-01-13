from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import argparse
import sys
import math

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

def run_eval():
  with tf.Graph().as_default(), tf.device('/cpu:0'):

    ### Read images and labels in from tfrecords file.
    image_one, label_one = captcha.inputs(False, FLAGS.batch_size, "valid_one")
    image_two, label_two = captcha.inputs(False, FLAGS.batch_size, "valid_two")

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

    ### Evaluate
    eval_correct_one = captcha.correct_num(logits_one, label_one)
    eval_correct_two = captcha.correct_num(logits_two, label_two)


    sess = tf.Session()    
    saver = tf.train.Saver()    
    saver.restore(sess, tf.train.latest_checkpoint(TRAIN_DIR_ONE))
    saver.restore(sess, tf.train.latest_checkpoint(TRAIN_DIR_TWO))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count_one = 0
      true_count_two = 0
      total_true_count_one = 0
      total_true_count_two = 0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      print('>> loop: %d, total_sample_count: %d' % (num_iter, total_sample_count))
      while step < num_iter and not coord.should_stop():
        true_count_one = sess.run(eval_correct_one)
        true_count_two = sess.run(eval_correct_two)
        total_true_count_one += true_count_one
        total_true_count_two += true_count_two
        precision_one = true_count_one / FLAGS.batch_size
        precision_two = true_count_two / FLAGS.batch_size
        print('>> %s Step %d Net 1: true/total: %d/%d precision @ 1 = %.3f'
                    %(datetime.now(), step, true_count_one, FLAGS.batch_size, precision_one))
        print('>> %s Step %d Net 2: true/total: %d/%d precision @ 1 = %.3f'
                    %(datetime.now(), step, true_count_two, FLAGS.batch_size, precision_two))
        step += 1
      precision_one = total_true_count_one / total_sample_count
      precision_two = total_true_count_two / total_sample_count
      print('>> %s Net 1 true/total: %d/%d precision @ 1 = %.3f'
                    %(datetime.now(), total_true_count_one, total_sample_count, precision_one)) 
      print('>> %s Net 2 true/total: %d/%d precision @ 1 = %.3f'
                    %(datetime.now(), total_true_count_two, total_sample_count, precision_two))       
    except Exception as e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
    coord.join(threads)
    sess.close()


def main(_):
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  run_eval()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_examples',
      type=int,
      default=20000,
      help='Number of examples to run validation.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.'
  )
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./captcha_train',
      help='Directory where to restore checkpoint.'
  )
  parser.add_argument(
      '--eval_dir',
      type=str,
      default='./captcha_eval',
      help='Directory where to write event logs.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
