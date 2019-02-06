"""
Script to run the MIL workflow on numpy saved tile caches

V1 of the script adds external tracking of val and test lists
for use with test_*.py
"""
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import numpy as np
import argparse
import datetime
import pickle
import glob
import time
import sys 
import os 

from milk.utilities import data_utils
from milk.eager import MilkEager

with open('../dataset/case_dict_obfuscated.pkl', 'rb') as f:  
# with open('../dataset/cases_md5.pkl', 'rb') as f:  
  case_dict = pickle.load(f)

from milk.encoder_config import deep_args

def filter_list_by_label(lst):
  lst_out = []
  for l in lst:
    l_base = os.path.basename(l)
    l_base = os.path.splitext(l_base)[0]
    if case_dict[l_base] != 2:
      lst_out.append(l)
  print("Got list length {}; returning list length {}".format(
    len(lst), len(lst_out)
  ))
  return lst_out

def case_label_fn(data_path):
  case = os.path.splitext(os.path.basename(data_path))[0]
  y_ = case_dict[case]
  # print(data_path, y_)
  return y_

def load_lists(data_patt, val_list_file, test_list_file):
  data_list = glob.glob(data_patt)
  val_list = []
  with open(val_list_file, 'r') as f:
    for l in f:
      val_list.append(l.replace('\n', ''))
  test_list = []
  with open(test_list_file, 'r') as f:
    for l in f:
      test_list.append(l.replace('\n', ''))

  train_list = []
  for d in data_list:
    if (d not in val_list) and (d not in val_list):
      train_list.append(d)

  return train_list, val_list, test_list

def val_step(model, val_generator, batch_size=8, steps=50):
  losses = np.zeros(steps)
  for k in range(steps):
    x, y = next(val_generator)
    yhat = model(x, batch_size=batch_size, training=False)
    loss = tf.keras.losses.categorical_crossentropy(y_true=tf.constant(y, tf.float32), y_pred=yhat)
    losses[k] = np.mean(loss)

  val_loss = np.mean(losses)
  return val_loss

class ShouldStop():
  """ Track the validation loss

  send a stop signal if we haven't improved within a patience window
  """
  def __init__(self, patience = 5):
    self.val_loss = np.inf
    self.n_calls = 0
    self.patience = patience
    self.since_last_improvement = 0

  def should_stop(self, new_loss):
    self.n_calls += 1
    self.since_last_improvement += 1

    if new_loss < self.val_loss:
      self.val_loss = new_loss
      self.since_last_improvement = 0
      return False
    elif self.since_last_improvement >= self.patience:
      return True
    else:
      return False

def main(args):
  """ 
  1. Create generator datasets from the provided lists
  2. train and validate Milk

  v0 - create datasets within this script
  v1 - factor monolithic training_utils.mil_train_loop !!
  tpu - replace data feeder and mil_train_loop with tf.keras.Model.fit()
  """
  # Take care of passed in test and val lists for the ensemble experiment
  # we need both test list and val list to be given.
  if (args.test_list is not None) and (args.val_list is not None):
    train_list, val_list, test_list = load_lists(
      os.path.join(args.data_patt, '*.npy'), 
      args.val_list, args.test_list)
  else:
    train_list, val_list, test_list = data_utils.list_data(
      os.path.join(args.data_patt, '*.npy'), 
      val_pct=args.val_pct, 
      test_pct=args.test_pct, 
      seed=args.seed)
  
  if args.verbose:
    print("train_list:")
    print(train_list)
    print("val_list:")
    print(val_list)
    print("test_list:")
    print(test_list)

  ## Filter out unwanted samples:
  train_list = filter_list_by_label(train_list)
  val_list = filter_list_by_label(val_list)
  test_list = filter_list_by_label(test_list)

  train_list = data_utils.enforce_minimum_size(train_list, args.bag_size, verbose=True)
  val_list = data_utils.enforce_minimum_size(val_list, args.bag_size, verbose=True)
  test_list = data_utils.enforce_minimum_size(test_list, args.bag_size, verbose=True)
  transform_fn_internal = data_utils.make_transform_fn(args.x_size, args.y_size, 
                                              args.crop_size, args.scale,
                                              eager=True)
  train_x, train_y = data_utils.load_list_to_memory(train_list, case_label_fn)
  val_x, val_y = data_utils.load_list_to_memory(val_list, case_label_fn)

  # Wrap transform fn in an apply_:
  def get_transform_fn(fn):
    def apply_fn(x_bag):
      x_bag = [fn(x) for x in x_bag]
      return np.stack(x_bag, 0)
    return apply_fn

  transform_fn = get_transform_fn(transform_fn_internal)

  def train_generator():
    return data_utils.generate_from_memory(train_x, train_y, batch_size=1,
      bag_size=args.bag_size, #transform_fn=transform_fn,
      pad_first_dim=False)
  def val_generator():
    return data_utils.generate_from_memory(val_x, val_y, batch_size=1,
      bag_size=args.bag_size, #transform_fn=transform_fn, 
      pad_first_dim=False)

  train_dataset = data_utils.tf_dataset(train_generator, batch_size=args.batch_size, 
    preprocess_fn=transform_fn, buffer_size=100, iterator=True)
  val_dataset = data_utils.tf_dataset(val_generator, batch_size=args.batch_size, 
    preprocess_fn=transform_fn, buffer_size=100, iterator=True)

  print('Testing batch generator')
  ## Some api change between nightly built TF and R1.5
  x, y = next(train_dataset)
  print('x: ', x.shape)
  print('y: ', y.shape)

  #with tf.device('/gpu:0'): 
  print('Model initializing')
  model = MilkEager(encoder_args=deep_args, mil_type=args.mil,
                    deep_classifier=args.deep_classifier)
  #model.build_encode_fn(training=True, verbose=False, batch_size=64)
  yhat = model(tf.constant(x), batch_size=16, training=True, verbose=True)
  print('yhat:', yhat.shape)

  exptime = datetime.datetime.now()
  exptime_str = exptime.strftime('%Y_%m_%d_%H_%M_%S')
  out_path = os.path.join(args.save_prefix, '{}.h5'.format(exptime_str))
  if not os.path.exists(os.path.dirname(out_path)):
    os.makedirs(os.path.dirname(out_path)) 

  # Todo : clean up
  val_list_file = os.path.join('./val_lists', '{}.txt'.format(exptime_str))
  with open(val_list_file, 'w+') as f:
    for v in val_list:
      f.write('{}\n'.format(v))

  test_list_file = os.path.join('./test_lists', '{}.txt'.format(exptime_str))
  with open(test_list_file, 'w+') as f:
    for v in test_list:
      f.write('{}\n'.format(v))

  ## Write out arguments passed for this session
  arg_file = os.path.join('./args', '{}.txt'.format(exptime_str))
  with open(arg_file, 'w+') as f:
    for a in vars(args):
      f.write('{}\t{}\n'.format(a, getattr(args, a)))

  # API BUG Keras optimizer has no attribute `apply_gradients`
  # optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate, decay=1e-6)
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  model.summary()

  ## Replace randomly initialized weights after model is compiled and on the correct device.
  if args.pretrained_model is not None and os.path.exists(args.pretrained_model) and not args.dont_use_pretrained:
    print('Replacing random weights with weights from {}'.format(args.pretrained_model))
    try:
      model.load_weights(args.pretrained_model, by_name=True)
    except Exception as e:
      print(e)

  if args.early_stop:
    stopper = ShouldStop(patience = 10)
  else:
    stopper = lambda x: False

  try:
    for epc in range(args.epochs):
      for k in range(args.steps_per_epoch):
        with tf.GradientTape() as tape:
          x, y = next(train_dataset)
          yhat = model(tf.constant(x), batch_size=32, training=True)
          loss = tf.keras.losses.categorical_crossentropy(y_true=tf.constant(y, dtype=tf.float32), y_pred=yhat)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        if k % 50 == 0:
          print('{:06d}: loss={:3.5f}'.format(k, np.mean(loss)))
          for y_, yh_ in zip(y, yhat):
            print('\t{} {}'.format(y_, yh_))

      val_loss = val_step(model, val_dataset, batch_size=32, steps=50)
      print('val_loss = {}'.format(val_loss))
      if stopper.should_stop(val_loss):
        break

  except KeyboardInterrupt:
    print('Keyboard interrupt caught')

  except Exception as e:
    print('Other error caught')
    print(type(e))
    print(e)

  finally:
    model.save_weights(out_path)
    print('Saved model: {}'.format(out_path))
    print('Training done. Find val and test datasets at')
    print(val_list_file)
    print(test_list_file)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Model settings
  parser.add_argument('--mil',              default = 'attention', type=str)
  parser.add_argument('--scale',            default = 1.0, type=float)
  parser.add_argument('--x_size',           default = 128, type=int)
  parser.add_argument('--y_size',           default = 128, type=int)
  parser.add_argument('--bag_size',         default = 50, type=int)
  parser.add_argument('--crop_size',        default = 96, type=int)
  parser.add_argument('--batch_size',       default = 1, type=int)
  parser.add_argument('--freeze_encoder',   default = False, action='store_true')
  parser.add_argument('--gated_attention',  default = True, action='store_false')
  parser.add_argument('--deep_classifier',  default = False, action='store_true')

  # Optimizer settings
  parser.add_argument('--learning_rate',    default = 1e-4, type=float)
  parser.add_argument('--steps_per_epoch',  default = 5000, type=int)
  parser.add_argument('--epochs',           default = 20, type=int)

  # Experiment / data settings
  parser.add_argument('--seed',             default = None, type=int)
  parser.add_argument('--val_pct',          default = 0.2, type=float)
  parser.add_argument('--test_pct',         default = 0.2, type=float)
  parser.add_argument('--data_patt',        default = '../dataset/tiles_reduced', type=str)
  parser.add_argument('--save_prefix',      default = 'save', type=str)
  parser.add_argument('--pretrained_model', default = None, type=str)
  parser.add_argument('--dont_use_pretrained', default = False, action='store_true')

  parser.add_argument('--verbose',          default = False, action='store_true')
  parser.add_argument('--val_list',         default = None, type=str)
  parser.add_argument('--test_list',        default = None, type=str)
  parser.add_argument('--early_stop',       default = False, action='store_true')
  args = parser.parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  tf.enable_eager_execution(config=config)
  main(args)
