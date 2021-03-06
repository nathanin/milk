"""
Script to run the MIL workflow on numpy saved tile caches

V1 of the script adds external tracking of val and test lists
for use with test_*.py
"""
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import datetime
import glob
import time
import sys 
import os 
import re

from milk.utilities import data_utils
from milk import Milk

with open('../dataset/cases_md5.pkl', 'rb') as f:
  case_dict = pickle.load(f)

def filter_list_by_label(lst):
  lst_out = []
  for l in lst:
    l_base = os.path.basename(l)
    l_base = os.path.splitext(l_base)[0]
    if CASE_LABEL_DICT[l_base] != 2:
      lst_out.append(l)
  print("Got list length {}; returning list length {}".format(
    len(lst), len(lst_out)
  ))
  return lst_out

CASE_PATT = r'SP_\d+-\d+'
def case_label_fn(data_path):
  case = re.findall(CASE_PATT, data_path)[0]
  y_ = CASE_LABEL_DICT[case]
  # print(data_path, y_)
  return y_

def load_lists(data_patt, val_list_file, test_list_file):
  data_list = glob.glob(data_patt)
  val_list = []
  with open(val_list, 'r') as f:
    for l in f:
      val_list.append(l.replace('\n', ''))
  test_list = []
  with open(test_list, 'r') as f:
    for l in f:
      test_list.append(l.replace('\n', ''))

  train_list = []
  for d in data_list:
    if (d not in val_list) and (d not in val_list):
      train_list.append(d)

  return train_list, val_list, test_list

def main(args):
  """ 
  1. Create generator datasets from the provided lists
  2. train and validate Milk

  v0 - create datasets within this script
  v1 - factor monolithic training_utils.mil_train_loop !!
  tpu - replace data feeder and mil_train_loop with tf.keras.Model.fit()

  tpu setup adapted from the fashion MNIST example
  """
  # Take care of passed in test and val lists for the ensemble experiment
  # we need both test list and val list to be given.
  if (args.test_list is not None) and (args.val_list is not None):
    train_list, val_list, test_list = load_lists(os.path.join(args.data_patt, '*.npy'), 
      args.val_list, args.test_list)

  else:
    train_list, val_list, test_list = data_utils.list_data(os.path.join(args.data_patt, '*.npy'), 
      val_pct=0.1, test_pct=0.3, seed=args.seed)

  if args.verbose:
    print("train_list:")
    print(train_list)
    print("val_list:")
    print(val_list)
    print("test_list:")
    print(test_list)

  ## Filter out unwanted samples:
  train_list = filter_list_by_label(train_list)
  val_list   = filter_list_by_label(val_list)
  test_list  = filter_list_by_label(test_list)

  train_list = data_utils.enforce_minimum_size(train_list, args.bag_size, verbose=True)
  val_list   = data_utils.enforce_minimum_size(val_list,   args.bag_size, verbose=True)
  test_list  = data_utils.enforce_minimum_size(test_list,  args.bag_size, verbose=True)

  transform_fn = data_utils.make_transform_fn(args.x_size, 
                        args.y_size, 
                        args.crop_size, 
                        args.scale)

  train_generator = data_utils.load_generator(train_list, 
    transform_fn=transform_fn, 
    bag_size=args.bag_size, 
    case_label_fn=case_label_fn)
  val_generator = data_utils.load_generator(val_list, 
    transform_fn=transform_fn, 
    bag_size=args.bag_size, 
    case_label_fn=case_label_fn)

  print('Testing batch generator')
  ## Some api change between nightly built TF and R1.5
  x, y = next(train_generator)
  print('x: ', x.shape)
  print('y: ', y.shape)

  encoder_args = {
    'depth_of_model': 32,
    'growth_rate': 64,
    'num_of_blocks': 4,
    'output_classes': 2,
    'num_layers_in_each_block': 8,
  }

  print('Model initializing')
  model = Milk(input_shape=(args.bag_size, args.crop_size, args.crop_size, 3), 
               encoder_args=encoder_args, mode=args.mil, use_gate=args.gated_attention,
               freeze_encoder=args.freeze_encoder)
  
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

  exptime = datetime.datetime.now()
  exptime_str = exptime.strftime('%Y_%m_%d_%H_%M_%S')
  out_path = os.path.join(args.save_prefix, '{}.h5'.format(exptime_str))
  if not os.path.exists(os.path.dirname(out_path)):
    os.makedirs(os.path.dirname(out_path)) # < recursively create the destination directory

  ## Write out cases for testing
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

  ## Transfer to TPU 
  if args.tpu:
    print('Setting up model on TPU')
    if 'COLAB_TPU_ADDR' not in os.environ:
      print('ERROR: Not connected to a TPU runtime!')
    else:
      tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
      print ('TPU address is', tpu_address)
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy)

  ## Compile the model
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['categorical_accuracy'])

  ## Replace randomly initialized weights after model is compiled and on the correct device.
  if os.path.exists(args.pretrained_model):
    model.load_weights(args.pretrained_model, by_name=True
    
  ## Fit
  try:
    model.fit_generator(generator=train_generator,
      validation_data=val_generator,
      validation_steps=25,
      steps_per_epoch=1000, 
      epochs=25,
      use_multiprocessing=True,
      workers=-1)
  except KeyboardInterrupt:
    print('Keyboard interrupt caught')
  except Exception as e:
    print('Other error caught')
    print(type(e))
    print(e)
  finally:
    ## Save
    model.save(out_path)
    print('Saved model: {}'.format(out_path))
    print('Training done. Find val and test datasets at')
    print(val_list_file)
    print(test_list_file)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mil',        default='attention', type=str)
  parser.add_argument('--scale',      default=1.0, type=float)
  parser.add_argument('--epochs',       default=50, type=int)
  parser.add_argument('--x_size',       default=128, type=int)
  parser.add_argument('--y_size',       default=128, type=int)
  parser.add_argument('--bag_size',     default=150, type=int)
  parser.add_argument('--crop_size',    default=96, type=int)
  parser.add_argument('--batch_size',     default=1, type=int)
  parser.add_argument('--learning_rate',  default=1e-5, type=float)
  parser.add_argument('--freeze_encoder',   default=False, action='store_true')
  parser.add_argument('--gated_attention',  default=False, action='store_true')

  parser.add_argument('--tpu',        default=False, action='store_true')
  parser.add_argument('--seed',       default=None, type=int)
  parser.add_argument('--val_pct',      default=0.3, type=float)
  parser.add_argument('--test_pct',     default=0.2, type=float)
  parser.add_argument('--data_patt',    default='../dataset/tiles', type=str)
  parser.add_argument('--save_prefix',    default='save', type=str)
  parser.add_argument('--pretrained_model', default='../pretraining/pretrained.h5', type=str)

  parser.add_argument('--verbose',      default=False, action='store_true')
  parser.add_argument('--val_list',     default=None, type=str)
  parser.add_argument('--test_list',    default=None, type=str)
  args = parser.parse_args()

  main(args)
