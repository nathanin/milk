from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import numpy as np
import pandas as pd
import traceback
import argparse
import datetime
import pickle
import glob
import time
import sys 
import os 

from milk.utilities import data_utils, MILDataset
from milk.eager import MilkEager

import collections
from milk.encoder_config import get_encoder_args


def calc_acc(ytrues, yhats):
  ytrues = np.array(ytrues)
  yhats = np.stack(yhats, axis=0)

  yhats_1 = yhats > 0.5
  yhats_mean = np.mean(yhats, axis=-1) > 0.5
  mean_accuracy = (yhats_mean == ytrues).mean()

  head_accuracy = []
  for h in range(yhats.shape[-1]):
    head_accuracy.append((yhats_1[:,h] == ytrues).mean())

  return mean_accuracy, head_accuracy


def print_accuracy(ytrues, yhats):
  print('\n\n')
  mean_accuracy, head_accuracy = calc_acc(ytrues, yhats)

  for i, h in enumerate(head_accuracy):
    print('Head {}: accuracy = {}'.format(i, h))

  print('Mean of all heads: accuracy: {}'.format(mean_accuracy))


def write_out(ytrues, yhats, outf):
  ytrues = np.array(ytrues)
  yhats = np.stack(yhats, axis=0)

  print('{} --> {}'.format(len(ytrues), outf))
  with open(outf, 'w+') as f:
    f.write( 'ytrue' )
    for h in range(args.heads):
      f.write(',h_{}'.format(h))
    f.write('\n')

    for s in range(len(yhats)):
      f.write('{}'.format(ytrues[s]))
      for h in range(args.heads):
        f.write(',{}'.format(yhats[s,h]))
      f.write('\n')


def main(args):

  #with tf.device('/gpu:0'): 
  print('Model initializing')
  encoder_args = get_encoder_args(args.encoder)
  model = MilkEager(encoder_args=encoder_args, 
                    mil_type=args.mil,
                    deep_classifier=args.deep_classifier,
                    batch_size=16,
                    temperature=args.temperature,
                    heads=args.heads)
  print('Running once to load CUDA')
  x = tf.zeros((1, 1, args.crop_size, args.crop_size, args.channels))

  ## This is really weird. eager mode complains when this is a range()
  ## It evern complains when it's a list(range())
  ## If the tf.contrib.eager.defun decorator is removed, it's OK
  ## So it's an autograph problem 
  all_heads = [0,1,2,3,4,5,6,7,8,9]
  yhat = model(x.gpu(), heads=all_heads, training=True, verbose=True)
  model.summary()

  model.load_weights(args.pretrained_model, by_name=True)

  ## Set up the data stuff
  data_factory = MILDataset(args.dataset, crop=args.crop_size, n_classes=2)
  data_factory.split_train_val('case_id', seed=args.seed)
 
  test_iterator = data_factory.tensorflow_iterator(mode='test', seed=args.seed, 
                                                   batch_size=1,
                                                   buffer_size=1,
                                                   threads=1,
                                                   subset=args.bag_size, 
                                                   attr='stage_code', 
                                                   eager=True)

  ## Track yhats
  print('-------------------------------------------------------\n\n')
  ytrues , yhats = [], []
  all_heads = list(range(args.heads))
  for k, (x, y) in enumerate(test_iterator):

    # print('{:03d}: ytrue = {}'.format(k, y[0,1]))
    ytrues.append(y[0,1])

    yhat = model(x.gpu(), training=False, heads=all_heads)
    yhat = np.array([yh[0,1] for yh in yhat])
    yhats.append(yhat)
    # print('     yhat = {}'.format(yhat))

    # Take a running mean and show it
    acc, _ = calc_acc(ytrues, yhats)
    print('\r{:03d} Accuracy: {:3.3f} %'.format(k, acc), end='', flush = True)
    # sys.stdout.flush()

  
  print('\n\n-------------------------------------------------------')
  print_accuracy(ytrues, yhats)
  write_out(ytrues, yhats, args.out)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Model and data
  parser.add_argument('--dataset',          default = None, type=str)
  parser.add_argument('--out',              default = 'test.csv', type=str)

  parser.add_argument('--mil',              default = 'attention', type=str)
  parser.add_argument('--scale',            default = 1.0, type=float)
  parser.add_argument('--heads',            default = 5, type=int)
  parser.add_argument('--x_size',           default = 128, type=int)
  parser.add_argument('--y_size',           default = 128, type=int)
  parser.add_argument('--bag_size',         default = 500, type=int)
  parser.add_argument('--crop_size',        default = 128, type=int)
  parser.add_argument('--channels',         default = 3, type=int)
  parser.add_argument('--batch_size',       default = 1, type=int)
  parser.add_argument('--freeze_encoder',   default = False, action='store_true')
  parser.add_argument('--gated_attention',  default = True, action='store_false')
  parser.add_argument('--deep_classifier',  default = False, action='store_true')
  parser.add_argument('--temperature',      default = 1, type=float)
  parser.add_argument('--encoder',          default = 'tiny', type=str)

  # Optimizer settings
  parser.add_argument('--learning_rate',    default = 1e-4, type=float)
  parser.add_argument('--accumulate',       default = 1, type=float)
  parser.add_argument('--epochs',           default = 50, type=int)

  # Experiment / data settings
  parser.add_argument('--seed',             default = 999, type=int)
  parser.add_argument('--val_pct',          default = 0.2, type=float)
  parser.add_argument('--test_pct',         default = 0.2, type=float)
  parser.add_argument('--save_prefix',      default = 'save', type=str)
  parser.add_argument('--pretrained_model', default = None, type=str)

  # old - displaced by letting pretrained_model be None
  parser.add_argument('--dont_use_pretrained', default = False, action='store_true')

  parser.add_argument('--verbose',          default = False, action='store_true')
  parser.add_argument('--val_list',         default = None, type=str)
  parser.add_argument('--test_list',        default = None, type=str)
  parser.add_argument('--early_stop',       default = False, action='store_true')
  args = parser.parse_args()

  config = tf.ConfigProto(log_device_placement=False)
  config.gpu_options.allow_growth = False
  tf.enable_eager_execution(config=config)

  main(args)
