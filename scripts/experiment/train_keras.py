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
import pickle
import glob
import time
import sys 
import os 

from milk.utilities import data_utils
from milk.utilities import training_utils
from milk import Milk, MilkBatch

with open('../dataset/case_dict_obfuscated.pkl', 'rb') as f:  
# with open('../dataset/cases_md5.pkl', 'rb') as f:  
  case_dict = pickle.load(f)

from milk.encoder_config import get_encoder_args

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
  transform_fn = data_utils.make_transform_fn(args.x_size, 
                                              args.y_size, 
                                              args.crop_size, 
                                              args.scale, 
                                              normalize=False)
  train_x, train_y = data_utils.load_list_to_memory(train_list, case_label_fn)
  val_x, val_y = data_utils.load_list_to_memory(val_list, case_label_fn)

  train_generator = data_utils.generate_from_memory(train_x, train_y, 
      batch_size=args.batch_size,
      bag_size=args.bag_size, 
      transform_fn=transform_fn,)
  val_generator = data_utils.generate_from_memory(val_x, val_y, 
      batch_size=args.batch_size,
      bag_size=args.bag_size, 
      transform_fn=transform_fn,)

  print('Testing batch generator')
  ## Some api change between nightly built TF and R1.5
  x, y = next(train_generator)
  print('x: ', x.shape)
  print('y: ', y.shape)
  del x
  del y 

  print('Model initializing')
  encoder_args = get_encoder_args(args.encoder)
  model = MilkBatch(input_shape=(args.crop_size, args.crop_size, 3), 
                    encoder_args=encoder_args, mode=args.mil, use_gate=args.gated_attention,
                    batch_size = args.batch_size, bag_size = args.bag_size, 
                    temperature=args.temperature, freeze_encoder=args.freeze_encoder, 
                    deep_classifier=args.deep_classifier)
  
  if args.tpu:
    # Need to use tensorflow optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  elif args.accumulate > 1:
    optimizer = training_utils.AdamAccumulate(lr=args.learning_rate, accum_iters=args.accumulate)
  else:
    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate, decay=1e-6)

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

  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['categorical_accuracy'])
  model.summary()

  ## Replace randomly initialized weights after model is compiled and on the correct device.
  if args.pretrained_model is not None and os.path.exists(args.pretrained_model):
    print('Replacing random weights with weights from {}'.format(args.pretrained_model))
    model.load_weights(args.pretrained_model, by_name=True)

  if args.early_stop:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                         min_delta = 0.00001, 
                                         patience = 5, 
                                         verbose = 1, 
                                         mode = 'auto',)
    ]
  else:
    callbacks = []

  try:
    # for epc in range(args.epochs):
    #   for k in range(int(args.steps_per_epoch)):
    #     batchx, batchy = next(train_generator)
    #     model.train_on_batch(batchx, batchy)
    #     if k % 10 == 0:
    #       batchpred = model.predict(batchx)
    #       print(batchpred, batchy)

    model.fit_generator(generator=train_generator,
        validation_data=val_generator,
        validation_steps=100,
        steps_per_epoch=args.steps_per_epoch, 
        epochs=args.epochs,
        callbacks=callbacks)
  except KeyboardInterrupt:
    print('Keyboard interrupt caught')
  except Exception as e:
    print('Other error caught')
    print(type(e))
    print(e)
  finally:
    model.save(out_path)
    print('Saved model: {}'.format(out_path))
    print('Training done. Find val and test datasets at')
    print(val_list_file)
    print(test_list_file)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mil',              default = 'attention', type=str)
  parser.add_argument('--scale',            default = 1.0, type=float)
  parser.add_argument('--x_size',           default = 128, type=int)
  parser.add_argument('--y_size',           default = 128, type=int)
  parser.add_argument('--bag_size',         default = 50, type=int)
  parser.add_argument('--crop_size',        default = 96, type=int)
  parser.add_argument('--batch_size',       default = 1,  type=int)
  parser.add_argument('--freeze_encoder',   default = False, action='store_true')
  parser.add_argument('--gated_attention',  default = True, action='store_false')
  parser.add_argument('--temperature',      default = 1, type=float)
  parser.add_argument('--encoder',          default = 'big', type=str)
  parser.add_argument('--deep_classifier',  default = False, action='store_true')

  # Optimizer settings
  parser.add_argument('--learning_rate',    default = 1e-4, type=float)
  parser.add_argument('--accumulate',       default = 1, type=int)
  parser.add_argument('--steps_per_epoch',  default = 500, type=int)
  parser.add_argument('--epochs',           default = 50, type=int)

  # Experiment / data settings
  parser.add_argument('--seed',             default = None, type=int)
  parser.add_argument('--val_pct',          default = 0.2, type=float)
  parser.add_argument('--test_pct',         default = 0.2, type=float)
  parser.add_argument('--data_patt',        default = '../dataset/tiles_reduced', type=str)
  parser.add_argument('--save_prefix',      default = 'save', type=str)
  parser.add_argument('--pretrained_model', default = None, type=str)

  # old
  parser.add_argument('--dont_use_pretrained', default = False, action='store_true')

  parser.add_argument('--verbose',          default = False, action='store_true')
  parser.add_argument('--val_list',         default = None, type=str)
  parser.add_argument('--test_list',        default = None, type=str)
  parser.add_argument('--early_stop',       default = False, action='store_true')
  parser.add_argument('--tpu',              default = False, action='store_true')
  args = parser.parse_args()

  main(args)
