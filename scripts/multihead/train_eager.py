"""
Script to run the MIL workflow on numpy saved tile caches

V1 of the script adds external tracking of val and test lists
for use with test_*.py
"""
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import numpy as np
import traceback
import argparse
import datetime
import pickle
import glob
import time
import sys 
import os 
import gc

from milk.utilities import data_utils, MILDataset
from milk.eager import MilkEager

import collections
from milk.encoder_config import get_encoder_args



class ShouldStop():
  """ Track the validation loss
  send a stop signal if we haven't improved within a patience window
  """
  def __init__(self, min_calls = 5, patience = 5):
    self.min_calls = min_calls
    self.val_loss = np.inf
    self.n_calls = 0
    self.patience = patience
    self.since_last_improvement = 0

  def should_stop(self, new_loss):
    self.n_calls += 1
    # min_calls trumps the loss; also don't increment the loss counter
    if self.n_calls < self.min_calls:
      return False

    self.since_last_improvement += 1
    print('checking loss = {:3.5f} vs {:3.5f}'.format(new_loss,
      self.val_loss))

    ret = False
    if new_loss < self.val_loss:
      print('improved. resetting counter.')
      self.val_loss = new_loss
      self.since_last_improvement = 0
      ret = False
    elif self.since_last_improvement >= self.patience:
      ret = True
    else:
      ret = False

    print('calls since last improvement = ',
      self.since_last_improvement)

    if self.n_calls < self.min_calls:
      ret = False

    return ret


class GradientAccumulator():
  def __init__(self, variable_list, n = 10, batch_size = 1):
    # variable_list to track the correct order of variables 
    self.variable_list = variable_list
    self.grad_counter = 0
    self.n = n
    self.batch_size = batch_size
    self.grads_and_vars = collections.defaultdict(list)

  def track(self, grads, variables):
    for g, v in zip(grads, variables):
      self.grads_and_vars[v.name].append(g)
    self.grad_counter += 1
    if self.grad_counter == self.n:
      should_update = True
    else:
      should_update = False
    return should_update

  def accumulate(self):
    grads = []
    #for v, g in self.grads_and_vars.items():
    for v in self.variable_list:
      g = self.grads_and_vars[v.name]
      if any(x is None for x in g):
        grads.append(None)
        continue
      if self.n == 1:
        grads.append(g[0])
      else:
        gmean = tf.reduce_mean(g, axis=0, keep_dims=False)
        grads.append(gmean)
    self.reset()
    return grads

  def reset(self):
    self.grads_and_vars = collections.defaultdict(list)
    self.grad_counter = 0

def eval_acc(ytrue, yhat):
  ytrue_max = np.argmax(ytrue, axis=-1)
  yhat_max = np.argmax(yhat, axis=-1)
  acc = (ytrue_max == yhat_max).mean()
  return acc


def create_outputs(args):
  exptime = datetime.datetime.now()
  exptime_str = exptime.strftime('%Y_%m_%d_%H_%M_%S')
  out_path = os.path.join(args.out_base, args.save_prefix, '{}.h5'.format(exptime_str))
  if not os.path.exists(os.path.dirname(out_path)):
    os.makedirs(os.path.dirname(out_path)) 

  ## Write out arguments passed for this session
  arg_file = os.path.join(args.out_base, './args', '{}.txt'.format(exptime_str))
  if not os.path.exists(os.path.dirname(arg_file)):
    os.makedirs(os.path.dirname(arg_file))
  with open(arg_file, 'w+') as f:
    for a in vars(args):
      f.write('{}\t{}\n'.format(a, getattr(args, a)) )

  return out_path, exptime_str



def val_step(model, val_iterator, val_len):
  head = np.random.randint(model.heads)
  print('Validating head {} for {} steps'.format(head, val_len))
  losses = []
  for k, (x, y) in enumerate(val_iterator):
    yhat = model(x, heads = [head])
    loss = tf.keras.losses.categorical_crossentropy(
      y_true=tf.constant(y, tf.float32), y_pred=yhat[0])
    losses.append(np.mean(loss))
    print('\r{}: {}  ({})  '.format(k, np.mean(losses), ((k+1) % val_len)), 
      end='', flush=True)
    if (k+1) % val_len == 0: break

  print('')
  val_loss = np.mean(losses)
  return val_loss


# @tf.contrib.eager.defun
def train_epoch(model, optimizer, train_iterator, train_len, epc, trackers, args):
  trainable_variables = model.trainable_variables
  train_head = [epc % args.heads]
  avglosses, steptimes = [], []
  # print('\nTraining head {}'.format(train_head))
  for k, (x,y) in enumerate(train_iterator):
    # train_head = [k % args.heads]
    tstart = time.time()
    with tf.GradientTape() as tape:
      # print('Processing {} {}\t'.format(x.shape, y), end='')
      # x, y = next(train_iterator)
      yhat = model(tf.constant(x).gpu(), training=True, heads=train_head)
      loss = tf.keras.losses.categorical_crossentropy(
        y_true=tf.constant(y, dtype=tf.float32), y_pred=yhat[0])

    grads = tape.gradient(loss, trainable_variables)
    loss_mn = np.mean(loss)
    acc = eval_acc(y, yhat)
    avglosses.append(loss_mn)
    
    trackers['loss'].append(loss_mn)
    trackers['acc'].append(acc)
    trackers['step'] += 1

    # with tf.device('/cpu:0'):
        # should_update = accumulator.track(grads, trainable_variables)

    tend = time.time()
    steptimes.append(tend - tstart)
    # if should_update:
    #   grads = accumulator.accumulate()
    # with tf.device('/cpu:0'):
    # print('Applying gradients')
    optimizer.apply_gradients(zip(grads, trainable_variables))
    print('\r{:07d}: loss={:3.5f} dt={:3.3f}s   '.format(k, 
      np.mean(avglosses), np.mean(steptimes)), end='', flush=1)
    if (k+1) % train_len == 0: break

  return trackers


# @profile
def main(args):
  """ 
  1. Create generator datasets from the provided lists
  2. train and validate Milk

  v0 - create datasets within this script
  v1 - factor monolithic training_utils.mil_train_loop !!
  tpu - replace data feeder and mil_train_loop with tf.keras.Model.fit()
  July 4 2019 - added MILDataset that takes away all the dataset nonsense
  """
  out_path, exptime_str = create_outputs(args)

  data_factory = MILDataset(args.dataset, crop=args.crop_size, n_classes=2)
  data_factory.split_train_val('case_id', seed=args.seed)

  #with tf.device('/gpu:0'): 
  print('Model initializing')
  encoder_args = get_encoder_args(args.encoder)
  model = MilkEager(encoder_args=encoder_args, 
                    mil_type=args.mil,
                    deep_classifier=args.deep_classifier,
                    batch_size=16,
                    temperature=args.temperature,
                    heads = args.heads)
  print('Running once to load CUDA')
  x = tf.zeros((1, 1, args.crop_size, args.crop_size, args.channels))

  ## The way we give the list is very particular.
  all_heads = [0,1,2,3,4,5,6,7,8,9]
  yhat = model(x, heads=all_heads, training=True, verbose=True)
  print('yhat: {} ({} {})'.format(yhat[0], yhat[0].shape, yhat[0].dtype))
  model.summary()

  ## Replace randomly initialized weights after model is compiled and on the correct device.
  if args.pretrained_model is not None and os.path.exists(args.pretrained_model):
    print('Replacing random weights with weights from {}'.format(args.pretrained_model))
    try:
      model.load_weights(args.pretrained_model, by_name=True)
    except Exception as e:
      print(e)

  ## Controlling overfitting by monitoring a metric, with some patience since the last improvement
  if args.early_stop:
    stopper = ShouldStop(patience = args.heads)
  else:
    stopper = lambda x: False

  # accumulator = GradientAccumulator(n = args.accumulate, variable_list=trainable_variables)

  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  trackers = {stat: [] for stat in ['loss', 'acc']}
  trackers['step'] = 0

  data_factory.tensorflow_iterator(mode='train', ref='train', repeats=args.epochs,
    subset=args.bag_size, seed=None, attr='stage_code', threads=args.threads)
  # data_factory.tensorflow_iterator(mode='val', ref='val', repeats=args.epochs,
  #   subset=args.bag_size, seed=None, attr='stage_code', threads=args.threads)

  # val_iterator, val_len = data_factory.iterator_refs['val']
  train_iterator, train_len = data_factory.iterator_refs['train']

  try:
    for epc in range(args.epochs):

      tf.set_random_seed(1)
      # gc.collect()
      # data_factory.clear_mem()
      # val_iterator = data_factory.python_iterator(mode='val', 
      #   subset=args.bag_size, seed=epc, attr='stage_code',
      #   data_fn=data_fn, label_fn=label_fn)
      # data_factory.close_refs()
      # data_factory = MILDataset(args.dataset, crop=args.crop_size, n_classes=2)
      # data_factory.split_train_val('case_id', seed=args.seed)
      # val_iterator = data_factory.iterator_refs['val']

      # val_loss = val_step(model, val_iterator, val_len)
      # print('\nepc: {} val_loss = {}'.format(epc, val_loss))
      # if args.early_stop and stopper.should_stop(val_loss):
      #   break

      # data_factory.tensorflow_iterator(mode='train', ref='train',
      #   seed=epc, batch_size=args.batch_size, threads=args.threads,
      #   subset=args.bag_size, attr='stage_code', eager=True)
      # train_iterator = data_factory.iterator_refs['train']
      trackers = train_epoch(model, optimizer, train_iterator, train_len, epc, trackers, args)

    # if epc % args.snapshot_epochs == 0:
    #   snapshot_path = out_path.replace('.h5', '-{:03d}.h5'.format(epc))
    #   print('Snapshotting to {}'.format(snapshot_path))
    #   model.save_weights(snapshot_path)

  except KeyboardInterrupt:
    print('Keyboard interrupt caught')

  except Exception as e:
    print('Other error caught')
    print(e)
    traceback.print_tb(e.__traceback__)

  finally:
    model.save_weights(out_path)
    print('Saved model: {}'.format(out_path))

    # Save the loss profile
    training_stats = os.path.join(args.out_base ,'save', '{}_training_curves.txt'.format(exptime_str))
    print('Dumping training stats --> {}'.format(training_stats))
    with open(training_stats, 'w+') as f:
      for l, a, s in zip(trackers['loss'], trackers['acc'], np.arange(trackers['step'])):
        f.write('{:06d}\t{:3.5f}\t{:3.5f}\n'.format(s, l, a))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Model and data
  parser.add_argument('--dataset',          default = None, type=str)

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
  parser.add_argument('--threads',          default = 4, type=int)

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
  parser.add_argument('--out_base',         default = 'trained_model', type=str)
  parser.add_argument('--snapshot_epochs',  default = 5, type=int)

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
