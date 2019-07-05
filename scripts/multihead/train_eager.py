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

from milk.utilities import data_utils, MILDataset
from milk.eager import MilkEager

import collections
from milk.encoder_config import get_encoder_args


def val_step(model, val_generator):
  losses = np.zeros(steps)
  for k , (x, y) in enumerate(val_generator):
    yhat = model(x)
    loss = tf.keras.losses.categorical_crossentropy(
      y_true=tf.constant(y, tf.float32), y_pred=yhat)
    losses[k] = np.mean(loss)

  val_loss = np.mean(losses)
  return val_loss

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



def main(args):
  """ 
  1. Create generator datasets from the provided lists
  2. train and validate Milk

  v0 - create datasets within this script
  v1 - factor monolithic training_utils.mil_train_loop !!
  tpu - replace data feeder and mil_train_loop with tf.keras.Model.fit()
  July 4 2019 - added MILDataset that takes away all the dataset nonsense
  """

  data_factory = MILDataset(args.dataset, crop=args.crop_size, n_classes=2)

  #with tf.device('/gpu:0'): 
  print('Model initializing')
  encoder_args = get_encoder_args(args.encoder)
  model = MilkEager(encoder_args=encoder_args, 
                    mil_type=args.mil,
                    deep_classifier=args.deep_classifier,
                    batch_size=64,
                    temperature=args.temperature,
                    heads = args.heads)
  print('Running once to load CUDA')
  x = tf.zeros((1, 1, args.crop_size, args.crop_size, args.channels))
  yhat = model(x.gpu(), head='all', training=True, verbose=True)
  print('Running again for time')
  tstart = time.time()
  yhat = model(x.gpu(), head=0, training=True, verbose=True)
  tend = time.time()
  print('yhat:', yhat.shape, 'tdelta = {:3.4f}'.format(tend-tstart))
  model.summary()

  exptime = datetime.datetime.now()
  exptime_str = exptime.strftime('%Y_%m_%d_%H_%M_%S')
  out_path = os.path.join(args.save_prefix, '{}.h5'.format(exptime_str))
  if not os.path.exists(os.path.dirname(out_path)):
    os.makedirs(os.path.dirname(out_path)) 

  ## Write out arguments passed for this session
  arg_file = os.path.join('./args', '{}.txt'.format(exptime_str))
  with open(arg_file, 'w+') as f:
    for a in vars(args):
      f.write('{}\t{}\n'.format(a, getattr(args, a)))

  ## Replace randomly initialized weights after model is compiled and on the correct device.
  if args.pretrained_model is not None and os.path.exists(args.pretrained_model):
    print('Replacing random weights with weights from {}'.format(args.pretrained_model))
    try:
      model.load_weights(args.pretrained_model, by_name=True)
    except Exception as e:
      print(e)

  ## Controlling overfitting by monitoring a metric, with some patience since the last improvement
  if args.early_stop:
    stopper = ShouldStop(patience = 5)
  else:
    stopper = lambda x: False

  trainable_variables = model.trainable_variables
  accumulator = GradientAccumulator(n = args.accumulate, variable_list=trainable_variables)

  losstracker, acctracker, steptracker, avglosses, steptimes = [], [], [], [], []
  totalsteps = 0
  try:
    for epc in range(args.epochs):
      optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate/(epc+1))
      training_iterator = data_factory.tensorflow_iterator(mode='train', seed=args.seed, 
        subset=args.bag_size, attr='stage_code', eager=True)

      for k, (x, y) in enumerate(training_iterator):
        totalsteps += 1
        tstart = time.time()
        with tf.GradientTape() as tape:
          # x, y = next(train_gen)
          yhat = model(x.gpu(), training=True)
          loss = tf.keras.losses.categorical_crossentropy(
            y_true=tf.constant(y, dtype=tf.float32), y_pred=yhat)

        loss_mn = np.mean(loss)
        acc = eval_acc(y, yhat)
        avglosses.append(loss_mn)
        losstracker.append(loss_mn)
        acctracker.append(acc)
        steptracker.append(totalsteps)

        grads = tape.gradient(loss, trainable_variables)
        should_update = accumulator.track(grads, trainable_variables)

        tend = time.time()
        steptimes.append(tend - tstart)
        if should_update:
          grads = accumulator.accumulate()
          #for g, v in zip(grads, trainable_variables):
          #  print(v.name, v.shape, g.shape)
          optimizer.apply_gradients(zip(grads, trainable_variables))

        if k % 20 == 0:
          print('{:06d}: loss={:3.5f} dt={:3.3f}s'.format(k, 
            np.mean(avglosses), np.mean(steptimes)))
          avglosses, steptimes = [], []
          for y_, yh_ in zip(y, yhat):
            print('\t{} {}'.format(y_, yh_))

      val_iterator = data_factory.tensorflow_iterator(mode='val', seed=args.seed, 
        subset=args.bag_size, attr='stage_code', eager=True)
      val_loss = val_step(model, val_gen)
      print('epc: {} val_loss = {}'.format(epc, val_loss))

      ## Clean up hanging resources from the iterators
      del training_iterator
      del val_iterator

      if args.early_stop and stopper.should_stop(val_loss):
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

    # Save the loss profile
    training_stats = os.path.join('save', '{}_training_curves.txt'.format(exptime_str))
    with open(training_stats, 'w+') as f:
      for l, a, s in zip(losstracker, acctracker, steptracker):
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
  parser.add_argument('--cases_subset',     default = 64, type=int)
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
  config.gpu_options.allow_growth = True
  tf.enable_eager_execution(config=config)


  def case_label_fn(data_path):
    case = os.path.splitext(os.path.basename(data_path))[0]
    y_ = case_dict[case]
    return y_

  main(args)
