"""
Minimal example with bagged-MNIST

## bagged-MNIST
To fit the Multiple Instance Learning problem statement, we consider bags of MNIST images.
First, pick a digit, or rule, to consider "positive".
Choose "9".
Draw a bag of size N. If B contains at least one digit "9", label B positive.

Then, learn a MIL model using this target.

To load up pretrained weights copy them over by name:
https://github.com/keras-team/keras/issues/1873

"""
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as kb
import numpy as np

import argparse
import os 

from milk import Milk, MilkBatch
from milk.encoder_config import get_encoder_args
from milk.utilities import training_utils

def rearrange_bagged_mnist(x, y, positive_label):
  """
  For simplicity, rearrange the mnist digits from
  x ~ all classes 
  
  into

  x_pos ~ positive class(es)
  x_neg ~ positive class(es)
  """
  positive_mask = y == positive_label
  negative_mask = y != positive_label
  x_pos = x[positive_mask, ...] / 255.
  x_neg = x[negative_mask, ...] / 255.

  return x_pos, x_neg

def generate_negative_bag(x_neg, N):
  n_x_neg = x_neg.shape[0]
  neg_indices = np.random.choice(range(n_x_neg), N, replace=False)
  xbag = x_neg[neg_indices,...]
  np.random.shuffle(xbag)

  return xbag

def generate_positive_bag(x_pos, x_neg, N):
  n_x_pos = x_pos.shape[0]
  min_positive = int(N * 0.1)
  max_positive = int(N * 0.25)
  # n_pos = int(np.random.uniform(low=1, high=int(N * 0.2)))
  n_pos = int(np.random.uniform(low=min_positive, 
                  high=max_positive))
  # print('Generating bag with {} positive instances'.format(n_pos))
  pos_indices = np.random.choice(range(n_x_pos), n_pos, replace=False)
  xbag = [x_pos[pos_indices,...]]

  n_neg = N - n_pos
  xbag.append(generate_negative_bag(x_neg, n_neg))

  xbag = np.concatenate(xbag, axis=0)
  np.random.shuffle(xbag)
  return xbag

def generate_bagged_mnist(x_pos, x_neg, N, batch):
  """
  x ~ (samples, h, w)
  y ~ (samples)
  N ~ int. the size of the bag 

  return:
  bag_x ~ (1, N, h, w, (c))
  bag_y ~ (1, 2)
  """
  print('Set up generator with batch={}'.format(batch))
  # Coin flip for generating a positive or negative bag:
  while True:
    batch_x, batch_y = [], []
    for _ in range(batch):
      y = np.random.choice([0,1])
      y_onehot = np.zeros((1,2), dtype=np.float32)
      y_onehot[0,y] = 1

      if y == 0:
        xbag = generate_negative_bag(x_neg, N)
        xbag = np.expand_dims(xbag, axis=0)
        xbag = np.expand_dims(xbag, axis=-1)
      else:
        xbag = generate_positive_bag(x_pos, x_neg, N)
        xbag = np.expand_dims(xbag, axis=0)
        xbag = np.expand_dims(xbag, axis=-1)
        
      batch_x.append(xbag.astype(np.float32))
      batch_y.append(y_onehot)
    
    yield np.concatenate(batch_x, axis=0), np.concatenate(batch_y, axis=0)

def main(args):
  if args.mnist is not None:
    (train_x, train_y), (test_x, test_y) = mnist.load_data(args.mnist)
  else:
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

  print('train_x:', train_x.shape, train_x.dtype, train_x.min(), train_x.max())
  print('train_y:', train_y.shape)
  print('test_x:', test_x.shape)
  print('test_y:', test_y.shape)
  
  positive_label = np.random.choice(range(10))
  print('using positive label = {}'.format(positive_label))

  train_x_pos, train_x_neg = rearrange_bagged_mnist(train_x, train_y, positive_label)
  test_x_pos, test_x_neg = rearrange_bagged_mnist(test_x, test_y, positive_label)
  print('rearranged training set:')
  print('\ttrain_x_pos:', train_x_pos.shape, train_x_pos.dtype, 
    train_x_pos.min(), train_x_pos.max())
  print('\ttrain_x_neg:', train_x_neg.shape)
  print('\ttest_x_pos:', test_x_pos.shape)
  print('\ttest_x_neg:', test_x_neg.shape)

  generator = generate_bagged_mnist(train_x_pos, train_x_neg,   args.n, args.batch_size)
  val_generator = generate_bagged_mnist(test_x_pos, test_x_neg, args.n, args.batch_size)
  batch_x, batch_y = next(generator)
  print('batch_x:', batch_x.shape, 'batch_y:', batch_y.shape)

  encoder_args = get_encoder_args('mnist')
  model = MilkBatch(input_shape=(28, 28, 1), 
               encoder_args=encoder_args, 
               mode=args.mil,
               batch_size = args.batch_size,
               bag_size = args.n,
               deep_classifier=True)

  # if args.gpus > 1:
  #   print('Duplicating model onto 2 GPUs')
  #   model = tf.keras.utils.multi_gpu_model(model, args.gpus, cpu_merge=True, cpu_relocation=False)

  # batch_input = tf.keras.layers.Input(shape=(args.n, 28,28, 1))
  # batch_logits = []
  # for k in range(args.batch_size):
  #   bag = tf.keras.layers.Lambda(lambda x: x[k,...])(batch_input)
  #   encoding = model(bag)
  #   batch_logits.append(encoding)

  # batch_logits = tf.keras.layers.Concatenate(axis=0)(batch_logits)
  # batch_model = tf.keras.Model(inputs=batch_input, outputs=batch_logits)

  # optimizer = tf.keras.optimizers.Adam(lr=args.lr)
  optimizer = training_utils.AdamAccumulate(lr=args.lr, accum_iters=5)
  model.compile(optimizer=optimizer,
                loss = tf.keras.losses.categorical_crossentropy,
                metrics = ['categorical_accuracy'])
  model.summary()

  if args.pretrained is not None and os.path.exists(args.pretrained):
    print('Restoring weights from {}'.format(args.pretrained))
    model.load_weights(args.pretrained, by_name = True)
  else:
    print('Pretrained model not found ({}). Continuing end 2 end.'.format(args.pretrained))

  model.fit_generator(generator=generator, 
                            validation_data=val_generator,
                            validation_steps=100,
                            steps_per_epoch=args.epoch_steps, 
                            epochs=args.epochs)
  model.save(args.o)
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', default='./bagged_mnist.h5', type=str)
  parser.add_argument('-n', default=100, type=int)
  parser.add_argument('--lr',  default=1e-4, type=float)
  parser.add_argument('--tpu',   default=False, action='store_true')
  parser.add_argument('--mil',   default='attention', type=str)
  parser.add_argument('--gpus',   default=1, type=int)
  parser.add_argument('--mnist', default=None)
  parser.add_argument('--ntest', default=25, type=int)
  parser.add_argument('--decay', default=1e-5, type=float)
  parser.add_argument('--epochs', default=10, type=int)
  parser.add_argument('--pretrained', default=None)
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--epoch_steps', default=1e3, type=int)
  parser.add_argument('--max_fraction_positive', default=0.3, type=int)
  parser.add_argument('--min_fraction_positive', default=0.1, type=int)

  args = parser.parse_args()

  main(args)
