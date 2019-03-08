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
from tensorflow.keras.datasets import mnist
import numpy as np

import argparse
import os 

from milk.eager import MilkEager
from milk.encoder_config import get_encoder_args

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

  generator = generate_bagged_mnist(train_x_pos, train_x_neg, args.n, args.batch_size)
  val_generator = generate_bagged_mnist(test_x_pos, test_x_neg, args.n, args.batch_size)
  batch_x, batch_y = next(generator)
  print('batch_x:', batch_x.shape, 'batch_y:', batch_y.shape)

  encoder_args = get_encoder_args('mnist')
  model = MilkEager(encoder_args=encoder_args, 
               mil_type=args.mil,
               deep_classifier=True,
  )
  y_dummy = model(batch_x, verbose=True)
  model.summary()

  if args.pretrained is not None and os.path.exists(args.pretrained):
    model.load_weights(args.pretrained, by_name = True)
  else:
    print('Pretrained model not found ({}). Continuing end 2 end.'.format(args.pretrained))

  if args.gpus > 1:
    print('Duplicating model onto 2 GPUs')
    model = tf.keras.utils.multi_gpu_model(model, args.gpus, cpu_merge=True, cpu_relocation=False)

  optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)

  try:
    for k in range(int(args.steps_per_epoch * args.epochs)):
      with tf.GradientTape() as tape:
        x, y = next(generator)
        yhat = model(x, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y_true=tf.constant(y, dtype=tf.float32), y_pred=yhat)

      grads = tape.gradient(loss, model.variables)
      optimizer.apply_gradients(zip(grads, model.variables))

      if k % 50 == 0:
        print('{:06d}: loss={:3.5f}'.format(k, np.mean(loss)))
        for y_, yh_ in zip(y, yhat):
          print('\t{} {}'.format(y_, yh_))


  except KeyboardInterrupt:
    print('Keyboard interrupt caught')

  except Exception as e:
    print('Other error caught')
    print(type(e))
    print(e)

  finally:
    model.save_weights(args.o)
    print('Saved model: {}'.format(args.o))
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', default='./bagged_mnist_eager.h5', type=str)
  parser.add_argument('-n', default=50, type=int)
  parser.add_argument('--lr',   default=1e-5, type=float)
  parser.add_argument('--tpu',  default=False, action='store_true')
  parser.add_argument('--mil',  default='attention', type=str)
  parser.add_argument('--gpus', default=1, type=int)
  parser.add_argument('--mnist', default=None)
  parser.add_argument('--ntest', default=25, type=int)
  parser.add_argument('--decay', default=1e-5, type=float)
  parser.add_argument('--epochs', default=10, type=int)
  parser.add_argument('--pretrained', default=None)
  parser.add_argument('--steps_per_epoch', default=1e3, type=int)
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--max_fraction_positive', default=0.3, type=int)
  parser.add_argument('--min_fraction_positive', default=0.1, type=int)

  args = parser.parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  tf.enable_eager_execution(config=config)

  main(args)
