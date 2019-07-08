from MILDataset import MILDataset, create_dataset
from svsutils import PythonIterator, cpramdisk, Slide

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import traceback
import argparse
import time
import cv2
import sys
import os

def test_read(args):
  """
  Write some graphs 
  always according to average time / 10 consecutive reads

  - vary 
  """

  mildataset = MILDataset(args.data_h5)

  if args.test_splitting:
    # Check consistency of the lists, and that they shuffle with each initialization
    mildataset.split_train_val(seed1=999, seed2=np.random.randint(low=100, high=10000))
    ref = mildataset.tr_datasets
    for k in range(5):
      print('\nSpinning up iterator # {}'.format(k))
      it = mildataset.python_iterator(mode='train', subset=3, attr='stage_code', seed=999)
      for _ in it:
        shuf = mildataset.tr_datasets
        matches, misses = 0, 0
        for s in shuf:
          if s not in ref:
            # print('{} not in reference'.format(s))
            misses += 1
          else:
            matches += 1
        print('Matches {} Misses {}'.format(matches, misses))
        break

  # Benchmark reading - python iterator
  if args.test_python:
    print('\nBenchmarking python iterator')
    it = mildataset.python_iterator(mode='train', subset=args.subset, 
                                    attr='stage_code', seed=999)
    i, t = 0, time.time()
    for x, y in it:
      i += 1
      if i % 50 == 0:
        print(i, x.shape, x.dtype, y, y.dtype)
    te = time.time()
    dt = te - t
    mt = dt / i
    print('Python iterator: ', i) 
    print(dt)
    print(mt)

  if args.test_tf and args.eager:
    print('\nBenchmarking TensorFlow iterator in eager mode')
    it = mildataset.tensorflow_iterator(mode='train', subset=args.subset, 
                                        attr='stage_code', seed=999, eager=True)
    i, t = 0, time.time()
    for x, y in it:
      x, y = x.numpy(), y.numpy()
      i += 1
      if i % 10 == 0:
        print(i, x.shape, x.dtype, y, y.dtype)
    te = time.time()
    dt = te - t
    mt = dt / i
    print('TensorFlow iterator: ', i) 
    print(dt)
    print(mt)


  elif args.test_tf and not args.eager:
    print('\nBenchmarking TensorFlow iterator in graph mode')
    ds = mildataset.tensorflow_iterator(mode='train', subset=args.subset, 
                                        attr='stage_code', seed=999, eager=False)
    xop, yop = ds.get_next()
    i, t = 0, time.time()
    # for x, y in it:
    with tf.Session() as sess:
      while True:
        try:
          x, y = sess.run([xop, yop])
          i += 1
          if i % 10 == 0:
            print(i, x.shape, x.dtype, y, y.dtype)
        except: 
          break

    te = time.time()
    dt = te - t
    mt = dt / i
    print('TensorFlow iterator: ', i) 
    print(dt)
    print(mt)

if __name__ == "__main__":

  p = argparse.ArgumentParser()
  p.add_argument('data_h5')

  p.add_argument('--subset', default=100, type=int)
  p.add_argument('--test_splitting', default=False, action='store_true')
  p.add_argument('--test_python', default=False, action='store_true')
  p.add_argument('--test_tf', default=False, action='store_true')

  p.add_argument('--eager', default=False, action='store_true')

  args = p.parse_args()

  if args.eager:
    tf.enable_eager_execution()

  test_read(args)