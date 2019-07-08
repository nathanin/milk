#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import time
import sys
import os
import gc
import psutil

from milk.utilities import MILDataset
from memory_profiler import profile

import matplotlib.pyplot as plt

datap = './pnbx-10x-chunk.h5'
ss=25
thr=8

@profile
def main():
  data_factory = MILDataset(datap, crop=128, n_classes=2)
  data_factory.split_train_val('case_id', seed=555)

  memoryused = []
  for e in range(10):
    print(e)
    data_factory.tensorflow_iterator(mode='val', 
          seed=1, batch_size=1, buffer_size=16, 
          subset=ss, threads=thr,
          attr='stage_code', eager=True)
    # for k, (x, y) in enumerate(data_factory.tf_dataset.take(data_factory.len_tf_ds)):
    for k, (x, y) in enumerate(data_factory.tf_dataset):
      memx = psutil.virtual_memory().used / 2 ** 30
      print('\r{} mem: {} '.format(k, memx), end='', flush=True)
      memoryused.append(memx)

    memx = psutil.virtual_memory().used / 2 ** 30
    print('\nEnd val dataset consumption: {}'.format(memx))
    memoryused.append(memx)

    data_factory.clear_mem()
    memx = psutil.virtual_memory().used / 2 ** 30
    print('"Cleared": {}'.format(memx))
    memoryused.append(memx)

    data_factory.close_refs()
    data_factory = MILDataset(datap, crop=128, n_classes=2)
    data_factory.split_train_val('case_id', seed=555)


    data_factory.tensorflow_iterator(mode='train', 
          seed=1, batch_size=1, buffer_size=16, 
          subset=ss, threads=thr,
          attr='stage_code', eager=True)
    for k, (x, y) in enumerate(data_factory.tf_dataset):
      memx = psutil.virtual_memory().used / 2 ** 30
      print('\r{} mem: {} '.format(k, memx), end='', flush=True)
      memoryused.append(memx)

    memx = psutil.virtual_memory().used / 2 ** 30
    print('\nEnd tst dataset consumption: {}'.format(memx))
    memoryused.append(memx)

    data_factory.clear_mem()
    memx = psutil.virtual_memory().used / 2 ** 30
    print('"Cleared": {}'.format(memx))
    memoryused.append(memx)

  del data_factory

  plt.plot(memoryused)
  plt.show()

tf.enable_eager_execution()
main()
