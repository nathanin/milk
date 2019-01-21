import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import shutil
import glob
import time
import cv2
import os

def CifarRecords(src, batch=32, xsize=32, ysize=32, n_classes=10, 
                    parallel=8, buffer=1024, shuffle=True):

  def decode(example):
    features = {'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                'image': tf.FixedLenFeature((), tf.string, default_value=''),}
    pf = tf.parse_single_example(example, features)
    label = tf.squeeze(pf['label'])
    image = pf['image']
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.reshape(image, (32, 32, 3))
    return image, label

  def preprocess(example):
    image, label = decode(example)
    image = tf.image.random_flip_left_right(image)
    image = tf.multiply(tf.cast(image, tf.float32), 1 / 255.)
    label = tf.one_hot(label, n_classes)
    return image, label

  dataset = tf.data.TFRecordDataset(src)
  dataset = dataset.repeat()
  dataset = dataset.shuffle(5000)
  dataset = dataset.map(preprocess, num_parallel_calls=parallel)
  dataset = dataset.prefetch(buffer)
  dataset = dataset.batch(batch)
  return dataset


if __name__ == '__main__':
  print('Testing CIFAR-10 Dataset')
  
  parser = argparse.ArgumentParser()
  parser.add_argument('src')
  parser.add_argument('--ntest', default=100, type=int)
  args = parser.parse_args()

  src = args.src
  assert os.path.exists(src)
  dataset = CifarRecords(src)

  iterator = dataset.make_one_shot_iterator()
  img_op, label_op = iterator.get_next()

  if os.path.exists('testbatch'):
    shutil.rmtree('testbatch')
  os.makedirs('testbatch')

  with tf.Session() as sess:
    print('Pulling first batch:')
    img_, label_ = sess.run([img_op, label_op])
    print('\t', img_.shape, label_.shape)
    print('\t', img_.dtype, label_.dtype)
    print('\t', img_.min(), img_.max())

    for i, im in enumerate(img_):
      oname = 'testbatch/{:04d}.jpg'.format(i)
      cv2.imwrite(oname, im[:,:,::-1]*255)

    times = np.zeros(args.ntest)
    for i in range(args.ntest):
      tstart = time.time()
      img_, label_ = sess.run([img_op, label_op])
      tend = time.time()
      times[i] = tend - tstart

      out='examples/{:04d}.jpg'.format(i)

    print('Average over {} batches: {} +/- {}'.format(
      args.ntest, np.mean(times), np.std(times)
    ))

