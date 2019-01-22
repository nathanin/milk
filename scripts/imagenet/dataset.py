"""
Spin up a tf.data.Dataset for a list of paths
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import glob
import time
import cv2
import os

def ImageNetRecords(src, batch=32, xsize=96, ysize=96, n_classes=1000, 
                    parallel=8, buffer=1024, shuffle=True):

  def decode(example):
    features = {'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                'height': tf.FixedLenFeature((), tf.int64, default_value=0),
                'width': tf.FixedLenFeature((), tf.int64, default_value=0),
                'encoded': tf.FixedLenFeature((), tf.string, default_value=''),}

    pf = tf.parse_single_example(example, features)
    height = tf.squeeze(pf['height'])
    width = tf.squeeze(pf['width'])
    label = tf.squeeze(pf['label'])

    image = pf['encoded']
    image = tf.image.decode_jpeg(image)


    # image = tf.decode_raw(image, tf.uint8)
    # img_shape = tf.stack([height, width, 3], axis=0)
    # image = tf.reshape(image, img_shape)

    # image = tf.cast(image, tf.float32)

    return image, label

  def preprocess(example):
    # image = tf.read_file(image_path)
    # image = tf.image.decode_jpeg(image, channels=3)
    image, label = decode(example)

    # image = tf.image.crop_to_bounding_box(image, 16, 16, xsize+16, ysize+16)
    # image = tf.random_crop(image, [xsize+32, ysize+32, 3])
    image = tf.image.resize_images(image, [xsize, ysize])

    image = tf.multiply(image, 1 / 255.)
    label = tf.one_hot(label, n_classes)
    return image, label

  record_patt = os.path.join(src, 'train-*')
  filenames = tf.data.Dataset.list_files(record_patt)

  dataset = filenames.apply(
    tf.contrib.data.parallel_interleave(
        lambda filename: tf.data.TFRecordDataset(filename),
        cycle_length=24))
  # dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.repeat()
  dataset = dataset.map(preprocess, num_parallel_calls=parallel)
  dataset = dataset.prefetch(buffer)
  dataset = dataset.batch(batch)

  return dataset


if __name__ == '__main__':
  print('Testing ImageNet Dataset')
  
  parser = argparse.ArgumentParser()
  parser.add_argument('src')
  args = parser.parse_args()

  src = args.src
  dataset = ImageNetRecords(src)

  iterator = dataset.make_one_shot_iterator()
  img_op, label_op = iterator.get_next()

  with tf.Session() as sess:
    print('Pulling first batch:')
    img_, label_ = sess.run([img_op, label_op])
    print('\t', img_.shape, label_.shape)
    print('\t', img_.dtype, label_.dtype)
    print('\t', img_.min(), img_.max())

    for i, im in enumerate(img_):
      oname = 'testbatch/{:04d}.jpg'.format(i)
      cv2.imwrite(oname, im[:,:,::-1]*255)

    times = np.zeros(100)
    for i in range(100):
      tstart = time.time()
      img_, label_ = sess.run([img_op, label_op])
      tend = time.time()
      times[i] = tend - tstart

    print('Average over 100 batches: {} +/- {}'.format(
      np.mean(times), np.std(times)
    ))

