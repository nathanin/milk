import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import time
import cv2

def ImageNetDataset(src, batch=32, xsize=96, ysize=96, n_classes=1000, 
                    parallel=8, buffer=1024, shuffle=True, testing=False):

  def preprocess(image_path, image_label):
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # image = tf.image.crop_to_bounding_box(image, 16, 16, xsize+16, ysize+16)
    # image = tf.random_crop(image, [xsize+32, ysize+32, 3])
    image = tf.image.resize_images(image, [xsize, ysize])

    image = image / 255.
    label = tf.one_hot(image_label, n_classes)
    return image, label

  imagelist = pd.read_csv(src, header=None, index_col=None, sep='\t')
  if shuffle:
    imagelist = imagelist.sample(frac=1)
  n_imgs = imagelist.shape[0]
  print('imagelist', imagelist.shape)

  image_paths = imagelist[1].values
  image_labels = imagelist[0].values
  print(image_paths.shape)
  print(image_labels.shape)

  image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
  image_labels = tf.convert_to_tensor(image_labels, dtype=tf.int32)

  dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
  dataset = dataset.shuffle(1000) # these should be strings
  dataset = dataset.prefetch(256)
  dataset = dataset.map(preprocess, num_parallel_calls=parallel)
  dataset = dataset.prefetch(buffer)
  dataset = dataset.batch(batch)

  return dataset, n_imgs

if __name__ == '__main__':
  print('Testing ImageNet Dataset')
  
  parser = argparse.ArgumentParser()
  parser.add_argument('src')
  args = parser.parse_args()

  src = args.src
  dataset = ImageNetDataset(src, testing=True)
  iterator = dataset.make_one_shot_iterator()
  img_op, label_op = iterator.get_next()

  with tf.Session() as sess:
    print('Pulling first batch:')
    img_, label_ = sess.run([img_op, label_op])
    print('\t', img_.shape, label_.shape)
    print('\t', img_.dtype, label_.dtype)
    print('\t', img_.min(), img_.max())

    times = np.zeros(100)
    for i in range(100):
      tstart = time.time()
      img_, label_ = sess.run([img_op, label_op])
      tend = time.time()
      times[i] = tend - tstart

    print('Average over 100 batches: {} +/- {}'.format(
      np.mean(times), np.std(times)
    ))

