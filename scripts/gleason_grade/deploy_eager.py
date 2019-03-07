"""
Test classifier on svs's

"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import shutil
import glob
import cv2
import os

from svs_reader import Slide, reinhard

import seaborn as sns
from matplotlib import pyplot as plt

from milk.utilities import ClassificationDataset
from milk.eager import ClassifierEager
from milk.encoder_config import get_encoder_args

from sklearn.metrics import roc_auc_score, roc_curve, classification_report

# Hard coded into the dataset
class_mnemonics = {
  0: 'GG3',
  1: 'GG4',
  2: 'GG5',
  3: 'BN',
  4: 'ST',
}

colors = np.array([[175, 33, 8],
           [20, 145, 4],
           [177, 11, 237],
           [14, 187, 235],
           [255, 255, 255]
          ])

def colorize(rgb, prob):
  prob = cv2.resize(prob, dsize=rgb.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
  prob_sum = prob.sum(axis=-1)
  mask = np.zeros(list(prob.shape[:2])+[3], dtype=np.uint8) 
  amax = np.argmax(prob, axis=-1)
  amax[prob_sum < 1.-1e-3] = 5
  for k in range(5):
    mask[amax==k] = colors[k,:]
  
  colored = rgb * 0.6 + mask * 0.3
  return colored

def get_wrapped_fn(svs):
  def wrapped_fn(idx):
    coords = svs.tile_list[idx]
    img = svs._read_tile(coords)
    return img, idx
  return wrapped_fn

def get_img_idx(svs, batch_size, prefetch, threads=4):
  wrapped_fn = get_wrapped_fn(svs)
  def read_region_at_index(idx):
    return tf.py_func(func = wrapped_fn,
                      inp  = [idx],
                      Tout = [tf.float32, tf.int64],
                      stateful = False)      

  dataset = tf.data.Dataset.from_generator(generator=svs.generate_index,
    output_types=tf.int64)
  dataset = dataset.map(read_region_at_index, num_parallel_calls=threads)
  dataset = dataset.prefetch(prefetch)
  dataset = dataset.batch(batch_size)
  # iterator = dataset.make_one_shot_iterator()
  # img, idx = iterator.get_next()
  iterator = tf.contrib.eager.Iterator(dataset)
  return iterator

def transfer_to_ramdisk(src, ramdisk):
  print('Transferring {} --> {}'.format(src, ramdisk))
  basename = os.path.basename(src)
  dst = os.path.join(ramdisk, basename)
  shutil.copyfile(src, dst)
  return dst

def vect_to_tile(x, target):
  batch = x.shape[0]
  dim = x.shape[-1]
  return np.tile(x, target*target).reshape(batch, target, target, dim)


def main(args):
  ## Search for slides
  slide_list = sorted(glob.glob(os.path.join(args.slide_dir, '*.svs')))
  print('Found {} slides'.format(len(slide_list)))
  if args.shuffle:
    np.random.shuffle(slide_list)

  encoder_args = get_encoder_args(args.encoder)
  model = ClassifierEager(encoder_args=encoder_args, deep_classifier=True,
                          n_classes=args.n_classes)
  fake_data = tf.constant(np.zeros((1, args.input_dim, args.input_dim, 3), dtype=np.float32))
  yhat_ = model(fake_data)
  model.load_weights(args.snapshot)
  model.summary()
  if not os.path.exists(args.save_dir):
    # shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

  ## Loop over found slides:
  for src in slide_list:
    basename = os.path.basename(src).replace('.svs', '')
    dst = os.path.join(args.save_dir, '{}.npy'.format(basename))
    if os.path.exists(dst):
      print('{} exists. Skipping'.format(dst))
      continue

    ramdisk_path = transfer_to_ramdisk(src, args.ramdisk)  # never use the original src
    try:
      fgpth = os.path.join(args.fgdir, '{}_fg.png'.format(basename))
      fgimg = cv2.imread(fgpth, 0)
      svs = Slide(slide_path = ramdisk_path, 
            preprocess_fn = lambda x: (x/255.).astype(np.float32) ,
            background_speed  = 'image',
            background_image  = fgimg,
            process_mag=args.mag,
            process_size=args.input_dim,
            oversample_factor=1.1)
      svs.initialize_output(name='prob', dim=args.n_classes, mode='full')
      svs.initialize_output(name='rgb', dim=3, mode='full')
      n_tiles = len(svs.tile_list)
      prefetch = min(512, n_tiles)
      # Get tensors for image an index
      iterator = get_img_idx(svs, args.batch_size, prefetch)
      batches = 0
      for img_, idx_ in iterator:
        batches += 1
        yhat = model(img_, training=False)
        yhat = yhat.numpy()
        idx_ = idx_.numpy()
        img_ = img_.numpy()
        yhat = vect_to_tile(yhat, args.input_dim)
        svs.place_batch(yhat, idx_, 'prob', mode='full')
        svs.place_batch(img_, idx_, 'rgb',  mode='full')
        if batches % 50 == 0:
          print('\tbatch {:04d}'.format(batches))

      svs.make_outputs(reference='prob')
      prob_img = svs.output_imgs['prob']
      rgb_img = svs.output_imgs['rgb'] * 255
      color_img = colorize(rgb_img, prob_img)
      dst = os.path.join(args.save_dir, '{}.npy'.format(basename))
      np.save(dst, (prob_img * 255).astype(np.uint8))
      dst = os.path.join(args.save_dir, '{}.jpg'.format(basename))
      cv2.imwrite(dst, rgb_img[:,:,::-1])
      dst = os.path.join(args.save_dir, '{}_c.jpg'.format(basename))
      cv2.imwrite(dst, color_img[:,:,::-1])
    except Exception as e:
      print(e)
    finally:
      try:
        print('Closing SVS')
        svs.close()
      except:
        print('No SVS to close')

      os.remove(ramdisk_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--slide_dir', default='../dataset/svs/', type=str)
  parser.add_argument('--shuffle', default=False, action='store_true')
  parser.add_argument('--snapshot', default='./pretrained.h5', type=str)
  parser.add_argument('--n_classes', default=5, type=int)
  parser.add_argument('--input_dim', default=96, type=int)
  parser.add_argument('--mag', default=5, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--save_dir', default=None, type=str)
  parser.add_argument('--ramdisk', default='/dev/shm', type=str)
  parser.add_argument('--encoder', default='big', type=str)
  parser.add_argument('--fgdir', default='../usable_area/inference', type=str)
  args = parser.parse_args()

  assert args.save_dir is not None

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  tf.enable_eager_execution(config=config)
  main(args)
  
