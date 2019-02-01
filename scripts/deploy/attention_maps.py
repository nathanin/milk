"""
Test classifier on svs's

We need two files to translate between obfuscated filenames, and the 
original svs files:

1. case_dict_obfuscated.pkl is a pickled dictionary with the form:
   {uid: label, ...}
   We ust it to go from the uid's saved in the test list for a particular run to labels

2. uid2slide.pkl is a pickled dictionary with the form:
   {uid: [/path/to/1.svs, /path/to/2.svs,...], ...}
   We use it to translate from the uid given in the test list to a set of slides
    for that particular case. There may be more than one slide, so we'll just choose one.

   Be aware that the slide lists in uid2slide need to point somewhere on the local filesystem.

BUG there's a memory leak

"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import datetime
import hashlib
import shutil
import pickle
import time
import glob
import sys
import cv2
import os
import re

import seaborn as sns
from matplotlib import pyplot as plt

from svs_reader import Slide
from svs_reader.normalize import reinhard
from attimg import draw_attention

from milk.utilities import data_utils
from milk.utilities import model_utils
from milk.utilities import training_utils
from milk.utilities import model_utils
from milk import Milk, MilkEncode, MilkPredict, MilkAttention

sys.path.insert(0, '../experiment')
from encoder_config import encoder_args

def get_wrapped_fn(svs):
  def wrapped_fn(idx):
    coords = svs.tile_list[idx]
    img = svs._read_tile(coords)
    return img, idx

  return wrapped_fn

def get_img_idx(svs, batch_size, prefetch, generate_subset=False, sample=1.0):
  wrapped_fn = get_wrapped_fn(svs)
  def read_region_at_index(idx):
    return tf.py_func(func = wrapped_fn,
                        inp  = [idx],
                        Tout = [tf.float32, tf.int64],
                        stateful = False)            

  # Replace svs.generate_index with a rolled-out generator that minimizes
  # the amount of unnecessary processing to do:
  # if generate_subset:


  dataset = tf.data.Dataset.from_generator(generator=svs.generate_index,
      output_types=tf.int64)
  dataset = dataset.map(read_region_at_index, num_parallel_calls=4)
  dataset = dataset.prefetch(prefetch)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  img, idx = iterator.get_next()
  return img, idx
        
def transfer_to_ramdisk(src, ramdisk):
  """
  Convolve with a random string in case we're in parallel mode.
  """
  basename = os.path.basename(src)
  dst = os.path.join(ramdisk, '{}_{}'.format(
    hashlib.md5('{}'.format(np.random.randn()).encode()).hexdigest(), 
    basename))
  print('Transferring {} --> {}'.format(src, dst))
  shutil.copyfile(src, dst)

  return dst

def read_list(test_file):
  test_list = []
  with open(test_file, 'r') as f:
    for l in f:
      test_list.append(l.replace('\n', ''))
  
  return test_list

def process_slide(svs, encode_model, y_op, att_op, z_pl, args, return_z=False):
  n_tiles = len(svs.tile_list)
  prefetch = min(256, n_tiles)

  # Get tensors for image and index -- spin up a new op 
  # that consumes from the iterator
  img, idx = get_img_idx(svs, args.batch_size, prefetch)
  z_op = encode_model(img)

  batches = 0
  zs = []
  indices = []
  print('Processing {} tiles'.format(n_tiles))
  while True:
    try:
      batches += 1
      z, idx_ = sess.run([z_op, idx])
      zs.append(z)
      indices.append(idx_)
      if batches % 25 == 0:
          print('batch {:04d}\t{}\t{}'.format(
                  batches, z.shape, batches*args.batch_size))
    except tf.errors.OutOfRangeError:
      print('Done')
      break

  zs = np.concatenate(zs, axis=0)
  indices = np.concatenate(indices)
  
  if return_z:
    return zs, indices

  yhat = sess.run(y_op, feed_dict={z_pl: zs})
  print('yhat:', yhat)

  att = sess.run(att_op, feed_dict={z_pl: zs})
  att = np.squeeze(att)
  print('att:', att.shape)

  return yhat, att, indices

"""
Track attention on an included/not-included basis.
'sample' argument controls the % of the whole slide to use in each mcdropout iteration
"""
def process_slide_mcdropout(svs, encode_model, y_op, att_op, z_pl, args):
  yhats = []
  atts = np.zeros((len(svs.tile_list), args.mcdropout_t))-1
  zs, indices_all = process_slide(svs, encode_model, y_op, att_op, 
                              z_pl, args, return_z=True)

  n_tiles = len(svs.tile_list)
  n_sample = int(n_tiles * args.mcdropout_sample)
  print('zs:', zs.shape)
  print('indices:', indices_all.shape)
  for t in range(args.mcdropout_t):
    zs_sample = np.random.choice(range(n_tiles), n_sample)
    zs_ = zs[zs_sample, :]
    indices = indices_all[zs_sample]
    print('sampled {} images from z ({})'.format(n_sample, zs_.shape))

    yhat = sess.run(y_op, feed_dict={z_pl: zs_})
    print('yhat:', yhat)

    att = sess.run(att_op, feed_dict={z_pl: zs_})
    att = np.squeeze(att)
    print('att:', att.shape)

    ## Use returned indices to map attentions back to the tiles
    att_map = np.zeros(len(svs.tile_list))
    att_map[indices] = att
    atts[:,t] = att_map

    yhats.append(yhat)

  yhats = np.concatenate(yhats, axis=0)
  yhat_mean = np.mean(yhats, axis=0, keepdims=True)
  yhat_std = np.std(yhats, axis=0, keepdims=True)
  print('\tyhat_std:', yhat_std)

  atts[atts==-1] = np.nan
  att_mean = np.nanmean(atts, axis=1, keepdims=True)
  att_std  = np.nanstd(atts, axis=1, keepdims=True)

  return yhat_mean, att_mean, yhat_std, att_std, indices_all

def main(args, sess):
  slide_list = read_list(args.test_list)

  print('Found {} slides'.format(len(slide_list)))
  encode_model = MilkEncode(input_shape=(args.input_dim, args.input_dim, 3), 
               encoder_args=encoder_args, deep_classifier=args.deep_classifier)
  if args.deep_classifier:
    input_shape = 256
  else:
    input_shape = 512
  predict_model = MilkPredict(input_shape=[input_shape], mode=args.mil, use_gate=args.gated_attention)
  attention_model = MilkAttention(input_shape=[input_shape], use_gate=args.gated_attention)

  print('setting encoder weights')
  encode_model.load_weights(args.snapshot, by_name=True)
  print('setting predict weights')
  predict_model.load_weights(args.snapshot, by_name=True)
  print('setting attention weights')
  attention_model.load_weights(args.snapshot, by_name=True)

  z_pl = tf.placeholder(shape=(None, input_shape), dtype=tf.float32)
  y_op = predict_model(z_pl)
  att_op = attention_model(z_pl)

  ## Loop over found slides:
  for i, src in enumerate(slide_list):
    print('\nSlide {}'.format(i))
    basename = os.path.basename(src).replace('.svs', '')
    fgpth = os.path.join(args.fgdir, '{}_fg.png'.format(basename))
    att_dst = os.path.join(args.odir, '{}_att.npy'.format(basename))

    if os.path.exists(att_dst) and not args.overwrite:
      print('{} exists. Continuing.'.format(att_dst))
      continue

    ramdisk_path = transfer_to_ramdisk(src, args.ramdisk)  # never use the original src
    try:
      if os.path.exists(fgpth):
        print('Using fg image at : {}'.format(fgpth))
        fgimg = cv2.imread(fgpth, 0)
        svs = Slide(slide_path        = ramdisk_path, 
                    # background_speed  = 'accurate',
                    background_speed  = 'image',
                    background_image  = fgimg,
                    preprocess_fn     = lambda x: (reinhard(x)/255.).astype(np.float32) ,
                    process_mag       = args.mag,
                    process_size      = args.input_dim,
                    oversample_factor = args.oversample,
                    verbose           = False)
      else:
        svs = Slide(slide_path        = ramdisk_path, 
                    background_speed  = 'accurate',
                    preprocess_fn     = lambda x: (reinhard(x)/255.).astype(np.float32) ,
                    process_mag       = args.mag,
                    process_size      = args.input_dim,
                    oversample_factor = args.oversample,
                    verbose           = False)
      
      svs.initialize_output(name='attention', dim=1, mode='tile')
      n_tiles = len(svs.tile_list)

      if not args.mcdropout:
        yhat, att, indices = process_slide(svs, 
            encode_model, y_op, att_op, z_pl, args)
      else:
        yhat, att, yhat_sd, att_sd, indices = process_slide_mcdropout(svs, 
            encode_model, y_op, att_op, z_pl, args)

      print('\tSlide predicted: {}'.format(yhat))

      svs.place_batch(att, indices, 'attention', mode='tile')
      attention_img = np.squeeze(svs.output_imgs['attention'])
      attention_img = attention_img * (1. / attention_img.max())
      attention_img = draw_attention(attention_img, n_bins=50)
      print('attention image:', attention_img.shape, 
            attention_img.dtype, attention_img.min(),
            attention_img.max())

      np.save(att_dst, att)
      img_dst = os.path.join(args.odir, '{}_img.png'.format(basename))
      cv2.imwrite(img_dst, attention_img)
      yhat_dst = os.path.join(args.odir, '{}_ypred.npy'.format(basename))
      np.save(yhat_dst, yhat)

      svs.close()
    except Exception as e:
      print(e)
    finally:
      os.remove(ramdisk_path)

if __name__ == '__main__':
  """
  for instance:

  python attention_maps.py --odir tcga-prad --test_list ./tcga_prad_slides.txt --snapshot <path> --fgdir ./tcga-prad-fg
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--odir',       default=None, type=str)  # Required
  parser.add_argument('--test_list',  default=None, type=str)  # Required
  parser.add_argument('--snapshot',   default=None, type=str)  # Required
  
  parser.add_argument('--mag',        default=5, type=int)
  parser.add_argument('--fgdir',      default='../usable_area/inference', type=str)
  parser.add_argument('--savedir',    default='../experiment/save', type=str)
  parser.add_argument('--ramdisk',    default='/dev/shm', type=str)
  parser.add_argument('--n_classes',  default=2, type=int)
  parser.add_argument('--input_dim',  default=96, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--oversample', default=1.25, type=float)
  parser.add_argument('--randomize',  default=False, action='store_true')
  parser.add_argument('--overwrite',  default=False, action='store_true')

  parser.add_argument('--mil',        default='attention', type=str)
  parser.add_argument('--mcdropout',  default=False, action='store_true')
  parser.add_argument('--mcdropout_t', default=25, type=int)
  parser.add_argument('--deep_classifier', default=True, action='store_false') ## almost-always True
  parser.add_argument('--gated_attention', default=True, action='store_false') ## almost-always True
  parser.add_argument('--mcdropout_sample', default=0.25, type=int)
  args = parser.parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    main(args, sess)
    
