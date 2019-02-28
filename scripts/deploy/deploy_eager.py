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

BUG there's a leak

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

from svs_reader import Slide, reinhard
from attimg import draw_attention

from milk.utilities import data_utils
from milk.utilities import model_utils
from milk.utilities import training_utils
from milk.utilities import model_utils
from milk.eager import MilkEager

# uid2label = pickle.load(open('../dataset/case_dict_obfuscated.pkl', 'rb'))
uid2label = pickle.load(open('../dataset/cases_md5.pkl', 'rb'))
uid2slide = pickle.load(open('../dataset/uid2slide.pkl', 'rb'))

from milk.encoder_config import big_args as encoder_args

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
  # iterator = dataset.make_one_shot_iterator()

  iterator = tf.contrib.eager.Iterator(dataset)
  return iterator
        
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

def vect_to_tile(x, target):
    batch = x.shape[0]
    dim = x.shape[-1]
    return np.tile(x, target*target).reshape(batch, target, target, dim)

def read_test_list(test_file):
  test_list = []
  with open(test_file, 'r') as f:
    for l in f:
      test_list.append(l.replace('\n', ''))
  
  return test_list

def get_slidelist_from_uids(uids, process_all=True):
  """ Translate the numpy name into a path to some slides

  Each case can have multiple slides, so decide what to do with them
  """
  slide_list_out = []
  labels = []
  for uid in uids:
    try:
      slide_list = uid2slide[uid]
      if len(slide_list) > 0:
        if process_all:
          slide_list_out += slide_list
          labels += [uid2label[uid]]*len(slide_list)
        else:
          slide_list_out.append(np.random.choice(slide_list))
          labels.append(uid2label[uid])
      else:
        print('WARNING UID: {} found no associated slides'.format(uid))
    except:
      print('WARNING UID: {} not found in slide list'.format(uid))

  print('Testing on slides:')
  for i, (s, l) in enumerate(zip(slide_list_out, labels)):
    print('{:03d} {}\t\t{}'.format(i, s, l))
  return slide_list_out, labels

def process_slide(svs, model, args, return_z=False):
  n_tiles = len(svs.tile_list)
  prefetch = min(512, n_tiles)

  # Get tensors for image and index -- spin up a new op 
  # that consumes from the iterator
  iterator = get_img_idx(svs, args.batch_size, prefetch)

  batches = 0
  zs = []
  indices = []
  print('Processing {} tiles'.format(n_tiles))
  for imgs, idx_ in iterator:
    batches += 1
    z = model.encode_bag(imgs, batch_size=args.batch_size, training=True, return_z=True)
    zs.append(z)
    indices.append(idx_)
    if batches % 10 == 0:
        print('batch {:04d}\t{}\t{}'.format( batches, z.shape, batches*args.batch_size ))

  zs = tf.concat(zs, axis=0)
  indices = np.concatenate(indices)
  print('zs: ', zs.shape, zs.dtype)
  print('indices: ', indices.shape)

  if args.mil == 'average':
    z_mil = tf.reduce_mean(zs, axis=0, keep_dims=True)
    att = None
  elif args.mil == 'attention':
    z_mil, att = model.mil_attention(zs, training=False, verbose=True, return_att=True)
    att = np.squeeze(att)
    print('att:', att.shape)

  yhat = model.apply_classifier(z_mil, training=False, verbose=True)
  print('yhat:', yhat)

  # de-reference the iterator maybe?
  iterator = []
  del iterator

  return yhat, att, indices

"""
Track attention on an included/not-included basis.
'sample' argument controls the % of the whole slide to use in each mcdropout iteration
"""
# def process_slide_mcdropout(svs, encode_model, args):
#   yhats = []
#   atts = np.zeros((len(svs.tile_list), args.mcdropout_t))-1
#   zs, indices_all = process_slide(svs, encode_model, y_op, att_op, 
#                               z_pl, args, return_z=True)

#   n_tiles = len(svs.tile_list)
#   n_sample = int(n_tiles * args.mcdropout_sample)
#   print('zs:', zs.shape)
#   print('indices:', indices_all.shape)
#   for t in range(args.mcdropout_t):
#     zs_sample = np.random.choice(range(n_tiles), n_sample)
#     zs_ = zs[zs_sample, :]
#     indices = indices_all[zs_sample]
#     print('sampled {} images from z ({})'.format(n_sample, zs_.shape))

#     yhat = sess.run(y_op, feed_dict={z_pl: zs_})
#     print('yhat:', yhat)

#     att = sess.run(att_op, feed_dict={z_pl: zs_})
#     att = np.squeeze(att)
#     print('att:', att.shape)

#     ## Use returned indices to map attentions back to the tiles
#     att_map = np.zeros(len(svs.tile_list))
#     att_map[indices] = att
#     atts[:,t] = att_map

#     yhats.append(yhat)

#   yhats = np.concatenate(yhats, axis=0)
#   yhat_mean = np.mean(yhats, axis=0, keepdims=True)
#   yhat_std = np.std(yhats, axis=0, keepdims=True)
#   print('\tyhat_std:', yhat_std)

#   atts[atts==-1] = np.nan
#   att_mean = np.nanmean(atts, axis=1, keepdims=True)
#   att_std  = np.nanstd(atts, axis=1, keepdims=True)

#   return yhat_mean, att_mean, yhat_std, att_std, indices_all

def read_list(test_file):
  test_list = []
  with open(test_file, 'r') as f:
    for l in f:
      test_list.append(l.replace('\n', ''))
  
  return test_list

def main(args):
  # Translate obfuscated file names to paths if necessary
  slide_list = read_list(args.test_list)
  print('Found {} slides'.format(len(slide_list)))

  # snapshot = os.path.join(args.savedir, '{}.h5'.format(args.timestamp))
  # trained_model = load_model(snapshot)
  # if args.mcdropout:
  #   encoder_args['mcdropout'] = True

  model = MilkEager( encoder_args=encoder_args, mil_type=args.mil, deep_classifier=args.deep_classifier )

  x_pl = np.zeros((1, args.batch_size, args.input_dim, args.input_dim, 3), dtype=np.float32)
  yhat = model(tf.constant(x_pl), verbose=True)
  print('yhat:', yhat.shape)

  print('setting model weights')
  model.load_weights(args.snapshot, by_name=True)

  ## Loop over found slides:
  yhats = []
  for i, src in enumerate(slide_list):
    print('\nSlide {}'.format(i))
    basename = os.path.basename(src).replace('.svs', '')
    fgpth = os.path.join(args.fgdir, '{}_fg.png'.format(basename))
    if os.path.exists(fgpth):
      ramdisk_path = transfer_to_ramdisk(src, args.ramdisk)  # never use the original src
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
      ## require precomputed background; Exit.
      print('Required foreground image not found ({})'.format(fgpth))
      continue
    
    svs.initialize_output(name='attention', dim=1, mode='tile')

    # if args.mcdropout:
    #   yhat, att, yhat_sd, att_sd, indices = process_slide_mcdropout(svs, model, args)
    # else:
    yhat, att, indices = process_slide(svs, model, args)

    yhats.append(yhat)
    print('\tSlide predicted: {}'.format(yhat))

    if args.mil == 'average':
      print('Average MIL; continuing')
    elif args.mil == 'attention':
      svs.place_batch(att, indices, 'attention', mode='tile')
      attention_img = np.squeeze(svs.output_imgs['attention'])
      attention_img = attention_img * (1. / attention_img.max())
      attention_img = draw_attention(attention_img, n_bins=50)
      print('attention image:', attention_img.shape, 
            attention_img.dtype, attention_img.min(),
            attention_img.max())

      dst = os.path.join(args.odir, '{}_att.npy'.format(basename))
      np.save(dst, att)

      dst = os.path.join(args.odir, '{}_img.png'.format(basename))
      cv2.imwrite(dst, attention_img)

    yhat_dst = os.path.join(args.odir, '{}_ypred.npy'.format(basename))
    np.save(yhat_dst, yhat)

    try:
      svs.close()
      os.remove(ramdisk_path)
      del svs
    except:
      print('{} already removed'.format(ramdisk_path))

if __name__ == '__main__':
  """
  for instance:

  python deploy_eager.py \
    --odir tcga-prad \
    --test_list ./tcga_prad_slides.txt \
    --snapshot <path> \
    --fgdir ./tcga-prad-fg
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--odir',       default=None, type=str)  # Required
  parser.add_argument('--test_list',  default=None, type=str)  # Required
  parser.add_argument('--snapshot',   default=None, type=str)  # Required
  
  parser.add_argument('--mag',        default=5, type=int)
  parser.add_argument('--fgdir',      default='tcga-prad-fg', type=str)
  parser.add_argument('--ramdisk',    default='/dev/shm', type=str)
  parser.add_argument('--n_classes',  default=2, type=int)
  parser.add_argument('--input_dim',  default=96, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--oversample', default=1.1, type=float)
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
  tf.enable_eager_execution(config=config)

  # with tf.Session(config=config) as sess:
  main(args)
    
