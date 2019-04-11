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

from milk.encoder_config import get_encoder_args

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

  # Shuffle indices
  def generate_shuffled_index():
    indices = np.arange(len(svs.tile_list))
    np.random.shuffle(indices)
    for i in indices:
      yield i

  dataset = tf.data.Dataset.from_generator(generator=generate_shuffled_index,
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

def process_slide(svs, model, args, return_z=None):
  n_tiles = len(svs.tile_list)
  prefetch = min(256, n_tiles)

  # Get tensors for image and index -- spin up a new op 
  # that consumes from the iterator
  iterator = get_img_idx(svs, args.batch_size, prefetch)

  batches = 0
  zs = []
  indices = []
  print('Processing {} tiles'.format(n_tiles))
  for imgs, idx_ in iterator:
    batches += 1
    z = model.encode_bag(imgs, training=False, return_z=True)
    zs.append(z)
    indices.append(idx_)
    if batches % 10 == 0:
        print('batch {:04d}\t{}\t{}'.format( batches, z.shape, batches*args.batch_size ))

  zs = tf.concat(zs, axis=0)
  indices = np.concatenate(indices)
  print('zs: ', zs.shape, zs.dtype)
  print('indices: ', indices.shape)

  if args.mil == 'attention':
    z_att, instance_score = model.mil_attention(zs, training=False, verbose=True, return_att=True)
    yhat = model.apply_classifier(z_att, training=False, verbose=True)
  elif args.mil == 'instance':
    instance_score = zs[:,1]
    yhat = tf.reduce_mean(zs, axis=0, keepdims=True)
  else:
    print('Unsupported MIL type {}.'.format(args.mil))

  print('yhat:', yhat)

  instance_score = np.squeeze(instance_score)
  print('att:', instance_score.shape)

  return yhat, instance_score, indices

def main(args):
  # Translate obfuscated file names to paths if necessary
  test_list = os.path.join(args.testdir, '{}.txt'.format(args.timestamp))
  test_list = read_test_list(test_list)
  test_unique_ids = [os.path.basename(x).replace('.npy', '') for x in test_list]
  if args.randomize:
    np.random.shuffle(test_unique_ids)

  if args.max_slides:
    test_unique_ids = test_unique_ids[:args.max_slides]

  slide_list, slide_labels = get_slidelist_from_uids(test_unique_ids)

  print('Found {} slides'.format(len(slide_list)))

  snapshot = os.path.join(args.savedir, '{}.h5'.format(args.timestamp))
  # trained_model = load_model(snapshot)
  # if args.mcdropout:
  #   encoder_args['mcdropout'] = True

  encoder_args = get_encoder_args(args.encoder)
  model = MilkEager(encoder_args=encoder_args,
                    mil_type=args.mil, 
                    batch_size=args.batch_size, 
                    deep_classifier=args.deep_classifier,
                    temperature=args.temperature)

  x_pl = np.zeros((1, args.batch_size, args.input_dim, args.input_dim, 3), dtype=np.float32)
  yhat = model(tf.constant(x_pl), verbose=True)
  print('yhat:', yhat.shape)

  print('setting model weights')
  model.load_weights(snapshot, by_name=True)

  ## Loop over found slides:
  yhats = []
  ytrues = []
  for i, (src, lab) in enumerate(zip(slide_list, slide_labels)):
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
                  # preprocess_fn     = lambda x: (reinhard(x)/255.).astype(np.float32),
                  preprocess_fn     = lambda x: (x/255.).astype(np.float32),
                  process_mag       = args.mag,
                  process_size      = args.input_dim,
                  oversample_factor = args.oversample,
                  verbose           = False)
    else:
      ## require precomputed background; Exit.
      print('Required foreground image not found ({})'.format(fgpth))
      continue
    
    svs.initialize_output(name='attention', dim=1, mode='tile')
    n_tiles = len(svs.tile_list)

    yhat, att, indices = process_slide(svs, model, args)
    print('returned attention:', np.min(att), np.max(att), att.shape)

    yhat = yhat.numpy()
    yhats.append(yhat)
    ytrues.append(lab)
    print('\tSlide label: {} predicted: {}'.format(lab, yhat))

    svs.place_batch(att, indices, 'attention', mode='tile')
    attention_img = np.squeeze(svs.output_imgs['attention'])
    attention_img = attention_img * (1. / attention_img.max())
    attention_img = draw_attention(attention_img, n_bins=25)
    print('attention image:', attention_img.shape, 
          attention_img.dtype, attention_img.min(),
          attention_img.max())

    dst = os.path.join(args.odir, args.timestamp, '{}_{}_{:3.3f}_att.npy'.format(basename, lab, yhat[0,1]))
    np.save(dst, att)

    dst = os.path.join(args.odir, args.timestamp, '{}_{}_{:3.3f}_img.png'.format(basename, lab, yhat[0,1]))
    cv2.imwrite(dst, attention_img)

    try:
      svs.close()
      os.remove(ramdisk_path)
    except:
      print('{} already removed'.format(ramdisk_path))

  yhats = np.concatenate(yhats, axis=0)
  ytrues = np.array(ytrues)
  acc = (np.argmax(yhats, axis=-1) == ytrues).mean()
  print(acc)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Required -- no default  
  parser.add_argument('--timestamp',  default=None, type=str)

  # Changed often -- with defaults
  parser.add_argument('--mag',        default=5, type=int)
  parser.add_argument('--odir',       default='images-attention', type=str)
  parser.add_argument('--fgdir',      default='../usable_area/inference', type=str)
  parser.add_argument('--savedir',    default='../experiment/save', type=str)
  parser.add_argument('--testdir',    default='../experiment/test_lists', type=str)
  parser.add_argument('--ramdisk',    default='/dev/shm', type=str)
  parser.add_argument('--n_classes',  default=2, type=int)
  parser.add_argument('--input_dim',  default=96, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--oversample', default=1., type=float)

  parser.add_argument('--randomize',  default=False, action='store_true')
  parser.add_argument('--max_slides',  default=None, type=int)

  parser.add_argument('--mil',        default='attention', type=str)
  # parser.add_argument('--mcdropout',  default=False, action='store_true')
  # parser.add_argument('--mcdropout_t', default=25, type=int)
  parser.add_argument('--deep_classifier', default=True, action='store_true')
  parser.add_argument('--encoder', default='big', type=str)
  parser.add_argument('--temperature', default=1., type=float)
  parser.add_argument('--cls_normalize', default=True, type=bool)
  parser.add_argument('--gated_attention', default=True, action='store_false')
  parser.add_argument('--mcdropout_sample', default=0.25, type=int)
  args = parser.parse_args()

  if not os.path.exists(os.path.join(args.odir, args.timestamp)):
    os.makedirs(os.path.join(args.odir, args.timestamp))
  assert args.timestamp is not None

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  tf.enable_eager_execution(config=config)

  main(args)
    
