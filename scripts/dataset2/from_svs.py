from milk import MILDataset, create_dataset
from svsutils import PythonIterator, cpramdisk, Slide, TensorflowIterator

import tensorflow as tf
import numpy as np
import pandas as pd
import traceback
import argparse
import cv2
import sys
import os

def stack_tiles(slide, args):
  print('Streaming from slide with {} tiles'.format(len(slide.tile_list)))
  stack = []
  skipped , saved = 0, 0
  it_factory = PythonIterator(slide, args)
  # eager_iterator = TensorflowIterator(slide, args, dtypes=[tf.uint8, tf.int64])
  # eager_iterator = eager_iterator.make_iterator()

  tmax = args.tmax if args.tmax > 0 else len(slide.tile_list)
  for k, (img, _) in enumerate(it_factory.yield_one()):
  # for k, (img, idx) in enumerate(eager_iterator):
    # Skip white and black tiles
    # print(img)
    # print(img.shape, img.dtype)
    # img = np.squeeze(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    white = gray > args.thresh_white
    black = gray < args.thresh_black
    thresh = np.prod(gray.shape) * args.thresh_pct
    if white.sum() > thresh or black.sum() > thresh:
      skipped += 1
      continue
    if k % 50 == 0:
      print('Tile {:04d}: {} \t skipped: {} saved: {}'.format(
        k, img.shape, skipped, saved))
    
    saved += 1
    stack.append(img[:,:,::-1])
    if saved >= tmax:
      print('Hit tmax: {}'.format(tmax))
      break

  # del eager_iterator
  stack = np.stack(stack, axis=0)

  print('Finished {:04d}: {} \t skipped: {}'.format(k, img.shape, skipped))
  print('Returning stack: {} ({})'.format(stack.shape, stack.dtype))
  return stack


def load_foreground_npy(npypath):
  print('fg : {}'.format(npypath))
  mask = np.argmax(np.load(npypath), axis=-1)
  mask = (mask == 1).astype(np.uint8)

  return mask

def read_labels(label_path, attrs):
  if os.path.splitext(label_path)[-1] == '.tsv':
    sep = '\t'
  elif os.path.splitext(label_path)[-1] == '.csv':
    sep = ','
  labels = pd.read_csv(label_path, index_col=None, header=0, sep=sep)  
  d = {}
  for c in range(labels.shape[0]):
    uid = labels.loc[c, 'unique_id']
    d[uid] = [labels.loc[c, k] for k in attrs]
  return d

def main(args):
  ## Read in the slides
  slides = []
  with open(args.slides, 'r') as f:
    for line in f:
      lstrip = line.strip()
      if os.path.exists(lstrip):
        slides.append(lstrip)
      else:
        print(lstrip)
  print('Working with {} slides'.format(len(slides)))

  ## Initialize the dataset; store the metadata file
  with open(args.meta_file, 'r') as f:
    lines = [line for line in f]

  ## Read in the labels
  attrs = ['case_id', 'stage_str', 'stage_code']
  labels = read_labels(args.labels, attrs)
  print('Working with labels:', len(labels))

  ## Create the dataset
  meta_string = ''.join(lines)
  if not os.path.exists(args.data_h5):
    create_dataset(args.data_h5, meta_string)
  else:
    print('HDF5 {} already exists'.format(args.data_h5))

  ## Load the dataset
  mildataset = MILDataset(args.data_h5, meta_string)

  # slides = [s for s in slides if s not in mildataset.data_group_names]
  for i, src in enumerate(slides):
    print('\n[\t{}/{}\t]'.format(i, len(slides)))
    print('File {:04d} {} --> {}'.format(i, src, args.ramdisk))
    basename = os.path.splitext(os.path.basename(src))[0]

    try:
      lab = labels[basename]
      print(basename, lab)
      if lab[-1] > 1:
        print('Skipping unused labels')
        continue
    except Exception as e:
      print('basename {} no labels.'.format(basename))
      traceback.print_tb(e.__traceback__)

    rdsrc = cpramdisk(src, args.ramdisk)
    try:
      slide = Slide(rdsrc, args)
      tile_stack = stack_tiles(slide, args)
      mildataset.new_dataset(basename, tile_stack, attrs, lab)

    except Exception as e:
      print('Breaking')
      traceback.print_tb(e.__traceback__)

    finally:
      print('Removing {}'.format(rdsrc))
      os.remove(rdsrc)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('data_h5')
  p.add_argument('meta_file')
  p.add_argument('slides')
  p.add_argument('labels')

  # p.add_argument('--nmax', default=10, type=int)
  p.add_argument('--tmax', default=1000, type=int)
  p.add_argument('--clobber', default=False, action='store_true')
  p.add_argument('--thresh_white', default=230, type=int)
  p.add_argument('--thresh_black', default=20,  type=int)
  p.add_argument('--thresh_pct',   default=0.15, type=float)

  # Slide options
  p.add_argument('-r', dest='ramdisk', default='/dev/shm', type=str)
  p.add_argument('-b', dest='batchsize', default=1, type=int)
  p.add_argument('--mag',   dest='process_mag', default=10, type=int)
  p.add_argument('--chunk', dest='process_size', default=196, type=int)
  p.add_argument('--bg',    dest='background_speed', default='fast', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.0, type=float)

  args = p.parse_args()

  tf.enable_eager_execution()
  main(args)