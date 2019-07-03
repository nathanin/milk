#!/usr/bin/env python

"""

"""

import argparse
import cv2
import glob
import os

from scipy.special import softmax
import numpy as np
import seaborn as sns
colors = np.array([[255, 255, 57], # Bright yellow
           [198, 27, 27],  # Red
           [11, 147, 8], # Green
           [252, 141, 204],# Pink
           [255, 255, 255],
          ])
mixture = [0.5, 0.5]

# TODO fix to always return sorted lists
def matching_basenames(l1, l2):
  l1_base = [os.path.splitext(os.path.basename(x))[0].split('_')[0] for x in l1]
  l2_base = [os.path.splitext(os.path.basename(x))[0].split('_')[0] for x in l2]
  matching = np.intersect1d(l1_base, l2_base)
  l1_out = []
  for l in l1:
    b = os.path.splitext(os.path.basename(l))[0].split('_')[0]
    if b in matching: l1_out.append(l)
  l2_out = []
  for l in l2:
    b = os.path.splitext(os.path.basename(l))[0].split('_')[0]
    if b in matching: l2_out.append(l)
  return l1_out, l2_out

def color_mask(mask):
  uq = np.unique(mask)
  r = np.zeros(shape=mask.shape, dtype=np.uint8)
  g = np.copy(r)
  b = np.copy(r)
  for u in uq:
    u_m = mask == u
    c = colors[u, :]
    r[mask==u] = c[0]
    g[mask==u] = c[1]
    b[mask==u] = c[2]
  newmask = np.dstack((b,g,r))
  return newmask


def overlay_img(base, pred):
  img = cv2.imread(base)
  ishape = img.shape[:2][::-1]
  y = np.load(pred)
  # y = cv2.imread(pred, 0)
  y = cv2.resize(y, fx=0, fy=0, dsize=ishape, interpolation=cv2.INTER_LINEAR)
  ymax = np.argmax(y, axis=-1)

  ## Find unprocessed space
  ymax[np.sum(y, axis=-1) < 1e-2] = 4 # white

  # Find pure black and white in the img
  gray = np.mean(img, axis=-1)
  img_w = gray > 220
  img_b = gray < 10

  ycolor = color_mask(ymax)
  img = np.add(img*mixture[0], ycolor*mixture[1])
  channels = np.split(img, 3, axis=-1)
  for c in channels:
    c[img_w] = 255
    c[img_b] = 255
  img = np.dstack(channels)
  return cv2.convertScaleAbs(img)


def main(args):
  baseimgs = sorted(glob.glob('{}/*rgb.jpg'.format(args.s)))
  predictions = sorted(glob.glob('{}/*{}'.format(args.p, args.r)))
  baseimgs, predictions = matching_basenames(baseimgs, predictions)

  for bi, pr in zip(baseimgs, predictions):
    dst = pr.replace(args.r, 'overlay.jpg')
    # if os.path.exists(dst):
    #   print('{} exists'.format(dst))
    #   continue

    combo = overlay_img(bi, pr)
    print('{} --> {}'.format(combo.shape, dst))
    cv2.imwrite(dst, combo)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', default=None, type=str,
    help='Path to some folder with files conforming to <-p>/*<-r> pattern') # path to some predictions
  parser.add_argument('-s', default=None, type=str,
    help='Path to some folder with hi-res images to emblazen with color') # path to some target hi-res images
  parser.add_argument('-r', default='.npy', type=str,
    help='Pattern to match <-p>/*<-r>') # pattern to match
  # parser.add_argument('-d', default='hires_imgs', type=str)

  args = parser.parse_args()
  main(args)
