"""
Dump images from the npy caches

Optionally add in a border
"""
import numpy as np
import argparse
import glob
import cv2
import os

def list_caches(src):
  lst = glob.glob('./{}/*.npy'.format(src))
  return lst

def load_cache(pth, n_imgs, size):
  x_cache = np.load(pth, mmap_mode='r')

  idx = np.random.choice(range(x_cache.shape[0]), min(n_imgs, x_cache.shape[0]))
  imgs = [cv2.resize(x, dsize=(size, size)) for x in x_cache[idx,...]]
  return imgs

def main(args):
  caches = list_caches(args.src)

  total = 0
  for cache in caches:
    print(cache)
    imgs = load_cache(cache, args.n_per_cache, args.size)
    for img in imgs:
      fname = os.path.join(args.dst, '{:05d}.jpg'.format(total))
      print(fname)
      cv2.imwrite(fname, img[:,:,::-1])
      total+=1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src', default='./tiles_reduced')
  parser.add_argument('--dst', default='./tile_images')
  parser.add_argument('--size', default=64, type=int)
  parser.add_argument('--border', default=False, action='store_true')
  parser.add_argument('--n_per_cache', default=4, type=int)

  args = parser.parse_args()
  main(args)