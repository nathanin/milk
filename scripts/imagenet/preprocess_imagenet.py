import numpy as np
import argparse
import shutil
import glob
import cv2
import os

def main(args):
  # train
  train_dirs = glob.glob(os.path.join(args.src, 'train', '*'))
  train_dst = os.path.join(args.src, 'train_list.txt')
  print(len(train_dirs))

  # if os.path.exists(train_dst):
  #   print('{} exists. exiting'.format(train_dst))
  #   return 1

  cldict = {}
  with open(train_dst, 'w+') as f:
    for i, td in enumerate(train_dirs):
      cl = td.split('/')[-1]
      cldict[cl] = i
      # imgs = glob.glob(os.path.join(args.src, 'train', td, '*.JPEG'))
      imgs = glob.glob(os.path.join(td, '*.JPEG'))

      for im in imgs:
        s = '{}\t{}\n'.format(i, im)
        f.write(s)

  # # val
  # val_imgs = glob.glob(os.path.join(args.src, 'val', '*.JPEG'))
  # val_dst = os.path.join(args.src, 'val_list.txt')
  # print(len(val_imgs))

  # # if os.path.exists(val_dst):
  # #   print('{} exists. exiting'.format(val_dst))
  # #   return 1

  # with open(val_dst, 'w+') as f:
  #   for im in val_imgs:
  #     print(im)
  #     cl = os.path.basename(im)[:9]

  #     i = cldict[cl]
  #     s = '{}\t{}\n'.format(i, im)
  #     f.write(s)

  # # test
  # test_imgs = glob.glob(os.path.join(args.src, 'test', '*.JPEG'))
  # test_dst = os.path.join(args.src, 'test_list.txt')
  # print(len(test_imgs))

  # # if os.path.exists(test_dst):
  # #   print('{} exists. exiting'.format(test_dst))
  # #   return 1

  # with open(test_dst, 'w+') as f:
  #   for im in test_imgs:
  #     cl = os.path.basename(im)[:9]

  #     i = cldict[cl]
  #     s = '{}\t{}\n'.format(i, im)
  #     f.write(s)


if __name__ == '__main__':
  # train_src = '/mnt/linux-data/imagenet/ILSVRC/Data/CLS-LOC/train'
  # train_dst = '/mnt/linux-data/imagenet/ILSVRC/Data/CLS-LOC/train_labels.txt'

  parser = argparse.ArgumentParser()
  parser.add_argument('src')
  args = parser.parse_args()
  main(args)