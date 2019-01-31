import numpy as np
import argparse
import glob
import cv2
import os

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from scipy.stats import wasserstein_distance

def get_files(src, tree_src, srch='*_att.npy'):
  fnames = glob.glob(os.path.join(src, srch))
  # fbases = [os.path.splitext(os.path.basename(x))[0] for x in fnames]
  fbases = [os.path.basename(x) for x in fnames]

  path_bases = glob.glob(os.path.join(tree_src, '*'))
  fpath_dict = {}
  for fbase in fbases:
    fbp = []
    for pb in path_bases:
      fn = os.path.join(pb, fbase)
      if os.path.exists(fn):
        fbp.append(fn)

    if len(fbp) >= 3:
      fpath_dict[fbase] = fbp

  return fpath_dict

def compare_values(fpath_dict, odir):
  fig = plt.figure(figsize=(2,2), dpi=180)
  for fbase, fpaths in fpath_dict.items():
    print(fbase)
    xlist = [np.load(x) for x in fpaths]
    xmean = np.mean(xlist, axis=0)
    xstdv = np.std(xlist, axis=0)

    print(xmean.shape, xstdv.shape)
    plt.clf()
    oname = os.path.join(odir, '{}'.format(fbase.replace('.npy', '.png')))
    xmax = 0
    perm = np.argsort(xlist[0])
    for x in xlist:
      # x -= x.min()
      # x *= (1 / x.max())
      x_sorted = x[perm]
      plt.scatter(np.arange(len(x)), x_sorted, s=2, alpha=0.2)
      xmax = max(xmax, x.max())

      # wd = wasserstein_distance(x, x_0)
      # print('\twd: {}'.format(wd))
    
    plt.ylabel('Scaled attention')
    plt.xlabel('Sorted tiles')
    plt.ylim([0, xmax])
    plt.title(fbase)
    plt.savefig(oname, bbox_inches='tight')

def compare_images(imgpath_dict, odir):
  for fbase, fpaths in imgpath_dict.items():
    xlist = [cv2.imread(x, -1) for x in fpaths]
    xlist = [x for x in xlist if len(x.shape) == 3]
    
    xmean_list = [np.mean(x, axis=-1) for x in xlist]

    try:
      xmean = np.mean(xlist, axis=0)
      xstd = np.std(xmean_list, axis=0)
      xstd *= (255 / xstd.max())
      xstd = 255 - xstd
      xstd = np.dstack([xstd]*3)
    except:
      print('fbase {} shape mismatches'.format(fbase))
      continue

    print(fbase, xmean.shape, xstd.shape)
    xout = np.concatenate([xmean, xstd], axis=0)
    oname = os.path.join(odir, '{}'.format(fbase.replace('.npy', '_img.png')))
    cv2.imwrite(oname, xout)

def main(args):
  print('Comparing values')
  fpath_dict = get_files(args.src, args.tree_src, srch='*_att.npy')
  compare_values(fpath_dict, args.odir)

  print('Getting consensus attention images')
  imgpath_dict = get_files(args.src, args.tree_src, srch='*_img.png')
  compare_images(imgpath_dict, args.odir)


if __name__ == '__main__':
  """
  Sample usage: With a directory `attention_images`:

  attention_images/
  |____ <timestamp 1>/
  |____ <timestamp 2>/
  |____ <timestamp 3>/
  ..

  Give it one of the sub-directories to get candidate names from

  python compare_attention.py ./attention_images/<timestamp 1>

  """
  parser = argparse.ArgumentParser()
  parser.add_argument('src')
  parser.add_argument('--odir', default='./attention_histograms')
  parser.add_argument('--tree_src', default='./attention_images')

  args = parser.parse_args()
  main(args)