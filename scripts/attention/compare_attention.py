import numpy as np
import argparse
import glob
import os

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def get_files(src, tree_src):
  fnames = glob.glob(os.path.join(src, '*_att.npy'))
  # fbases = [os.path.splitext(os.path.basename(x))[0] for x in fnames]
  fbases = [os.path.basename(x) for x in fnames]

  path_bases = glob.glob(os.path.join(tree_src, '*'))
  fpaths = {}
  for fbase in fbases:
    fbp = []
    for pb in path_bases:
      fn = os.path.join(pb, fbase)
      if os.path.exists(fn):
        fbp.append(fn)

    if len(fbp) >= 3:
      fpaths[fbase] = fbp

  return fpaths

def main(args):
  fpath_dict = get_files(args.src, args.tree_src)

  fig = plt.figure(figsize=(2,2), dpi=180)
  for fbase, fpaths in fpath_dict.items():
    print(fbase)
    xlist = [np.load(x) for x in fpaths]
    xmean = np.mean(xlist, axis=0)
    xstdv = np.std(xlist, axis=0)

    print(xmean.shape, xstdv.shape)
    plt.clf()
    oname = os.path.join(args.odir, '{}'.format(fbase.replace('.npy', '.png')))
    xmax = 0
    for x in xlist:
      # plt.hist(x, bins=50, density=True)
      plt.scatter(np.arange(len(x)), x, s=2, alpha=0.2)
      xmax = max(xmax, x.max())
    
    plt.ylim([0, xmax])
    plt.title(fbase)
    plt.savefig(oname, bbox_inches='tight')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('src')
  parser.add_argument('--odir', default='./attention_histograms')
  parser.add_argument('--tree_src', default='./attention_images')

  args = parser.parse_args()
  main(args)