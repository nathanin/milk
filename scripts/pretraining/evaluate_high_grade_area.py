import numpy as np
import argparse
import pickle
import glob
import cv2
import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

def draw(y, x, xlabel='area', ylabel='count', title='Area', saveto='area.png'):
  plt.clf()
  for c in np.unique(y):
    _ = plt.hist(x[y == c], 
                 bins=20, 
                 alpha=0.5,
                 label='M{}'.format(c))

  plt.title(title, fontsize=8)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.savefig(saveto, bbox_inches='tight')

def main(args):
  caselist = pickle.load(open('../dataset/uid2slide.pkl', 'rb'))
  casedict = pickle.load(open('../dataset/case_dict_obfuscated.pkl', 'rb'))
  ys, ratios, high_burden = [], [], []

  for k, v in caselist.items():
    y = casedict[k]
    if y == 2:
      continue

    areas = []
    fgareas = []
    for nl in v:
      basename = os.path.basename(nl).replace('.svs', '')
      npyf = os.path.join(args.srcdir, '{}.npy'.format(basename))
      fgf = os.path.join(args.fgdir, '{}_fg.png'.format(basename))

      if not os.path.exists(npyf):
        continue
      if not os.path.exists(fgf):
        continue

      x = np.load(npyf)
      h, w = x.shape[:2]
      hg_area = np.argmax(x, -1)

      fg_area = cv2.imread(fgf, -1)
      fg_area = cv2.resize(fg_area, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

      areas.append((hg_area == 1).sum())
      fgareas.append((fg_area > 0).sum())

    if len(areas) > 0:
      high_area = areas[np.argmax(areas)]
      high_fgarea = fgareas[np.argmax(areas)]

      r = high_area/high_fgarea
      ratios.append(r)
      ys.append(y)
      high_burden.append(high_area)
      print('Case', k, y, r, high_area, high_fgarea, areas)

  ys = np.array(ys)
  ratios = np.array(ratios)
  high_burden = np.log(np.array(high_burden))

  fig = plt.figure(figsize=(2,2), dpi=300)
  auc_ = roc_auc_score(y_true=ys, y_score=ratios / ratios.max())
  draw(ys, ratios, xlabel='High grade fraction', 
       title='High grade tumor area fraction\nAUC={:3.4f}'.format(auc_), saveto='high_grade_ratios.png')
  auc_ = roc_auc_score(y_true=ys, y_score=high_burden / high_burden.max())
  draw(ys, high_burden, 
       title='High grade tumor area\nAUC={:3.4f}'.format(auc_), saveto='high_grade_area.png')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--srcdir', default='wsi', type=str)
  parser.add_argument('--fgdir',  default='../usable_area/inference', type=str)

  args = parser.parse_args()
  main(args)