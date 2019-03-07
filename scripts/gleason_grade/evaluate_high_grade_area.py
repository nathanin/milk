import numpy as np
import argparse
import pickle
import glob
import cv2
import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

def specificity(y_true, y_pred):
  true_negative = np.sum((y_true == 0) * (y_pred == 0))
  condition_negative = np.sum(y_true == 0)
  return float(true_negative) / condition_negative

modelclass = RandomForestClassifier
def modelfit(param, labels):
  ntrain = int(len(labels) * 0.8)
  print('training cases:', ntrain)
  print('testing cases:', len(labels) - ntrain)
  train_idx = np.zeros(len(labels), dtype=np.bool)
  train_idx[np.random.choice(len(labels), ntrain)] = 1

  train_param = param[train_idx]
  train_labels = labels[train_idx]
  test_param = param[np.logical_not(train_idx)]
  test_labels = labels[np.logical_not(train_idx)]

  model = modelclass().fit(train_param, train_labels)
  train_pred = model.predict(train_param)
  train_acc = (train_pred == train_labels).mean()
  print('Logistic regression training accuracy = ', train_acc)

  test_pred = model.predict(test_param)
  test_acc = (test_pred == test_labels).mean()
  print('Logistic regression testing accuracy = ', test_acc)

  cm = confusion_matrix(y_true=test_labels, y_pred=test_pred)
  print(cm)
  print('Specificity:', specificity(y_true=test_labels, y_pred=test_pred))
  print('F1:', f1_score(y_true=test_labels, y_pred=test_pred))
  print(classification_report(y_true=test_labels, y_pred=test_pred))

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
      # high_area = areas[np.argmax(areas)]
      # high_fgarea = fgareas[np.argmax(areas)]

      high_area = np.sum(areas)
      high_fgarea = np.sum(fgareas)

      r = high_area/high_fgarea
      ratios.append(r)
      ys.append(y)
      high_burden.append(high_area)
      print('Case', k, len(v), y, '{:3.3f}'.format(r), high_area, high_fgarea, areas, sep='\t')

  ys = np.array(ys)
  ratios = np.array(ratios).reshape(-1,1)
  high_burden = np.array(high_burden).reshape(-1,1)

  fig = plt.figure(figsize=(2,2), dpi=300)

  saveto = os.path.join(args.savedir, 'high_grade_ratios.png')
  auc_ = roc_auc_score(y_true=ys, y_score=ratios / ratios.max())
  draw(ys, ratios, xlabel='High grade fraction', title='High grade tumor area fraction\nAUC={:3.4f}'.format(auc_), saveto=saveto)

  saveto = os.path.join(args.savedir, 'high_grade_area.png')
  auc_ = roc_auc_score(y_true=ys, y_score=high_burden / high_burden.max())
  draw(ys, high_burden, title='High grade tumor area\nAUC={:3.4f}'.format(auc_), saveto=saveto)

  print('High grade burden classifier:')
  modelfit(high_burden, ys)

  print('High grade ratio classifier:')
  modelfit(ratios, ys)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--srcdir', default='shallow_model/inference', type=str)
  parser.add_argument('--savedir', default='shallow_model', type=str)
  parser.add_argument('--fgdir',  default='../usable_area/inference', type=str)

  args = parser.parse_args()
  main(args)