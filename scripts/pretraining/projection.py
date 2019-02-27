"""
Test classifier in graph mode

"""
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import argparse
import pickle
import glob
import sys
import re
import os

import seaborn as sns
from matplotlib import pyplot as plt
sns.set(style='whitegrid')
plt.style.use('seaborn-whitegrid')

import MulticoreTSNE as TSNE

# from milk.classifier import Classifier
from milk.utilities import data_utils
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

# sys.path.insert(0, '../dataset')
# from dataset import CASE_LABEL_DICT
case_dict = pickle.load(open('../dataset/cases_md5.pkl', 'rb'))

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tfe.enable_eager_execution(config=config)

# Hard coded into the dataset
class_mnemonics = {
  0: 'GG3',
  1: 'GG4',
  2: 'GG5',
  3: 'BN',
  4: 'ST',
}

def filter_list_by_label(lst):
  lst_out = []
  for l in lst:
    l_base = os.path.basename(l)
    l_base = os.path.splitext(l_base)[0]
    if case_dict[l_base] < 2:
      lst_out.append(l)
  print("Got list length {}; returning list length {}".format(
    len(lst), len(lst_out)
  ))
  return lst_out

CASE_PATT = r'SP_\d+-\d+'
def case_label_fn(data_path):
  # case = re.findall(CASE_PATT, data_path)[0]
  # y_ = CASE_LABEL_DICT[case]
  case = os.path.splitext(os.path.basename(data_path))[0]
  y_ = case_dict[case]
  return y_

def draw_projection(z, yclassif, ytrue, title=None, savepath=None):
  proj = TSNE.MulticoreTSNE(n_jobs=-1).fit_transform(z)
  fig, axs = plt.subplots(1,2, figsize=(8,4))

  yclassif_max = np.argmax(yclassif, axis=-1)
  ax = axs[0]
  for c in range(5):
    if (yclassif_max == c).sum() == 0:
      continue

    proj_c = proj[yclassif_max == c, :]
    ax.scatter(proj_c[:,0], proj_c[:,1], 
      s = 3,
      alpha = 0.5,
      label='{}'.format(class_mnemonics[c]))
  ax.legend(frameon=True)

  ytrue_max = np.argmax(ytrue, axis=-1)
  ax = axs[1]
  for c in range(2):
    proj_c = proj[ytrue_max == c, :]
    ax.scatter(proj_c[:,0], proj_c[:,1], 
      s = 3,
      alpha = 0.5,
      label='M{}'.format(c))
  
  ax.legend(frameon=True)
  if title is not None: plt.suptitle(title)

  if savepath is None:
    plt.show()

  else:
    plt.savefig(savepath, bbox_inches='tight')


def main(args):
  layers = ['encoder_glob_pool', 'deep_4']

  model = tf.keras.models.load_model(args.snapshot)
  model.summary()

  enc_model = tf.keras.models.Model(inputs=model.inputs, 
                                    outputs=model.get_layer('encoder_glob_pool').output)
  deep_model = tf.keras.models.Model(inputs=model.inputs, 
                                     outputs=model.get_layer('deep_4').output)

  data_list = glob.glob(args.test_data)
  data_list = filter_list_by_label(data_list)
  print('Got {} tile caches'.format(len(data_list)))
  if args.n_cases > len(data_list):
    print('Got n_cases={} > {}; forcing replacement.'.format(
      args.n_cases, len(data_list)))
    replace = True
  else:
    replace = args.replace

  choice_cases = np.random.choice(data_list, 
    args.n_cases, replace=replace)

  transform_fn = data_utils.make_transform_fn(
    args.img_x, args.img_x, args.input_dim, args.scale,
    normalize=True)
  
  enc_features, deep_features, yclassif, ytrue = [], [], [], []
  for test_case in choice_cases:
    print('Test case: {}'.format(test_case))
    case_x, case_y = data_utils.load(
      data_path = test_case,
      transform_fn=transform_fn,
      const_bag=args.n_imgs,
      case_label_fn=case_label_fn)
    case_x = np.squeeze(case_x, axis=0)

    enc_z         = enc_model.predict(case_x)
    deep_z        = enc_model.predict(case_x)
    case_yclassif = model.predict(case_x)

    enc_features.append(enc_z)
    deep_features.append(deep_z)
    yclassif.append(case_yclassif)
    for _ in range(args.n_imgs):
      ytrue.append(case_y)

  enc_features = np.concatenate(enc_features, axis=0)
  deep_features = np.concatenate(deep_features, axis=0)
  yclassif = np.concatenate(yclassif, axis=0)
  ytrue = np.concatenate(ytrue, axis=0)
  print('Drawing projections from:')
  print('encoder features: {}'.format(enc_features.shape))
  print('deep features: {}'.format(deep_features.shape))
  print('yclassif: {}'.format(yclassif.shape))
  print('ytrue: {}'.format(ytrue.shape))

  draw_projection(enc_features, yclassif, ytrue, title='Encoder features', savepath='projected_encoder_features.png')
  draw_projection(deep_features, yclassif, ytrue, title='Classifier features', savepath='projected_classifier_features.png')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--snapshot', default='gleason_classifier_deep.h5', type=str)
  parser.add_argument('--test_data', default='../dataset/tiles_reduced/*.npy', type=str)

  parser.add_argument('--save', default=None, type=str)
  parser.add_argument('--img_x', default=128, type=int)  # We need to know original image dims 
  parser.add_argument('--scale', default=1.,  type=float)
  parser.add_argument('--n_imgs', default=64, type=int)
  parser.add_argument('--n_cases', default=250, type=int)
  parser.add_argument('--replace', default=False, action='store_true')  

  parser.add_argument('--n_classes', default=5, type=int)
  parser.add_argument('--input_dim', default=96, type=int)

  args = parser.parse_args()

  main(args)
  
