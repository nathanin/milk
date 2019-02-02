import tensorflow as tf
import numpy as np
import MulticoreTSNE as TSNE
import seaborn as sns
import matplotlib.pyplot as plt

import os
import glob
import argparse

from milk import Milk, data_utils
import sys
sys.path.insert(0, '../experiment')
from encoder_config import deep_encoder_args

import pickle
case_dict = pickle.load(open('../dataset/case_dict_obfuscated.pkl', 'rb'))

def read_test_files(test_list):
  test_files = []
  with open(test_list) as f:
    for L in f:
      test_files.append(L.replace('\n', ''))

  return test_files

def gen_data(test_list, bag_size=10):
  transform_fn = data_utils.make_transform_fn( 128, 128, 96, normalize=True )
  test_files = read_test_files(test_list)
  data = {}
  for pth in test_files:
    x = np.load(pth)
    n_x = x.shape[0]
    x = x[np.random.choice(range(n_x), min(n_x, bag_size)), ...]
    x = np.expand_dims(np.stack([transform_fn(x_) for x_ in x], axis=0), 0)

    print(pth, x.shape)
    data[pth] = x

  return data

def get_features(model, data):
  features = {}
  for key, val in data.items():
    casename = os.path.basename(key).replace('.npy', '')
    y = case_dict[casename]
    
    z = model(val)
    print(casename, y, z.shape)
    features[casename] = z

  return features

def project_features(features):
  print('projecting')
  f = []
  ys = []
  for k, v in features.items():
    y = case_dict[k]
    print(k, y, v.shape)
    f.append(v)
    ys += [y] * v.shape[0]

  ys = np.array(ys)
  fcat = np.concatenate(f, axis=0)
  print('fcat: ', fcat.shape)
  print('ys: ', ys.shape)

  proj = TSNE.MulticoreTSNE(n_jobs=-1).fit_transform(fcat)
  print('proj: ', proj.shape)
  return proj, ys

def draw_projection(proj, ys, attentions=None, dst=None):
  plt.clf()
  for c in np.unique(ys):
    if attentions is not None:
      plt.scatter(proj[ys == c, 0] , proj[ys == c, 1], label='{}'.format(c),
        c = attentions[ys == c], s=2)
    else:
      plt.scatter(proj[ys == c, 0] , proj[ys == c, 1], label='{}'.format(c), s=2)

  plt.legend(loc=2)
  plt.savefig(dst, bbox_inches='tight')

def main(args):
  snapshot = os.path.join(args.exphome, 'save', '{}.h5'.format(args.timestamp))
  assert os.path.exists(snapshot)

  model = Milk(input_shape = (args.bag_size, args.x_size, args.x_size, 3),
               encoder_args = deep_encoder_args,
               use_gate = True,
               mode = args.mil,
               deep_classifier = True)
  #model.load_weights(snapshot, by_name=True)
  model.summary()
  enc_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('encoder_glob_pool').output)
  enc_model.load_weights(snapshot, by_name=True)
  deep_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('deep_4').output)
  deep_model.load_weights(snapshot, by_name=True)
  cls_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('mil_dense_2').output)
  cls_model.load_weights(snapshot, by_name=True)
  att_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('att_2').output)
  att_model.load_weights(snapshot, by_name=True)
  
  data = gen_data(os.path.join(args.exphome, 'test_lists', '{}.txt'.format(args.timestamp)), bag_size=args.bag_size)

  fig = plt.figure(figsize=(3,3), dpi=300)

  attentions = get_features(att_model, data)
  attentions = np.squeeze(np.concatenate([v for k,v in attentions.items()], axis=0))
  print('attentions: ', attentions.shape)

  enc_features = get_features(enc_model, data)
  enc_tsne, ys = project_features(enc_features)
  draw_projection(enc_tsne, ys, attentions, dst='/home/nathan/Dropbox/projects/milk/encoder_tsne.png')

  deep_features = get_features(deep_model, data)
  deep_tsne, ys = project_features(deep_features)
  draw_projection(deep_tsne, ys, attentions, dst='/home/nathan/Dropbox/projects/milk/deep_tsne.png')

  cls_features = get_features(cls_model, data)
  cls_tsne, ys = project_features(cls_features)
  draw_projection(cls_tsne, ys, attentions=None, dst='/home/nathan/Dropbox/projects/milk/classifier_tsne.png')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--timestamp', default=None, type=str) # Required
  parser.add_argument('--exphome', default='../experiment', type=str) 
  parser.add_argument('--dst', default=None, type=str) 

  parser.add_argument('--mil', default='attention', type=str) 
  parser.add_argument('--bag_size', default=100, type=int) 
  parser.add_argument('--x_size', default=96, type=int) 

  args = parser.parse_args()

  tf.enable_eager_execution()
  main(args)

