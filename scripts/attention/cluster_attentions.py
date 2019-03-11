"""
Run inference on test cases and generate projections

This script takes full advantage of eager execution
"""
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import argparse
import datetime
import pickle
import shutil
import glob
import time
import sys 
import os 
import re

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
sns.set(style='whitegrid')
plt.style.use('seaborn-whitegrid')

import MulticoreTSNE as TSNE

from milk.utilities import data_utils
from milk.utilities import model_utils
from milk.utilities import training_utils

from milk.eager import MilkEager
from milk.encoder_config import get_encoder_args

with open('../dataset/case_dict_obfuscated.pkl', 'rb') as f:
  case_dict = pickle.load(f)

def read_test_list(test_file):
  test_list = []
  with open(test_file, 'r') as f:
    for l in f:
      test_list.append(l.replace('\n', ''))
  print('Got {} test cases'.format(len(test_list)))
  return test_list

def get_attention_extremes(attention, case_x, n=5):
  attention = np.squeeze(attention)
  att_sorted_idx = np.argsort(attention)
  top_att = np.random.choice(att_sorted_idx[-2*n:], n)
  low_att = np.random.choice(att_sorted_idx[:2*n], n)
  # top_att = np.random.choice(att_sorted_idx, n)
  high_att_idx = np.zeros_like(attention, dtype=np.bool)
  high_att_idx[top_att] = 1
  low_att_idx = np.zeros_like(attention, dtype=np.bool)
  low_att_idx[low_att] = 1
  high_att_imgs = case_x[high_att_idx, ...]
  low_att_imgs = case_x[low_att_idx, ...]
  return high_att_idx, high_att_imgs, low_att_idx, low_att_imgs

grade_dict = {
  0: 'GG3',
  1: 'GG4',
  2: 'GG5',
  3: 'BN',
  4: 'ST',
}
# clusterfn = lambda x: umap.UMAP().fit_transform(x)
clusterfn = lambda x: TSNE.MulticoreTSNE().fit_transform(x)

"""
Plot the features projected down to 2D, with image insets
"""
def cart2pol(x, y):
  rho = np.sqrt(x**2 + y**2)
  phi = np.arctan2(y, x)
  # pl = np.stack([phi, rho], axis=-1)
  return (rho, phi)

def pol2cart(rho, phi):
  x = rho * np.cos(phi)
  y = rho * np.sin(phi)
  return(x, y)

def inset_images(ax, z, att_index, images, edgecolor='k', rho_delta = 10):
  for att_img, att_idx in zip(images, att_index):
    coord_x, = z[att_idx,0]
    coord_y, = z[att_idx,1]
    rho, phi = cart2pol(coord_x, coord_y)
    coord_x_offs, coord_y_offs = pol2cart(rho+rho_delta, phi)
    im = OffsetImage(att_img, zoom=0.5)
    ab = AnnotationBbox(
            offsetbox=im, 
            xy=(coord_x, coord_y),
            xybox=(coord_x_offs, coord_y_offs),
            boxcoords='data',
            pad=0.1,
            bboxprops={'edgecolor': edgecolor},
            arrowprops=dict(arrowstyle='-', color='k'),
            frameon=True)
    ax.add_artist(ab)
  return ax

def draw_projection_with_images(z, attentions, high_attention,
    high_attention_images, low_attention, low_attention_images,
    savepath=None):

  ofs_radius = 10
  # 0-Center columns of z
  for k  in range(2):
    z[:,k] = z[:,k] - np.mean(z[:,k])
  # ax = plt.subplot(figsize=(4,4), dpi=300, projection='polar')
  fig = plt.figure(figsize=(4,4), dpi=300)
  ax = fig.add_subplot(111)
  print('Projection shape', z.shape)
  plt.scatter(z[:-2,0], z[:-2,1], c=attentions, s=10.0, 
    cmap='hot_r',
    # alpha=0.5,
    linewidths=0.5,
    edgecolors='k')
  plt.colorbar()
  plt.scatter(z[-2,0], z[-2,1], s=65.0, 
    c = 'g',
    # alpha=0.5,
    linewidths=0.5,
    edgecolors='k')
  plt.scatter(z[-1,0], z[-1,1], s=65.0, 
    c = 'b',
    # alpha=0.5,
    linewidths=0.5,
    edgecolors='k')
  plt.xticks([])
  plt.yticks([])
  xmin, xmax = plt.xlim()
  plt.xlim([xmin-ofs_radius, xmax+ofs_radius])
  ymin, ymax = plt.ylim()
  plt.ylim([ymin-ofs_radius, ymax+ofs_radius])
  att_index = np.argwhere(high_attention)
  ax = inset_images(ax, z, att_index, high_attention_images, 
    edgecolor='k', rho_delta=ofs_radius)
  att_index = np.argwhere(low_attention)
  ax = inset_images(ax, z, att_index, low_attention_images, 
    edgecolor='k', rho_delta=ofs_radius)
  if savepath is None:
    plt.show()
  else:
    # plt.savefig(savepath, bbox_inches='tight')
    plt.savefig(savepath)
  plt.close()

"""
Plot the features projected down to 2D
"""
def draw_projection(features, features_avg, features_att, attentions, savepath=None):
  plt.figure(figsize=(3,3), dpi=300)
  features = np.concatenate([features, features_avg, features_att], axis=0)
  z = clusterfn(features)
  plt.scatter(z[:-2,0], z[:-2,1], c=attentions, s=15.0, 
    cmap='hot_r',
    linewidths=0.5,
    edgecolors='k')
  plt.colorbar()
  plt.scatter(z[-2,0], z[-2,1], s=65.0, 
    linewidths=0.5,
    c = 'g',
    edgecolors='k')
  plt.scatter(z[-1,0], z[-1,1], s=65.0, 
    linewidths=0.5,
    c = 'b',
    edgecolors='k')
  plt.xticks([])
  plt.yticks([])
  if savepath is None:
    plt.show()
  else:
    # plt.savefig(savepath, bbox_inches='tight')
    plt.savefig(savepath)
  plt.close()
  return z 

"""
Plot the features, after MIL layer i.e. one vector/case
color / mark according to label / correct/incorrect
"""
color = ['b', 'r']
def draw_class_projection(features, ytrue, yhat, savepath=None):
  plt.figure(figsize=(3,3), dpi=180)
  z = clusterfn(features)
  ytrue = np.argmax(ytrue, axis=-1)
  yhat = np.argmax(yhat, axis=-1)
  correct = ytrue == yhat
  incorrect = ytrue != yhat
  for c in range(2):
    cidx = ytrue == c
    z_c = z[cidx, :]
    print('z_c', z_c.shape)
    z_cor = z_c[correct[cidx], :]
    z_inc = z_c[incorrect[cidx], :]
    print('z_cor', z_cor.shape)
    print('z_inc', z_inc.shape)
    plt.scatter(z_cor[:,0], z_cor[:,1], c=color[c], 
      s=5.0, marker='o',
      label='M{} Correct'.format(c))
    plt.scatter(z_inc[:,0], z_inc[:,1], c=color[c], 
      s=5.0, marker='x', alpha=0.5,
      label='M{} Incorrect'.format(c))
  plt.legend(frameon=True, fontsize=8)
  plt.xticks([])
  plt.yticks([])
  plt.title('Case Projections')
  if savepath is None:
    plt.show()
  else:
    plt.savefig(savepath, bbox_inches='tight')
  plt.close()

def main(args):
  transform_fn = data_utils.make_transform_fn(128, 128, args.input_dim, 1.0, normalize=True)

  snapshot = os.path.join(args.snapshot_dir, '{}.h5'.format(args.timestamp))
  test_list = os.path.join(args.test_list_dir, '{}.txt'.format(args.timestamp))

  encoder_args = get_encoder_args(args.encoder)
  model = MilkEager(encoder_args=encoder_args, 
                    deep_classifier=True, 
                    batch_size=args.batch_size,
                    temperature=args.temperature,
                    cls_normalize=args.cls_normalize)

  x_dummy = tf.zeros(shape=[1, args.batch_size, args.input_dim, args.input_dim, 3], dtype=tf.float32)
  retvals = model(x_dummy, verbose=True)
  model.load_weights(snapshot, by_name=True)
  model.summary()

  test_list = read_test_list(test_list)
  savebase = os.path.join(args.odir, args.timestamp)
  if os.path.exists(savebase):
    shutil.rmtree(savebase)
  os.makedirs(savebase)

  yhats, ytrues = [], []
  features_case, features_classifier = [], []
  for test_case in test_list:
    case_name = os.path.basename(test_case).replace('.npy', '')
    print(test_case, case_name)
    # case_path = os.path.join('../dataset/tiles_reduced', '{}.npy'.format(case_name))
    case_x = np.load(test_case)
    case_x = np.stack([transform_fn(x) for x in case_x], 0)
    ytrue = case_dict[case_name]
    print(case_x.shape, ytrue)
    ytrues.append(ytrue)

    if args.sample:
      case_x = case_x[np.random.choice(range(case_x.shape[0]), args.sample), ...]
      print(case_x.shape)

    features = model.encode_bag(case_x, training=True, return_z=True)
    print('features:', features.shape)
    features_att, attention = model.mil_attention(features, return_att=True, training=False)
    print('features:', features_att.shape, 'attention:', attention.shape)

    features_avg = np.mean(features, axis=0, keepdims=True)

    yhat_instances = model.apply_classifier(features, training=False)
    print('yhat instances:', yhat_instances.shape)
    yhat = model.apply_classifier(features_att, training=False)
    print('yhat:', yhat.shape)
    yhats.append(yhat)

    # yhat, attention, features, feat_case, feat_class = retvals
    # attention = np.squeeze(attention.numpy(), axis=0)
    high_att_idx, high_att_imgs, low_att_idx, low_att_imgs = get_attention_extremes(
      attention, case_x, n = 5)

    print('Case {}: predicted={} ({})'.format( test_case, np.argmax(yhat, axis=-1) , yhat.numpy()))

    attention = np.squeeze(attention.numpy())
    features = features.numpy()
    savepath = os.path.join(savebase, '{}_{:3.2f}.png'.format(case_name, yhat[0,1]))
    print('Saving figure {}'.format(savepath))
    z = draw_projection(features, features_avg, features_att, attention, savepath=savepath)

    # savepath = os.path.join(savebase, '{}_{:3.2f}_ys.png'.format(case_name, yhat[0,1]))
    # print('Saving figure {}'.format(savepath))
    # draw_projection_with_images(z, yhat_instances[:,1].numpy(), 
    #   high_att_idx, high_att_imgs, 
    #   low_att_idx, low_att_imgs, 
    #   savepath=savepath)

    savepath = os.path.join(savebase, '{}_{:3.2f}_imgs.png'.format(case_name, yhat[0,1]))
    print('Saving figure {}'.format(savepath))
    draw_projection_with_images(z, attention, 
      high_att_idx, high_att_imgs, 
      low_att_idx, low_att_imgs, 
      savepath=savepath)

    savepath = os.path.join(savebase, '{}_atns.npy'.format(case_name))
    np.save(savepath, attention)
    savepath = os.path.join(savebase, '{}_feat.npy'.format(case_name))
    np.save(savepath, features)

  yhats = np.concatenate(yhats, axis=0)
  yhats = np.argmax(yhats, axis=1)
  ytrues = np.array(ytrues)
  acc = (yhats == ytrues).mean()
  print(acc)
  cm = confusion_matrix(y_true=ytrues, y_pred=yhats)
  print(cm)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--timestamp', type=str)
  parser.add_argument('--encoder', default='big', type=str)
  parser.add_argument('--temperature', default=1., type=float)
  parser.add_argument('--cls_normalize', default=True, type=bool)

  parser.add_argument('--snapshot_dir', default='../experiment/save', type=str)
  parser.add_argument('--test_list_dir', default='../experiment/test_lists', type=str)

  parser.add_argument('--repeats', default=1, type=int)
  parser.add_argument('--feature_base', default='features/milk', type=str)
  parser.add_argument('--odir', default='projection', type=str)

  parser.add_argument('--input_dim', default=96, type=int)
  parser.add_argument('--batch_size', default=32, type=int)
  parser.add_argument('--sample', default=0, type=int)

  args = parser.parse_args()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  tf.enable_eager_execution(config=config)
  main(args)
