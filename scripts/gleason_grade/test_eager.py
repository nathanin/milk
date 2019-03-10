"""
Test classifier in graph mode

"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import os

import seaborn as sns
from matplotlib import pyplot as plt

from milk.utilities import ClassificationDataset
from milk.eager import ClassifierEager

from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import MulticoreTSNE as TSNE
from milk.encoder_config import get_encoder_args

# Hard coded into the dataset
class_mnemonics = {
  0: 'Low Grade',
  1: 'High Grade',
  2: 'GG5',
  3: 'Benign',
  4: 'Stroma',
}

colors = np.array([[175, 33, 8],
           [20, 145, 4],
           [177, 11, 237],
           [14, 187, 235],
           [0,0,0]
          ]) / 255.

def draw_projection(z, yclassif, savepath=None):
  proj = TSNE.MulticoreTSNE(n_jobs=-1).fit_transform(z)
  fig = plt.figure(figsize=(3,3), dpi=300)

  for c in range(5):
    if (yclassif == c).sum() == 0:
      continue
    
    if c == 2:
      continue

    proj_c = proj[yclassif == c, :]
    plt.scatter(proj_c[:,0], proj_c[:,1], 
      s = 3,
      alpha = 0.5,
      c = colors[c,:],
      label='{}'.format(class_mnemonics[c]))
  # plt.legend(frameon=True)
  plt.xticks([])
  plt.yticks([])

  if savepath is None:
    plt.show()
  else:
    plt.savefig(savepath, bbox_inches='tight')

def auc_curves(ytrue, yhat, n_classes=5, savepath=None):
  plt.figure(figsize=(3,3), dpi=300)
  for c in range(n_classes):
    if c == 2:
      continue
      
    ytrue_c = ytrue[:,c]
    yhat_c = yhat[:,c]
    auc_c = roc_auc_score(y_true=ytrue_c, y_score=yhat_c)
    fpr, tpr, _ = roc_curve(y_true=ytrue_c, y_score=yhat_c)

    plt.plot(fpr, tpr, 
      lw=2.5,
      c=colors[c,:],
      label='{} AUC={:3.3f}'.format(class_mnemonics[c], auc_c))
    
  plt.legend(frameon=True)
  plt.ylabel('Sensitivity')
  plt.xlabel('1 - specificity')

  if savepath is None:
    plt.show()
  else:
    plt.savefig(savepath, bbox_inches='tight')

def main(args):

  crop_size = int(args.input_dim / args.downsample)

  dataset = ClassificationDataset(
    record_path     = args.test_data,
    crop_size       = crop_size,
    downsample      = args.downsample,
    n_classes       = args.n_classes,
    batch           = args.batch_size,
    prefetch_buffer = args.prefetch_buffer,
    repeats         = args.repeats,
    eager           = True)

  batchx, batchy = next(dataset.iterator)
  print('Test batch:')
  print('batchx: ', batchx.get_shape())
  print('batchy: ', batchy.get_shape())

  if args.load_model:
    model = load_model(args.snapshot, compile=False)
  else:
    encoder_args = get_encoder_args(args.encoder)
    model = ClassifierEager(encoder_args=encoder_args,
                            n_classes=args.n_classes,)
    yhat = model(batchx, training=False, verbose=True)
    model.summary()
    model.load_weights(args.snapshot)

  # Loop:
  ytrue_vector, yhat_vector, features = [], [], []
  counter = 0

  for batchx, batchy in dataset.iterator:
    counter += 1
    # batchx, batchy = next(dataset.iterator)

    yhat_, feat_ = model(batchx, return_features=True, training=False, verbose=True)

    ytrue_vector.append(batchy)
    yhat_vector.append(yhat_)
    features.append(feat_)

    if counter % 10 == 0:
      print(counter, 'ytrue:', batchy.shape, 'yhat:', yhat_.shape)
      print(np.argmax(yhat_, axis=-1))
      print(np.argmax(batchy, axis=-1))

    # except tf.errors.OutOfRangeError:
    #   break

  features = np.concatenate(features, axis=0)
  ytrue_vector = np.concatenate(ytrue_vector, axis=0)
  yhat_vector = np.concatenate(yhat_vector, axis=0)
  print('features: ', features.shape)
  print('ytrues: ', ytrue_vector.shape)
  print('yhats: ', yhat_vector.shape)

  ytrue_max = np.argmax(ytrue_vector, axis=-1)
  yhat_max = np.argmax(yhat_vector, axis=-1)

  ytrue_max[ytrue_max == 2] = 1
  yhat_max[yhat_max == 2] = 1

  accuracy = np.mean(ytrue_max == yhat_max)
  print('Accuracy: {:3.3f}'.format(accuracy))
  print(classification_report(y_true=ytrue_max, y_pred=yhat_max))
  auc_curves(ytrue_vector, yhat_vector, savepath=args.save)

  draw_projection(features, yhat_max, savepath=args.saveproj)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_data', default='../dataset/gleason_grade_val_ext.tfrecord', type=str)
  parser.add_argument('--snapshot', default='./gleason_classifier_deep.h5', type=str)
  parser.add_argument('--n_classes', default=5, type=int)
  parser.add_argument('--input_dim', default=96, type=int)
  parser.add_argument('--downsample', default=0.25, type=float)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--repeats', default=3, type=int)
  parser.add_argument('--prefetch_buffer', default=2048, type=int)
  parser.add_argument('--save', default=None, type=str)
  parser.add_argument('--saveproj', default=None, type=str)
  parser.add_argument('--encoder', default='big', type=str)
  parser.add_argument('--load_model', default=False, action='store_true')

  args = parser.parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  tf.enable_eager_execution(config=config)

  main(args)
  
