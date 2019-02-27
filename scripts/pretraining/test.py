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
from milk.classifier import Classifier

from sklearn.metrics import roc_auc_score, roc_curve, classification_report

# Hard coded into the dataset
class_mnemonics = {
  0: 'GG3',
  1: 'GG4',
  2: 'GG5',
  3: 'BN',
  4: 'ST',
}
def auc_curves(ytrue, yhat, n_classes=5, savepath=None):
  plt.figure(figsize=(3,3), dpi=300)
  for c in range(n_classes):
    ytrue_c = ytrue[:,c]
    yhat_c = yhat[:,c]
    auc_c = roc_auc_score(y_true=ytrue_c, y_score=yhat_c)
    fpr, tpr, _ = roc_curve(y_true=ytrue_c, y_score=yhat_c)

    plt.plot(fpr, tpr, 
      label='{} AUC={:3.3f}'.format(class_mnemonics[c], auc_c))
    
  plt.legend(frameon=True)
  plt.ylabel('Sensitivity')
  plt.xlabel('1 - specificity')

  if savepath is None:
    plt.show()
  else:
    plt.savefig(savepath, bbox_inches='tight')

def main(args, sess):

  crop_size = int(args.input_dim / args.downsample)

  dataset = ClassificationDataset(
    record_path   = args.test_data,
    crop_size     = crop_size,
    downsample    = args.downsample,
    n_classes     = args.n_classes,
    batch       = args.batch,
    prefetch_buffer = args.prefetch_buffer,
    eager       = False,
    repeats     = args.repeats)
  sess.run(dataset.iterator.initializer)

  x = dataset.x_op
  ytrue = dataset.y_op

  model = load_model(args.snapshot)
  model.summary()
  sess.run(tf.global_variables_initializer())

  # Loop:
  ytrue_vector = []
  yhat_vector = []
  counter = 0
  while True:
    try:
      counter += 1
      ytrue_, xbatch = sess.run([ytrue, x])
      yhat_ = model.predict_on_batch(xbatch)

      print(yhat_.shape, xbatch.shape, yhat_.shape)

      ytrue_vector.append(ytrue_)
      yhat_vector.append(yhat_)

      if counter % 100 == 0:
        print(counter, 'ytrue:', ytrue_.shape, 'yhat:', yhat_.shape)

    except tf.errors.OutOfRangeError:
      break

  ytrue_vector = np.concatenate(ytrue_vector, axis=0)
  yhat_vector = np.concatenate(yhat_vector, axis=0)
  print('ytrues: ', ytrue_vector.shape)
  print('yhats: ', yhat_vector.shape)

  ytrue_max = np.argmax(ytrue_vector, axis=-1)
  yhat_max = np.argmax(yhat_vector, axis=-1)

  accuracy = np.mean(ytrue_max == yhat_max)
  print('Accuracy: {:3.3f}'.format(accuracy))

  print(classification_report(y_true=ytrue_max, y_pred=yhat_max))

  auc_curves(ytrue_vector, yhat_vector, savepath=args.save)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_data', default='../dataset/gleason_grade_val_ext.tfrecord', type=str)
  parser.add_argument('--snapshot', default='./gleason_classifier_deep.h5', type=str)
  parser.add_argument('--n_classes', default=5, type=int)
  parser.add_argument('--input_dim', default=96, type=int)
  parser.add_argument('--downsample', default=0.25, type=float)
  parser.add_argument('--batch', default=64, type=int)
  parser.add_argument('--repeats', default=3, type=int)
  parser.add_argument('--prefetch_buffer', default=2048, type=int)
  parser.add_argument('--save', default=None, type=str)

  args = parser.parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  main(args, sess)
  sess.close()
  
