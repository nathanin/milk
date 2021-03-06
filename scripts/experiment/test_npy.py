import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import argparse
import datetime
import glob
import time
import sys 
import os 
import re

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, 
  classification_report, confusion_matrix)

from milk.utilities import data_utils
from milk.utilities import model_utils
from milk.utilities import training_utils
from milk import MilkEncode, MilkPredict

# with open('../dataset/case_dict_obfuscated.pkl', 'rb') as f:
#with open('../dataset/cases_md5.pkl', 'rb') as f:
with open('case_dict_obfuscated.pkl', 'rb') as f:
  case_dict = pickle.load(f)

from milk.encoder_config import deep_args as encoder_args

def case_label_fn(data_path):
  case = os.path.splitext(os.path.basename(data_path))[0]
  y_ = case_dict[case]
  # print(data_path, y_)
  return y_

def read_test_list(test_file):
  test_list = []
  with open(test_file, 'r') as f:
    for l in f:
      test_list.append(l.replace('\n', ''))
  
  return test_list

def auc_curve(ytrue, yhat, savepath=None):
  plt.figure(figsize=(2,2), dpi=300)
  ytrue_c = ytrue[:,1]
  yhat_c = yhat[:,1]
  auc_c = roc_auc_score(y_true=ytrue_c, y_score=yhat_c)
  fpr, tpr, _ = roc_curve(y_true=ytrue_c, y_score=yhat_c)

  plt.plot(fpr, tpr, 
      label='M1 AUC={:3.3f}'.format(auc_c))
      
  plt.legend(loc='lower right', frameon=True)
  plt.ylabel('Sensitivity')
  plt.xlabel('1 - specificity')

  if savepath is None:
    plt.show()
  else:
    plt.savefig(savepath, bbox_inches='tight')


""" Return a pretty string """
def pprint_cm(ytrue, yhat, labels=['0', '1']):
  n_classes = len(labels)
  cm = confusion_matrix(y_true=ytrue, y_pred=yhat)
  s = '\nConfusion Matrix:\n'
  s += '\t{}\t{}'.format(*labels)
  for k in range(n_classes):
    s += '\n'
    s += '{}'.format(labels[k])
    for j in range(n_classes):
      s += '\t{: 3d}'.format(cm[k, j])

  s += '\n'
  return s

def test_eval(ytrue, yhat, savepath=None):
  ytrue = ytrue[:,1]
  yhat_c = np.argmax(yhat, axis=-1)
  yhat = yhat[:,1]
  auc_c = 'AUC = {:3.5f}\n'.format(roc_auc_score(y_true=ytrue, y_score=yhat))
  cr = classification_report(y_true=ytrue, y_pred=yhat_c) + '\n'
  cm = pprint_cm(ytrue, yhat_c) + '\n'
  if savepath is not None:
    with open(savepath, 'w+') as f:
        f.write(auc_c)
        f.write(cr)            
        f.write(cm)
  else:
    print(auc_c)
    print(cr)
    print(cm)

def run_sample(case_x, encode_model, predict_model, mcdropout=False, 
             batch_size=64):
  if mcdropout:
    yhats = []
    for _ in range(25):
      zs = encode_model.predict(case_x, batch_size=batch_size)
      yhat = predict_model.predict_on_batch(zs)
      yhats.append(yhat)
    
    yhats = np.stack(yhats, axis=0)
    yhat = np.mean(yhats, axis=0)
  else:
    zs = encode_model.predict(case_x, batch_size=batch_size)
    yhat = predict_model.predict_on_batch(zs)

  return yhat

def main(args):
  transform_fn = data_utils.make_transform_fn(args.x_size, args.y_size, 
                                              args.crop_size, args.scale)

  snapshot = 'save/{}.h5'.format(args.timestamp)
  # trained_model = load_model(snapshot)

  if args.mcdropout:
    encoder_args['mcdropout'] = True

  print('Model initializing')
  encode_model = MilkEncode(input_shape=(args.crop_size, args.crop_size, 3), 
                            encoder_args=encoder_args, deep_classifier=args.deep_classifier)
  encode_shape = list(encode_model.output.shape)
  x_pl = tf.placeholder(shape=(None, args.input_dim, args.input_dim, 3), dtype=tf.float32)
  z_op = encode_model(x_pl)

  input_shape = z_op.shape[-1]
  predict_model = MilkPredict(input_shape=[input_shape], mode=args.mil)

  print('loading weights from {}'.format(snapshot))
  encode_model.load_weights(snapshot, by_name=True)
  predict_model.load_weights(snapshot, by_name=True)

  # models = model_utils.make_inference_functions(encode_model,
  #                                               predict_model,
  #                                               trained_model, 
  #                                               attention_model=None)
  # encode_model, predict_model = models

  test_list = os.path.join(args.testdir, '{}.txt'.format(args.timestamp))
  test_list = read_test_list(test_list)

  ytrues = []
  yhats = []
  print('MC Dropout: {}'.format(args.mcdropout))
  for _ in range(args.n_repeat):
    for test_case in test_list:
      case_x, case_y = data_utils.load(
          data_path = test_case,
          transform_fn=transform_fn,
          case_label_fn=case_label_fn,
          all_tiles=True,
          )
      case_x = np.squeeze(case_x, axis=0)
      print('Running case x: ', case_x.shape)

      yhat = run_sample(case_x, encode_model, predict_model,
                        mcdropout=args.mcdropout,
                        batch_size=args.batch_size)
      ytrues.append(case_y)
      yhats.append(yhat)
      print(test_case, case_y, case_x.shape, yhat)

  ytrue = np.concatenate(ytrues, axis=0)
  yhat = np.concatenate(yhats, axis=0)

  ytrue_max = np.argmax(ytrue, axis=-1)
  yhat_max = np.argmax(yhat, axis=-1)
  accuracy = (ytrue_max == yhat_max).mean()
  print('Accuracy: {:3.3f}'.format(accuracy))

  if args.odir is not None:
    save_img = os.path.join(args.odir, '{}.png'.format(args.timestamp))
    save_metrics = os.path.join(args.odir, '{}.txt'.format(args.timestamp))
    save_yhat = os.path.join(args.odir, '{}.npy'.format(args.timestamp))
    save_ytrue = os.path.join(args.odir, '{}_ytrue.npy'.format(args.timestamp))
    np.save(save_yhat, yhat)
    np.save(save_ytrue, ytrue)
  else:
    save_img = None
    save_metrics = None
    save_yhat = None

  auc_curve(ytrue, yhat, savepath=save_img)
  test_eval(ytrue, yhat, savepath=save_metrics)

if __name__ == '__main__':
  # n_repeat controls re-sampling from the case
  # mcdropout is a flag for doing mcdropout to approximate posterior probability
  parser = argparse.ArgumentParser()
  parser.add_argument('--timestamp', default=None, type=str)
  parser.add_argument('--n_repeat', default=1, type=int)
  parser.add_argument('--mcdropout', default=False, action='store_true')
  parser.add_argument('--testdir', default='test_lists', type=str)
  parser.add_argument('--odir', default=None, type=str)

  parser.add_argument('--x_size', default=128, type=int)
  parser.add_argument('--y_size', default=128, type=int)
  parser.add_argument('--crop_size', default=96, type=int)
  parser.add_argument('--scale', default=1.0, type=float)
  parser.add_argument('--batch_size', default=64, type=int)

  parser.add_argument('--mil', default='attention', type=str)

  args = parser.parse_args()
  assert args.timestamp is not None
  main(args)
