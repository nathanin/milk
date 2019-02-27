import numpy as np
import os
import glob
import shutil
import argparse

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

"""
Example usage:

for fl in $( ls runs* ); do echo $rl; python gather_results.py $rl; done

"""

def read_list(runlist):
  timestamps = []
  with open(runlist, 'r') as f:
    for l in f:
      l_ = os.path.basename(l).replace('.txt\n', '')
      timestamps.append(l_)
  return timestamps

def get_yhat(timestamp, srcdir):
  target = os.path.join(srcdir, '{}.npy'.format(timestamp))
  yhat = np.load(target)
  return yhat

def get_ytrue(timestamp, srcdir):
  target = os.path.join(srcdir, '{}_ytrue.npy'.format(timestamp))
  ytrue = np.load(target)
  return ytrue

# https://en.wikipedia.org/wiki/Precision_and_recall
def specificity(y_true, y_pred):
  true_negative = np.sum((y_true == 0) * (y_pred == 0))
  condition_negative = np.sum(y_true == 0)
  return float(true_negative) / condition_negative

def main(args):
  print('\n\nrunlist:', args.runlist)
  timestamps = read_list(args.runlist)
  print(timestamps)  

  accs, aucs, precs, recs, specs, f1s, yhats, ytrues = [], [], [], [], [], [], [], []
  
  for ts in timestamps:
    yhat  = get_yhat(ts, args.src)
    ytrue = get_ytrue(ts, args.src)
    ytrue_argmax = np.argmax(ytrue, axis=-1)
    yhat_argmax = np.argmax(yhat, axis=-1)

    yhat_1 = np.argmax(yhat, axis=-1)
    ytrue_1 = np.argmax(ytrue, axis=-1)

    acc =  np.mean( ( yhat_1 == ytrue_1 ) )
    auc =  roc_auc_score(   y_true = ytrue,   y_score = yhat)
    prec = precision_score( y_true = ytrue_1, y_pred  = yhat_1)
    rec  = recall_score(    y_true = ytrue_1, y_pred  = yhat_1)
    spec = specificity(y_true=ytrue_argmax, y_pred=yhat_argmax)
    f1 = f1_score(y_true=ytrue_argmax, y_pred=yhat_argmax)

    print('acc={:3.3f}\tauc={:3.3f}\tprec={:3.3f}\trec={:3.3f}\tspec={:3.3f}\tf1={:3.3f}'.format(
      acc, auc, prec, rec, spec, f1))

    accs.append(acc)
    aucs.append(auc)
    precs.append(prec)
    recs.append(rec)
    specs.append(spec)
    f1s.append(f1)

    if args.ensemble:
      yhats.append(yhat)
      ytrues.append(ytrue)

  print('----------------------- mean-----------------------')
  acc_mn  = np.mean(accs) 
  auc_mn  = np.mean(aucs) 
  prec_mn = np.mean(precs) 
  rec_mn  = np.mean(recs) 
  spec_mn  = np.mean(specs) 
  f1_mn  = np.mean(f1s) 
  print('acc={:3.3f}\tauc={:3.3f}\tprec={:3.3f}\trec={:3.3f}\tspec={:3.3f}\tf1={:3.3f}'.format(
    acc_mn, auc_mn, prec_mn, rec_mn, spec_mn, f1_mn))

  acc_sd  = np.std(accs) 
  auc_sd  = np.std(aucs) 
  prec_sd = np.std(precs) 
  rec_sd  = np.std(recs) 
  spec_sd  = np.std(specs) 
  f1_sd  = np.std(f1s) 
  print('----------------------- std -----------------------')
  print('acc={:3.3f}\tauc={:3.3f}\tprec={:3.3f}\trec={:3.3f}\tspec={:3.3f}\tf1={:3.3f}'.format(
    acc_sd, auc_sd, prec_sd, rec_sd, spec_sd, f1_sd))

  if args.ensemble:
    print('Ensemble:')
    yhat = np.mean(yhats, axis=0)
    ytrue = ytrues[0]

    yhat_1 = np.argmax(yhat, axis=-1)
    ytrue_1 = np.argmax(ytrue, axis=-1)

    acc =  np.mean( ( yhat_1 == ytrue_1 ) )
    auc =  roc_auc_score(   y_true = ytrue,   y_score = yhat)
    prec = precision_score( y_true = ytrue_1, y_pred  = yhat_1)
    rec  = recall_score(    y_true = ytrue_1, y_pred  = yhat_1)
    spec = specificity(y_true=ytrue_1, y_pred=yhat_1)
    f1   = f1_score(y_true=ytrue_1, y_pred=yhat_1)
    print('acc={:3.3f}\tauc={:3.3f}\tprec={:3.3f}\trec={:3.3f}\tspec={:3.3f}\tf1={:3.3f}'.format(
      acc, auc, prec, rec, spec, f1))

  print('---------------------------------------------------')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('runlist', type=str)
  parser.add_argument('--src', default='result_test', type=str)
  parser.add_argument('--ensemble', default=False, action='store_true')

  args = parser.parse_args()
  assert os.path.exists(args.runlist)
  assert os.path.exists(args.src)

  main(args)

