import numpy as np
import pandas as pd
import shutil
import glob
import os

from matplotlib import pyplot as plt
import argparse

def read_log(filename, sampling=100):
  iterations, train_loss, train_acc = [], [], []
  sm_loss, sm_acc = [], []
  with open(filename, 'r') as f:
    for i, entry in enumerate(f):
      entries = entry.split('\t')
      sm_loss.append(float(entries[1]))
      sm_acc.append(float(entries[2]))

      if i % sampling == 0:
        sm_loss = np.mean(sm_loss)
        sm_acc = np.mean(sm_acc)

        iterations.append(int(entries[0]))
        train_loss.append(sm_loss)
        train_acc.append(sm_acc)

        sm_loss, sm_acc = [], []

  iterations = np.array(iterations)
  train_loss = np.array(train_loss)
  train_acc = np.array(train_acc)
  return iterations, train_loss, train_acc

def list_logs(dirname):
  filelist = glob.glob(os.path.join(dirname, '*_training_curves.txt'))
  return filelist

def draw_curves(iterations, train_loss, train_acc, label, ax):
  ax.plot(iterations, train_loss, lw=1, ls='-' ) 
  ax.plot(iterations, train_acc,  lw=1, ls='--') 

def main(args):
  logfiles = list_logs(args.savedir)

  fig = plt.figure(figsize=(3,2), dpi=300)
  ax = plt.gca()
  for filename in logfiles:
    basename = os.path.basename(filename).replace('_training_curves.txt', '')
    iterations, train_loss, train_acc = read_log(filename)
    draw_curves(iterations, train_loss, train_acc, label=basename, ax=ax)

  plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--savedir', default = '../experiment/save')

  args = parser.parse_args()
  main(args)
