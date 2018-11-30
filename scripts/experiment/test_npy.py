import tensorflow as tf
import tensorflow.contrib.eager as tfe
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
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from milk.utilities import data_utils
from milk.utilities import model_utils
from milk.utilities import training_utils
from milk import Milk

sys.path.insert(0, '..')
from dataset import CASE_LABEL_DICT

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tfe.enable_eager_execution(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X_SIZE = 128
Y_SIZE = 128
CROP_SIZE = 96
SCALE = 1.0
MIN_BAG = 150
MAX_BAG = 350
CONST_BAG = 200

CASE_PATT = r'SP_\d+-\d+'
def case_label_fn(data_path):
    case = re.findall(CASE_PATT, data_path)[0]
    y_ = CASE_LABEL_DICT[case]
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


def run_sample(case_x, model, mcdropout=None):
    case_x = tf.constant(case_x)
    if mcdropout is not None:
        yhats = []
        for _ in range(mcdropout):
            yhats.append(  model(case_x, training=True) )
        
        yhats = np.stack(yhats, axis=0)
        yhat = np.mean(yhats, axis=0)

    else:
        yhat = model(case_x, training=False)

    yhat = tf.nn.softmax(yhat)
    print('Returning: {}'.format(yhat))

    return yhat


transform_fn = data_utils.make_transform_fn(X_SIZE, Y_SIZE, CROP_SIZE, SCALE)
def main(args):
    model = Milk()
    x_dummy = tf.zeros(shape=[MIN_BAG, CROP_SIZE, CROP_SIZE, 3], 
                        dtype=tf.float32)
    yhat_dummy = model(x_dummy, verbose=True)
    saver = tfe.Saver(model.variables)

    if args.snapshot is None:
        print('Pulling most recent snapshot from {}'.format(args.snapshot_dir))
        snapshot = tf.train.latest_checkpoint(args.snapshot_dir)
    elif args.snapshot:
        snapshot = args.snapshot
    else:
        print('Must supply either --snapshot or --snapshot_dir')

    print('Restoring snapshot {}'.format(snapshot))
    saver.restore(snapshot)
    model.summary()

    test_list = read_test_list(args.test_list)

    ytrues = []
    yhats = []
    print('MC Dropout: {}'.format(args.mcdropout))
    for _ in range(args.n_repeat):
        for test_case in test_list:
            case_x, case_y = data_utils.load(
                data_path = test_case,
                transform_fn=transform_fn,
                all_tiles=True,
                # const_bag=200,
                min_bag=MIN_BAG,
                max_bag=MAX_BAG,
                case_label_fn=case_label_fn)

            yhat = run_sample(case_x, model, mcdropout=args.mcdropout)
            ytrues.append(case_y)
            yhats.append(yhat)
            print(test_case, case_y, case_x.shape, yhat.numpy())

    ytrue = np.concatenate(ytrues, axis=0)
    yhat = np.concatenate(yhats, axis=0)

    ytrue_max = np.argmax(ytrue, axis=-1)
    yhat_max = np.argmax(yhat, axis=-1)
    accuracy = (ytrue_max == yhat_max).mean()
    print('Accuracy: {:3.3f}'.format(accuracy))

    auc_curve(ytrue, yhat, savepath=args.savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default=None, type=str)
    parser.add_argument('--snapshot_dir', default=None, type=str)
    parser.add_argument('--timestamp', default=None, type=str)
    parser.add_argument('--test_list', type=str)
    parser.add_argument('--n_repeat', default=1, type=int)
    parser.add_argument('--mcdropout', default=None, type=int)
    parser.add_argument('--savepath', default=None, type=str)

    args = parser.parse_args()
    with tf.device('/gpu:0'):
        main(args)