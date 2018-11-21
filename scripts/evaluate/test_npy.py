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

transform_fn = data_utils.make_transform_fn(X_SIZE, Y_SIZE, CROP_SIZE, SCALE)

def main(args):
    model = Milk()
    x_dummy = tf.zeros(shape=[MIN_BAG, CROP_SIZE, CROP_SIZE, 3], 
                        dtype=tf.float32)
    yhat_dummy = model(x_dummy, verbose=True)
    saver = tfe.Saver(model.variables)
    saver.restore(args.snapshot)

    model.summary()

    test_list = []
    with open(args.test_list, 'r') as f:
        for l in f:
            test_list.append(l.replace('\n', ''))

    for test_case in test_list:
        case_x, case_y = data_utils.load(
            data_path = test_case,
            transform_fn=transform_fn,
            min_bag=MIN_BAG,
            max_bag=MAX_BAG,
            case_label_fn=case_label_fn)
        yhat = model(tf.constant(case_x))
        print(test_case, case_y, case_x.shape, yhat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str)
    parser.add_argument('--test_list', type=str)

    args = parser.parse_args()
    with tf.device('/gpu:0'):
        main(args)