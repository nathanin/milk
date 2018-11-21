"""
Test classifier in graph mode

"""
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import argparse
import glob
import os

import seaborn as sns
from matplotlib import pyplot as plt

from MulticoreTSNE import TSNE

from milk.classifier import Classifier
from milk.utilities import data_utils
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

sys.path.insert(0, '..')
from dataset import CASE_LABEL_DICT

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tfe.enable_eager_execution(config=config)

# Hard coded into the dataset
class_mnemonics = {
    0: 'GG3',
    1: 'GG4',
    2: 'GG5',
    3: 'BN',
    4: 'ST',
}

CASE_PATT = r'SP_\d+-\d+'
def case_label_fn(data_path):
    case = re.findall(CASE_PATT, data_path)[0]
    y_ = CASE_LABEL_DICT[case]
    # print(data_path, y_)
    return y_

def draw_projection():
    pass

def main(args):
    model = Classifier(n_classes=args.n_classes)
    x_dummy = tf.zeros((1, args.input_dim, args.input_dim, 3))
    z_dummy, y_dummy = model(x_dummy, 
                             verbose=True,
                             return_embedding_and_predict=True)
    print('z: {} y: {}'.format(z_dummy.shape, y_dummy.shape))

    saver = tfe.Saver(model.variables)
    saver.restore(args.snapshot)

    model.summary()

    data_list = glob.glob(args.test_data)
    print('Got {} tile caches'.format(len(data_list)))
    np.random.shuffle(data_list)

    transform_fn = data_utils.make_transform_fn(
        args.img_x, args.img_x, args.input_dim, args.scale)
    
    features = []
    yclassif = []
    ytrue = []
    for test_case in data_list[:args.n_cases]:
        case_x, case_y = data_utils.load(
            data_path = test_case,
            transform_fn=transform_fn,
            const_bag=args.n_imgs,
            case_label_fn=case_label_fn)

        case_z, case_yclassif = model(case_x,
                                      return_embedding_and_predict=True)

        features.append(case_z)
        yclassif.append(case_yclassif)
        ytrue.append(case_y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', 
        default='../dataset/tiles/*.npy', type=str)
        # default='../dataset/gleason_grade_val.tfrecord', type=str)
    parser.add_argument('--snapshot', 
        default='./trained/classifier-19999', type=str)

    ## Number of data caches to use
    parser.add_argument('--n_cases', default=10, type=int)
    ## Number of images per cache
    parser.add_argument('--n_imgs', default=64, type=int)
    ## Resolution in the image cache
    parser.add_argument('--img_x', default=128, type=int)  

    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--input_dim', default=96, type=int)
    parser.add_argument('--scale', default=1., type=float)
    parser.add_argument('--savefig', default=None, type=str)

    args = parser.parse_args()

    main(args)
    
