"""
Run inference on test cases and generate projections
"""
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
sns.set(style='whitegrid')
plt.style.use('seaborn-whitegrid')

import MulticoreTSNE as TSNE

from milk.utilities import data_utils
from milk.utilities import model_utils
from milk.utilities import training_utils
from milk import Milk
from milk.classifier import Classifier

sys.path.insert(0, '..')
from dataset import CASE_LABEL_DICT

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

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

grade_dict = {
    0: 'GG3',
    1: 'GG4',
    2: 'GG5',
    3: 'BN',
    4: 'ST',
}
def draw_projection(features, attentions, z_case, y_case, savepath=None):
    fig, axs = plt.subplots(1, 2, figsize=(4,2), dpi=300)
    # z = TSNE.MulticoreTSNE(n_jobs=-1).fit_transform(features)
    z = TSNE.MulticoreTSNE(n_jobs=-1).fit_transform(z_case)

    ax = axs[0]
    ax.set_title('Trained Encoder Attention', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(z[:,0], z[:,1], c=attentions, s=2.0, cmap='hot_r')

    ax = axs[1]
    ax.set_title('Pretrained Classifier', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    y_case = np.argmax(y_case, axis=-1)
    for c in range(5):
        cidx = y_case == c
        ax.scatter(z[cidx,0], z[cidx,1], 
            s=2.0,
            alpha=0.5,
            label='{}'.format(grade_dict[c]))
    ax.legend(frameon=True, fontsize=8, loc=4)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()

transform_fn = data_utils.make_transform_fn(X_SIZE, Y_SIZE, CROP_SIZE, SCALE,
    normalize=True)
def main(args):
    model = Classifier(n_classes=5)
    x_dummy = tf.zeros(shape=[32, CROP_SIZE, CROP_SIZE, 3], 
                        dtype=tf.float32)
    retvals = model(x_dummy, verbose=True, return_embedding_and_predict=True)

    saver = tfe.Saver(model.variables)
    saver.restore(args.classifier_snapshot)
    model.summary()

    test_list = read_test_list(args.test_list)

    features_all = []
    classifier_prediction = []
    attentions_all = []
    for test_case in test_list:
        case_name = os.path.basename(test_case).replace('.npy', '')
        case_x, case_y = data_utils.load(
            data_path = test_case,
            transform_fn=transform_fn,
            all_tiles=True,
            case_label_fn=case_label_fn)

        z_case = []
        y_case = []
        case_split = np.array_split(np.squeeze(case_x), 32)
        for x_split in case_split:
            z_split, y_split = model(x_split, 
                                     training=True, 
                                     return_embedding_and_predict=True)
            z_case.append(z_split)
            y_case.append(y_split)
        
        z_case = np.concatenate(z_case, axis=0)
        y_case = np.concatenate(y_case, axis=0)

        load_features = '{}_{}_feat.npy'.format(args.feature_base, case_name)
        features = np.load(load_features)
        load_attention = '{}_{}_att.npy'.format(args.feature_base, case_name)
        attention = np.load(load_attention)
        print('attention: {} z: {}'.format(attention.shape, z_case.shape))

        savepath = '{}_{}.png'.format(args.savebase, case_name)
        print('Saving figure {}'.format(savepath))
        draw_projection(features, attention, z_case, y_case, savepath=savepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_list', type=str)
    parser.add_argument('--classifier_snapshot', 
                        default='../pretraining/trained/classifier-19999',
                        type=str)
    parser.add_argument('--feature_base', default='features/milk', type=str)
    parser.add_argument('--savebase', default='figures/tsne', type=str)

    args = parser.parse_args()
    with tf.device('/gpu:0'):
        main(args)