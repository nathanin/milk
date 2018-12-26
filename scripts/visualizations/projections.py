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
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
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
    
    print('Got {} test cases'.format(len(test_list)))
    return test_list

def get_attention_extremes(attention, case_x, n=5):
    attention = np.squeeze(attention)
    att_sorted_idx = np.argsort(attention)
    top_att = att_sorted_idx[-n:]
    low_att = att_sorted_idx[:n]

    high_att_idx = np.zeros_like(attention, dtype=np.bool)
    high_att_idx[top_att] = 1

    low_att_idx = np.zeros_like(attention, dtype=np.bool)
    low_att_idx[low_att] = 1

    high_att_imgs = np.squeeze(case_x, axis=0)[high_att_idx, ...]
    low_att_imgs = np.squeeze(case_x, axis=0)[low_att_idx, ...]
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
        im = OffsetImage(att_img, zoom=0.2)
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

def draw_projection_with_images(z, attentions, 
        high_attention,
        high_attention_images,
        low_attention,
        low_attention_images,
        savepath=None):

    ofs_radius = 35
    # 0-Center columns of z
    for k  in range(2):
        z[:,k] = z[:,k] - np.mean(z[:,k])

    # ax = plt.subplot(figsize=(4,4), dpi=300, projection='polar')
    fig = plt.figure(figsize=(5,4), dpi=300)
    ax = fig.add_subplot(111)
    print('Projection shape', z.shape)

    plt.scatter(z[:,0], z[:,1], c=attentions, s=15.0, 
        cmap='hot_r',
        # alpha=0.5,
        linewidths=0.5,
        edgecolors='k')
    plt.xticks([])
    plt.yticks([])
    xmin, xmax = plt.xlim()
    plt.xlim([xmin-ofs_radius, xmax+ofs_radius])
    ymin, ymax = plt.ylim()
    plt.ylim([ymin-ofs_radius, ymax+ofs_radius])
    plt.colorbar()

    att_index = np.argwhere(high_attention)
    ax = inset_images(ax, z, att_index, high_attention_images, 
        edgecolor='k', rho_delta=ofs_radius)
        
    att_index = np.argwhere(low_attention)
    ax = inset_images(ax, z, att_index, low_attention_images, 
        edgecolor='k', rho_delta=ofs_radius)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()


"""
Plot the features projected down to 2D
"""
def draw_projection(features, attentions, savepath=None):
    plt.figure(figsize=(4,4), dpi=300)
    z = clusterfn(features)

    plt.scatter(z[:,0], z[:,1], c=attentions, s=15.0, 
        cmap='hot_r',
        linewidths=0.5,
        edgecolors='k')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')
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


transform_fn = data_utils.make_transform_fn(X_SIZE, Y_SIZE, CROP_SIZE, SCALE, 
    normalize=True)
def main(args):
    model = Milk()
    x_dummy = tf.zeros(shape=[MIN_BAG, CROP_SIZE, CROP_SIZE, 3], 
                        dtype=tf.float32)
    retvals = model(x_dummy, verbose=True, return_embedding=True)
    for k, retval in enumerate(retvals):
        print('retval {}: {}'.format(k, retval.shape))

    saver = tfe.Saver(model.variables)
    if args.snapshot is None:
        snapshot = tf.train.latest_checkpoint(args.snapshot_dir)
    else:
        snapshot = args.snapshot
    print('Restoring from {}'.format(snapshot))

    saver.restore(snapshot)
    model.summary()

    test_list = read_test_list(args.test_list)

    yhats, ytrues = [], []
    features_case, features_classifier = [], []
    for test_case in test_list:
        _features, _attention = [], []
        _high_attention ,_high_images = [], []
        _low_attention, _low_images = [], []
        for _ in range(args.repeats):
            case_name = os.path.basename(test_case).replace('.npy', '')
            case_x, case_y = data_utils.load(
                data_path = test_case,
                transform_fn=transform_fn,
                # const_bag=100,
                all_tiles=True,
                case_label_fn=case_label_fn)
            retvals = model(tf.constant(case_x), training=False, return_embedding=True)

            yhat, attention, features, feat_case, feat_class = retvals
            attention = np.squeeze(attention.numpy(), axis=0)
            high_att_idx, high_att_imgs, low_att_idx, low_att_imgs = get_attention_extremes(
                attention, case_x, n = 15)

            yhats.append(yhat.numpy())
            ytrues.append(case_y)
            features_case.append(feat_case.numpy())        
            features_classifier.append(feat_class.numpy())

            _features.append(features.numpy())
            _attention.append(attention)
            _high_attention.append(high_att_idx)
            _high_images.append(high_att_imgs)
            _low_attention.append(low_att_idx)
            _low_images.append(low_att_imgs)
            print('Case {}: label={} predicted={}'.format(
                test_case, np.argmax(case_y,axis=-1), np.argmax(yhat, axis=-1)))

        features = np.concatenate(_features, axis=0)
        attention = np.concatenate(_attention, axis=0)
        high_attention = np.concatenate(_high_attention, axis=0)
        high_attention_images = np.concatenate(_high_images, axis=0)
        low_attention = np.concatenate(_low_attention, axis=0)
        low_attention_images = np.concatenate(_low_images, axis=0)

        savepath = '{}_{}.png'.format(args.savebase, case_name)
        print('Saving figure {}'.format(savepath))
        z = draw_projection(features, attention, savepath=savepath)

        savepath = '{}_{}_imgs.png'.format(args.savebase, case_name)
        print('Saving figure {}'.format(savepath))
        draw_projection_with_images(z, attention, 
            high_attention, high_attention_images, 
            low_attention, low_attention_images, 
            savepath=savepath)

    yhats = np.concatenate(yhats, axis=0)
    ytrues = np.concatenate(ytrues, axis=0)
    features_case = np.concatenate(features_case, axis=0) 
    features_classifier = np.concatenate(features_classifier, axis=0)

    tlbase = os.path.splitext(os.path.basename(args.test_list))[0]
    savepath = '{}_{}.png'.format(args.savebase, tlbase)
    print('Saving case-wise projection: {}'.format(savepath))
    draw_class_projection(features_case, ytrues, yhats, savepath=savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default=None, type=str)
    parser.add_argument('--snapshot_dir', default=None, type=str)
    parser.add_argument('--test_list', type=str)
    parser.add_argument('--repeats', default=1, type=int)
    parser.add_argument('--feature_base', default='features/milk', type=str)
    parser.add_argument('--savebase', default='figures/projection', type=str)

    args = parser.parse_args()
    with tf.device('/gpu:0'):
        main(args)