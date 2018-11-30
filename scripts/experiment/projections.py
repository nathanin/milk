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

def get_extremes(vect, case_x, n=5):
    """
    Grab the n extremes of values in vect, and the images in case_x

    vect ~ a vector length N
    case_x ~ [N, x, y, d]
    n ~ an integer, usually n << N
    """
    vect = np.squeeze(vect)
    vect_sorted_idx = np.argsort(vect)
    top_vect = vect_sorted_idx[-n:]
    low_vect = vect_sorted_idx[:n]

    high_vect_idx = np.zeros_like(vect, dtype=np.bool)
    high_vect_idx[top_vect] = 1

    low_vect_idx = np.zeros_like(vect, dtype=np.bool)
    low_vect_idx[low_vect] = 1

    high_vect_imgs = np.squeeze(case_x, axis=0)[high_vect_idx, ...]
    low_vect_imgs = np.squeeze(case_x, axis=0)[low_vect_idx, ...]
    return high_vect_idx, high_vect_imgs, low_vect_idx, low_vect_imgs

def get_random(vect, case_x, n=25):
    vect = np.squeeze(vect)
    bigN = len(vect)

    idx = np.random.choice(range(bigN), n)
    vect_idx = np.zeros_like(vect, dtype=np.bool)
    vect_idx[idx] = 1

    imgs = np.squeeze(case_x, axis=0)[vect_idx, ...]
    return vect_idx, imgs

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
        ytrue, ypred,
        high_attention,
        high_attention_images,
        low_attention=None,
        low_attention_images=None,
        n_case_rows=1,
        savepath=None):

    if z.shape[-1] > 2:
        z = clusterfn(z)

    ofs_radius = 35
    # 0-Center columns of z
    for k  in range(2):
        z[:,k] = z[:,k] - np.mean(z[:,k])

    z_case = z[-n_case_rows, :]
    z = z[:-n_case_rows, :]
    # ax = plt.subplot(figsize=(4,4), dpi=300, projection='polar')
    fig = plt.figure(figsize=(5,4), dpi=300)
    ax = fig.add_subplot(111)
    print('Projection shape', z.shape)

    plt.scatter(z[:,0], z[:,1], c=attentions, s=15.0, 
        cmap='hot_r',
        # alpha=0.5,
        linewidths=0.5,
        edgecolors='k',
        label='Label={} Pred={}'.format(ytrue, ypred))
    plt.colorbar()

    if len(z_case.shape) == 1:
        z_case = np.expand_dims(z_case, 0)
    plt.scatter(z_case[:,0], z_case[:,1], s=35.0, 
        color='g',
        linewidths=0.5,
        edgecolors='k',
        label='Average')

    plt.xticks([])
    plt.yticks([])
    xmin, xmax = plt.xlim()
    plt.xlim([xmin-ofs_radius, xmax+ofs_radius])
    ymin, ymax = plt.ylim()
    plt.ylim([ymin-ofs_radius, ymax+ofs_radius])

    att_index = np.argwhere(high_attention)
    ax = inset_images(ax, z, att_index, high_attention_images, 
        edgecolor='k', rho_delta=ofs_radius)
        
    if low_attention is not None:
        att_index = np.argwhere(low_attention)
        ax = inset_images(ax, z, att_index, low_attention_images, 
            edgecolor='k', rho_delta=ofs_radius)

    plt.legend(frameon=True)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()


"""
Plot the features projected down to 2D
"""
def draw_projection(features, attentions, ytrue, ypred, 
                    n_case_rows=1,
                    savepath=None):
    plt.figure(figsize=(4,4), dpi=300)
    zall = clusterfn(features)
    z_case = zall[-n_case_rows, :]
    z = zall[:-n_case_rows, :]

    plt.scatter(z[:,0], z[:,1], c=attentions, s=15.0, 
        cmap='hot_r',
        linewidths=0.5,
        edgecolors='k',
        label='Label={} Pred={}'.format(ytrue, ypred))
    plt.colorbar()

    if len(z_case.shape) == 1:
        z_case = np.expand_dims(z_case, 0)
    plt.scatter(z_case[:,0], z_case[:,1], c='g', s=35.0, 
        linewidths=0.5,
        edgecolors='k',
        label='Average')
    plt.xticks([])
    plt.yticks([])
    plt.legend(frameon=True)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()

    return zall


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
    ## Initialize model and lists
    model = Milk()
    x_dummy = tf.zeros(shape=[MIN_BAG, CROP_SIZE, CROP_SIZE, 3], 
                        dtype=tf.float32)
    retvals = model(x_dummy, verbose=True, return_embedding=True, classify_instances=True)
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

    # Control MC dropout behavior
    training = False
    if args.repeats > 1:
        training = True

    for test_case in test_list:

        _embedding, _pvals, _feat_instances = [], [], []
        _repeat_case = []
        _high_pvals ,_high_images = [], []
        _low_pvals, _low_images = [], []
        for _ in range(args.repeats):

            case_name = os.path.basename(test_case).replace('.npy', '')
            case_x, case_y = data_utils.load(
                data_path = test_case,
                transform_fn=transform_fn,
                # const_bag=100,
                all_tiles=True,
                case_label_fn=case_label_fn)
            retvals = model(tf.constant(case_x), training=training, return_embedding=True,
                            classify_instances=True)

            yhat, embedding, feat_case, feat_class, feat_instances, yhat_instances = retvals
            pvals = np.squeeze(yhat_instances[:,1])

            # high_pval_idx, high_pval_imgs, low_pval_idx, low_pval_imgs = get_extremes(pvals, 
            #     case_x, n = 10)

            vect_idx, imgs = get_random(pvals, case_x, n=5)

            yhats.append(yhat.numpy())
            ytrues.append(case_y)
            features_case.append(feat_case.numpy())        
            features_classifier.append(feat_class.numpy())

            _embedding.append(embedding.numpy())
            _repeat_case.append(feat_case.numpy())
            _feat_instances.append(feat_instances.numpy())
            _pvals.append(pvals)
            _high_pvals.append(vect_idx)
            _high_images.append(imgs)
            # _low_pvals.append(low_pval_idx)
            # _low_images.append(low_pval_imgs)
            ytrue = np.argmax(case_y, axis=-1)
            ypred = np.argmax(yhat, axis=-1)
            print('Case {}: label={} predicted={}'.format(
                test_case, ytrue, ypred))

        embedding = np.concatenate(_embedding, axis=0)
        features_instances = np.concatenate(_feat_instances, axis=0)
        repeat_case_features = np.concatenate(_repeat_case, axis=0)
        pvals = np.concatenate(_pvals, axis=0)
        high_pvals = np.concatenate(_high_pvals, axis=0)
        high_pval_images = np.concatenate(_high_images, axis=0)
        # low_pvals = np.concatenate(_low_pvals, axis=0)
        # low_pval_images = np.concatenate(_low_images, axis=0)

        ## Concat instances with the mean
        features_projection = np.concatenate([embedding, repeat_case_features], axis=0)
        print('Projecting matrix {} into 2D'.format(features_projection.shape))

        savepath = '{}_{}.png'.format(args.savebase, case_name)
        print('Saving figure {}'.format(savepath))
        z = draw_projection(features_projection, pvals, ytrue, ypred, savepath=savepath)

        savepath = '{}_{}_imgs.png'.format(args.savebase, case_name)
        print('Saving figure {}'.format(savepath))
        draw_projection_with_images(z, pvals, ytrue, ypred,
            high_pvals, high_pval_images, 
            # low_pvals, low_pval_images, 
            n_case_rows=args.repeats,
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
    parser.add_argument('--feature_base', default='no_attention/features/milk', type=str)
    parser.add_argument('--savebase', default='no_attention/figures/projection', type=str)

    args = parser.parse_args()
    with tf.device('/gpu:0'):
        main(args)