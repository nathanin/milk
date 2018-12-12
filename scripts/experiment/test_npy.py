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
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from milk.utilities import data_utils
from milk.utilities import model_utils
from milk.utilities import training_utils
from milk import Milk, MilkEncode, MilkPredict

with open('../dataset/cases_md5.pkl', 'rb') as f:
    case_dict = pickle.load(f)

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


def run_sample(case_x, encode_model, predict_model, mcdropout=False, 
               batch_size=64):
    if mcdropout:
        yhats = []
        for _ in range(10):
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
    pretrained_model = load_model(snapshot)
    encoder_args = {
        'depth_of_model': 32,
        'growth_rate': 64,
        'num_of_blocks': 4,
        'output_classes': 2,
        'num_layers_in_each_block': 8,
    }
    if args.mcdropout:
        encoder_args['mcdropout'] = True

    print('Model initializing')
    encode_model = MilkEncode(input_shape=(args.crop_size, args.crop_size, 3), 
                 encoder_args=encoder_args)
    encode_shape = list(encode_model.output.shape)
    predict_model = MilkPredict(input_shape=[512], mode=args.mil)

    encode_model, predict_model = model_utils.make_inference_functions(encode_model,
                                                                       predict_model,
                                                                       pretrained_model)

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

    if args.savepath is not None:
        savepath = os.path.join(args.savepath, '{}.png'.format(args.timestamp))
    else:
        savepath = None
    auc_curve(ytrue, yhat, savepath=savepath)

if __name__ == '__main__':
    # n_repeat controls re-sampling from the case
    # mcdropout is a flag for doing mcdropout to approximate posterior probability
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', default=None, type=str)
    parser.add_argument('--n_repeat', default=1, type=int)
    parser.add_argument('--mcdropout', default=False, action='store_true')
    parser.add_argument('--testdir', default='test_lists', type=str)
    parser.add_argument('--savepath', default=None, type=str)

    parser.add_argument('--x_size', default=128, type=int)
    parser.add_argument('--y_size', default=128, type=int)
    parser.add_argument('--crop_size', default=96, type=int)
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--mil', default='attention', type=str)

    args = parser.parse_args()
    assert args.timestamp is not None
    main(args)