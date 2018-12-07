"""
Script to run the MIL workflow on numpy saved tile caches

V1 of the script adds external tracking of val and test lists
for use with test_*.py
"""
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import datetime
import glob
import time
import sys 
import os 
import re

from milk.utilities import data_utils
from milk import Milk

sys.path.insert(0, '..')
from dataset import CASE_LABEL_DICT

def filter_list_by_label(lst):
    lst_out = []
    for l in lst:
        l_base = os.path.basename(l)
        l_base = os.path.splitext(l_base)[0]
        if CASE_LABEL_DICT[l_base] != 2:
            lst_out.append(l)
    print("Got list length {}; returning list length {}".format(
        len(lst), len(lst_out)
    ))
    return lst_out

CASE_PATT = r'SP_\d+-\d+'
def case_label_fn(data_path):
    case = re.findall(CASE_PATT, data_path)[0]
    y_ = CASE_LABEL_DICT[case]
    # print(data_path, y_)
    return y_

def main(args):
    """ 
    1. Create generator datasets from the provided lists
    2. train and validate Milk

    v0 - create datasets within this script
    v1 - factor monolithic training_utils.mil_train_loop !!
    tpu - replace data feeder and mil_train_loop with tf.keras.Model.fit()
    """
    train_list, val_list, test_list = data_utils.list_data(args.data_patt, 
        val_pct=0.1, test_pct=0.3)

    ## Filter out unwanted samples:
    train_list = filter_list_by_label(train_list)
    val_list = filter_list_by_label(val_list)
    test_list = filter_list_by_label(test_list)

    train_list = data_utils.enforce_minimum_size(train_list, args.bag_size, verbose=True)
    val_list = data_utils.enforce_minimum_size(val_list, args.bag_size, verbose=True)
    test_list = data_utils.enforce_minimum_size(test_list, args.bag_size, verbose=True)
    transform_fn = data_utils.make_transform_fn(args.x_size, 
                                                args.y_size, 
                                                args.crop_size, 
                                                args.scale)

    train_generator = data_utils.load_generator(train_list, 
        transform_fn=transform_fn, 
        bag_size=args.bag_size, 
        case_label_fn=case_label_fn)
    val_generator = data_utils.load_generator(val_list, 
        transform_fn=transform_fn, 
        bag_size=args.bag_size, 
        case_label_fn=case_label_fn)

    print('Testing batch generator')
    ## Some api change between nightly built TF and R1.5
    x, y = next(train_generator)
    print('x: ', x.shape)
    print('y: ', y.shape)
    encoder_args = {
        'depth_of_model': 32,
        'growth_rate': 64,
        'num_of_blocks': 4,
        'output_classes': 2,
        'num_layers_in_each_block': 8,
    }

    print('Model initializing')
    model = Milk(input_shape=(args.bag_size, args.crop_size, args.crop_size, 3), 
                 encoder_args=encoder_args)
    
    if os.path.exists(args.pretrained_model):
        pretrained_model = load_model(args.pretrained_model)
        pretrained_layers = {l.name: l for l in pretrained_model.layers if 'encoder' in l.name}

        for l in model.layers:
            if 'encoder' not in l.name:
                continue
            try:
                w = pretrained_layers[l.name].get_weights()
                print('setting layer {}'.format(l.name))
                l.set_weights(w)
            except:
                print('error setting layer {}'.format(l.name))

        del pretrained_model 
        

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    output_strings = training_utils.setup_outputs(return_datestr=True)
    exptime_str = output_strings[4]
    out_path = '{}.h5'.format(exptime_str)

    val_list_file = os.path.join('./val_lists', '{}.txt'.format(exptime_str))
    with open(val_list_file, 'w+') as f:
        for v in val_list:
            f.write('{}\n'.format(v))

    test_list_file = os.path.join('./test_lists', '{}.txt'.format(exptime_str))
    with open(test_list_file, 'w+') as f:
        for v in test_list:
            f.write('{}\n'.format(v))

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['categorical_accuracy'])

    model.fit_generator(generator=train_generator,
        validation_data=val_generator,
        validation_steps=10,
        steps_per_epoch=1000, 
        epochs=50)

    print('Training done. Find val and test datasets at')
    print(val_list_file)
    print(test_list_file)
    model.save(out_path)
    print('Saved model: {}'.format(out_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--test_pct', default=0.1, type=float)
    parser.add_argument('--val_pct', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--data_patt', default='../dataset/tiles/*npy', type=str)
    parser.add_argument('--x_size', default=128, type=int)
    parser.add_argument('--y_size', default=128, type=int)
    parser.add_argument('--crop_size', default=96, type=int)
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--bag_size', default=150, type=int)
    parser.add_argument('--pretrained_model', default='../pretraining/pretrained.h5', type=str)

    args = parser.parse_args()


    main(args)
