"""
Script to run the MIL workflow on numpy saved tile caches

V1 of the script adds external tracking of val and test lists
for use with test_*.py
"""
from __future__ import print_function
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

EPOCHS = 50
TEST_PCT = 0.1
VAL_PCT = 0.2
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
SHUFFLE_BUFFER = 64
PREFETCH_BUFFER = 128

X_SIZE = 128
Y_SIZE = 128
CROP_SIZE = 96
SCALE = 1.0
MIN_BAG = 50
MAX_BAG = 150
CONST_BAG = 200

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

def main(train_list, val_list, test_list):
    """ 
    1. Create generator datasets from the provided lists
    2. train and validate Milk

    v0 - create datasets within this script
    v1 - factor monolithic training_utils.mil_train_loop !!
    v2 - use a new dataset class @ milk/utilities/mil_dataset.py

    """
    transform_fn = data_utils.make_transform_fn(X_SIZE, Y_SIZE, CROP_SIZE, SCALE)
    train_generator = lambda: data_utils.generator(train_list)
    val_generator = lambda: data_utils.generator(val_list)

    CASE_PATT = r'SP_\d+-\d+'
    def case_label_fn(data_path):
        case = re.findall(CASE_PATT, data_path)[0]
        y_ = CASE_LABEL_DICT[case]
        # print(data_path, y_)
        return y_

    def wrapped_fn(data_path):
        x, y = data_utils.load(data_path.numpy(), 
                            transform_fn=transform_fn, 
                            min_bag=MIN_BAG, 
                            max_bag=MAX_BAG,
                            case_label_fn=case_label_fn)

        return x, y

    def pyfunc_wrapper(data_path):
        return tf.contrib.eager.py_func(func = wrapped_fn,
            inp  = [data_path],
            Tout = [tf.float32, tf.float32],)
            # stateful = False

    ## Tensorflow Eager Iterators can't be on GPU yet
    train_dataset = tf.data.Dataset.from_generator(train_generator, (tf.string), output_shapes = None)
    train_dataset = train_dataset.map(pyfunc_wrapper, num_parallel_calls=2)
    train_dataset = train_dataset.prefetch(PREFETCH_BUFFER)
    train_dataset = tfe.Iterator(train_dataset)

    val_dataset = tf.data.Dataset.from_generator(val_generator, (tf.string), output_shapes = None)
    val_dataset = val_dataset.map(pyfunc_wrapper, num_parallel_calls=2)
    val_dataset = val_dataset.prefetch(PREFETCH_BUFFER)
    val_dataset = tfe.Iterator(val_dataset)

    print('Testing batch generator')
    ## Some api change between nightly built TF and R1.5
    x, y = train_dataset.next()
    print('x: ', x.shape)
    print('y: ', y.shape)

    ## Placae the model and optimizer on the gpu
    print('Placing model, optimizer, and gradient ops on GPU')
    with tf.device('/gpu:0'):
        print('Model initializing')
        model = Milk()

        print('Optimizer initializing')
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        print('Finding implicit gradients')
        grads = tfe.implicit_value_and_gradients(model_utils.loss_function)

    """ Run forward once to initialize the variables """
    t0 = time.time()
    _ = model(x.gpu(), verbose=True)
    print('Model forward step: {}s'.format(time.time() - t0))

    """ Set up training variables, directories, etc. """
    global_step = tf.train.get_or_create_global_step()

    output_strings = training_utils.setup_outputs(return_datestr=True)
    logdir = output_strings[0]
    savedir = output_strings[1]
    imgdir = output_strings[2]
    save_prefix = output_strings[3]
    exptime_str = output_strings[4]
    summary_writer = tf.contrib.summary.create_file_writer(logdir=logdir)

    val_list_file = os.path.join('./val_lists', '{}.txt'.format(exptime_str))
    with open(val_list_file, 'w+') as f:
        for v in val_list:
            f.write('{}\n'.format(v))

    test_list_file = os.path.join('./test_lists', '{}.txt'.format(exptime_str))
    with open(test_list_file, 'w+') as f:
        for v in test_list:
            f.write('{}\n'.format(v))

    training_args = {
        'EPOCHS': EPOCHS,
        'EPOCH_ITERS': len(train_list)*2,
        'global_step': global_step,
        'model': model,
        'optimizer': optimizer,
        'grads': grads,
        # 'saver': saver,
        'save_prefix': save_prefix,
        'loss_function': model_utils.loss_function,
        'accuracy_function': model_utils.accuracy_function,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'img_debug_dir': imgdir,
        'pretrain_snapshot': '../pretraining/trained/classifier-19999'
    }

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        saver, best_snapshot = training_utils.mil_train_loop(**training_args)

    print('Cleaning datasets')
    train_dataset = None
    test_dataset = None
    val_dataset = None
    print('\n\n')

    print('Training done. Find val and test datasets at')
    print(val_list_file)
    print(test_list_file)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fold', default=None, type=int)
    # args = parser.parse_args()
   
    data_patt = '../dataset/tiles/*npy'

    train_list, val_list, test_list = data_utils.list_data(data_patt, 
        val_pct=0.1, test_pct=0.3)

    ## Filter out unwanted samples:
    train_list = filter_list_by_label(train_list)
    val_list = filter_list_by_label(val_list)
    test_list = filter_list_by_label(test_list)

    main(train_list, val_list, test_list)
