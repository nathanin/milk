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
tf.enable_eager_execution(config=config)

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
MIN_BAG = 150
MAX_BAG = 350
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

def main(train_list, val_list, test_list, args):
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
    test_generator = lambda: data_utils.generator(test_list)

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

    logdir, savedir, imgdir, save_prefix, exptime_str = training_utils.setup_outputs()
    training_utils.write_train_val_test_lists(exptime_str, 
        train_list,
        val_list, 
        test_list)
    summary_writer = tf.contrib.summary.create_file_writer(logdir=logdir)

    training_args = {
        'EPOCHS': EPOCHS,
        'EPOCH_ITERS': len(train_list)*5,
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
        'pretrain_snapshot': args.pretraining
    }

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        saver, best_snapshot = training_utils.mil_train_loop(**training_args)

    print('Running TEST SET')
    val_dataset = [] # Clear validation -- we're done with it
    test_dataset = tf.data.Dataset.from_generator(test_generator, (tf.string), output_shapes = None)
    test_dataset = test_dataset.map(pyfunc_wrapper, num_parallel_calls=1)
    test_dataset = tfe.Iterator(test_dataset)

    best_snapshot = save_prefix+'-{}'.format(best_snapshot)
    print('Reverting model to snapshot {}'.format(best_snapshot))
    saver.restore(best_snapshot)
    print('\n\n------------------------- TEST -----------------------------')
    train_loss, test_loss, train_acc, test_acc = training_utils.mil_test_step(model, 
        grads=grads, 
        train_dataset=train_dataset, 
        val_dataset=test_dataset,
        global_step=global_step, 
        mean_batch=0, 
        N=100, 
        loss_function=model_utils.loss_function,
        accuracy_function=model_utils.accuracy_function)

    ## Write the test line
    print('Training Result Summary')
    test_str = 'train loss=[{:3.3f}] '.format(train_loss)
    test_str += 'TEST loss=[{:3.3f}] '.format(test_loss)
    test_str += 'train acc=[{:3.3f}] '.format(train_acc)
    test_str += 'TEST acc=[{:3.3f}] '.format(test_acc)
    print(test_str)

    print('Cleaning datasets')
    train_dataset = None
    test_dataset = None
    val_dataset = None
    print('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', default=None, type=int)
    parser.add_argument('--pretraining', default=None, type=str)

    args = parser.parse_args()
   
    data_patt = '../dataset/tiles/*npy'

    train_list, val_list, test_list = data_utils.list_data(data_patt, val_pct=0.1)

    ## Filter out unwanted samples:
    train_list = filter_list_by_label(train_list)
    val_list = filter_list_by_label(val_list)
    test_list = filter_list_by_label(test_list)

    main(train_list, val_list, test_list, args)
