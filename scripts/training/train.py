from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import time, datetime
import cPickle as pickle
import sys, os, glob
import argparse

fpath = os.path.dirname(os.path.realpath(__file__))
# from dataset import DatasetFactory
import data_utils
import utilities

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tfe.enable_eager_execution(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 50
TEST_PCT = 0.1
VAL_PCT = 0.2
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
SHUFFLE_BUFFER = 128
PREFETCH_BUFFER = 128

X_SIZE = 128
Y_SIZE = 128
CROP_SIZE = 96
SCALE = 1.0
MIN_BAG = 100
MAX_BAG = 300
CONST_BAG = 200

def main(train_list, val_list, test_list, Model, loss_function, logdir_base, fold_num):
    transform_fn = data_utils.make_transform_fn(X_SIZE, Y_SIZE, CROP_SIZE, SCALE)
    train_generator = lambda: data_utils.generator(train_list)
    val_generator = lambda: data_utils.generator(val_list)
    test_generator = lambda: data_utils.generator(test_list)

    def wrapped_fn(data_path):
        x, y = data_utils.load(data_path.numpy(), transform_fn=transform_fn, min_bag=MIN_BAG, max_bag=MAX_BAG)
        # x, y = data_utils.load(data_path.numpy(), transform_fn=transform_fn, const_bag=CONST_BAG)

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
        model = Model()

        print('Optimizer initializing')
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)

        print('Finding implicit gradients')
        grads = tfe.implicit_gradients(loss_function)

    """ Run forward once to initialize the variables """
    _ = model(x.gpu(), verbose=True)

    """ Set up training variables, directories, etc. """
    global_step = tf.train.get_or_create_global_step()

    exptime = datetime.datetime.now()
    logdir = '{}/bigbatch_log/{}'.format(logdir_base, 
        exptime.strftime('fold_{}_%Y_%m_%d_%H_%M_%S'.format(fold_num)))
    os.makedirs(logdir)
    summary_writer = tf.contrib.summary.create_file_writer(logdir=logdir)

    save_dir = '{}/bigbatch_snapshot/{}'.format(logdir_base, 
        exptime.strftime('fold_{}_%Y_%m_%d_%H_%M_%S'.format(fold_num)))
    os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, 'snapshot')

    img_debug_dir = '{}/bigbatch_output/{}'.format(logdir_base, 
        exptime.strftime('fold_{}_%Y_%m_%d_%H_%M_%S'.format(fold_num)))
    os.makedirs(img_debug_dir)

    training_args = {
        'EPOCHS': EPOCHS,
        'EPOCH_ITERS': len(train_list)*4,
        'global_step': global_step,
        'model': model,
        'optimizer': optimizer,
        'grads': grads,
        # 'saver': saver,
        'save_prefix': save_prefix,
        'loss_function': loss_function,
        'accuracy_function': accuracy_function,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'img_debug_dir': img_debug_dir
    }

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        saver, best_snapshot = utilities.train_loop(**training_args)

    print('Running TEST')
    val_dataset = [] # Clear validation -- we're done with it
    test_dataset = tf.data.Dataset.from_generator(test_generator, (tf.string), output_shapes = None)
    test_dataset = test_dataset.map(pyfunc_wrapper, num_parallel_calls=1)
    # test_dataset = test_dataset.prefetch(PREFETCH_BUFFER)
    test_dataset = tfe.Iterator(test_dataset)

    best_snapshot = save_prefix+'-{}'.format(best_snapshot)
    print('Reverting model to snapshot {}'.format(best_snapshot))
    saver.restore(best_snapshot)
    print('\n\n------------------------- TEST -----------------------------')
    train_loss, test_loss, train_acc, test_acc = utilities.test_step(model, loss_function,
        grads, train_dataset, test_dataset,
        global_step, 0, accuracy_function)

    ## Write the test line
    print('Writing summary to')
    result_file = '{}/bigbatch_result_{}.txt'.format(logdir_base, 
        exptime.strftime('fold_{}_%Y_%m_%d_%H_%M_%S'.format(fold_num)))
    test_str = 'train loss=[{:3.3f}] '.format(train_loss)
    test_str += 'TEST loss=[{:3.3f}] '.format(test_loss)
    test_str += 'train acc=[{:3.3f}] '.format(train_acc)
    test_str += 'TEST acc=[{:3.3f}] '.format(test_acc)
    print(result_file, test_str)
    with open(result_file, 'w+') as f:
        f.write(test_str)

    print('Cleaning datasets')
    train_dataset = None
    test_dataset = None
    val_dataset = None
    print('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arch_dir')
    parser.add_argument('--fold', default=None, type=int)

    args = parser.parse_args()
    arch_dir = args.arch_dir
    if args.fold is None:
        fold_num = np.random.randint(10)
        print('Random fold: {}'.format(fold_num))
    else:
        fold_num = args.fold

    # arch_dir = sys.argv[1]
    # fold_num = sys.argv[2]
    # arch_dirs = ['arch1', 'arch2', 'arch3', 'arch3b']
    data_patt = 'dataset/tiles/pruned_no_white/*npy'

    sys.path.insert(0, os.path.join(fpath, arch_dir))
    # from model import Model, loss_function, accuracy_function, debug_fn
    from bayesian_densenet import Model, loss_function, accuracy_function, debug_fn

    data_list = glob.glob(data_patt)
    # kf = KFold(n_splits=10, shuffle=True, random_state=1337)
    # for train_idx, test_idx in kf.split(data_list):
    fold_splits = pickle.load(open('folds_10.pkl', 'r'))
    fold_info = fold_splits[int(fold_num)]

    train_list = np.asarray(data_list)[fold_info[0]]
    test_list = np.asarray(data_list)[fold_info[1]]

    # print(train_list)
    # print(test_list)
    ## Split off val from train
    train_list, val_list = data_utils.split_train_test(train_list, test_pct=0.2)

    main(train_list, val_list, test_list, Model, loss_function, arch_dir, fold_num)
    tf.reset_default_graph()
