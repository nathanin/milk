from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import sys
import shutil
import argparse

sys.path.insert(0, '.')
# from segmentation_model import EagerSegmentation
# from segmentation_dataset import Dataset

from classification_model import EagerClassification as Model
from classification_dataset import Dataset

def train_step(model, dataset, optimizer, grad_fn):
    with tf.device('/gpu:0'):
        optimizer.apply_gradients(grad_fn(model, dataset))

def loss_fn(model, dataset):
    x, ytrue = dataset.iterator.next()

    with tf.device('/gpu:0'):
        yhat = model(x.gpu())
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=ytrue, logits=yhat)
        loss = tf.reduce_mean(loss)
    return loss

def get_nested_variables(model):
    """ Recurse through model and get all the variables

    https://github.com/tensorflow/tensorflow/issues/19250
    """
    print('\nFetching a list of all model variables...')
    all_variables = []
    print('DenseNet variables:')
    for v in model.densenet.variables:
        print(' ', v.name, v.shape)
        all_variables.append(v)

    print('DenseNet nested variables:')
    for dense_block in model.densenet.dense_blocks:
        print(' Dense Block:')
        for v in dense_block.variables:
            print('  ', v.name, v.shape)
            all_variables.append(v)
        
        print('  Convs:')
        for convblock in dense_block.blocks:
            for v in convblock.variables:
                print('   ', v.name, v.shape)
                all_variables.append(v)

    for transition_block in model.densenet.transition_blocks:
        print(' Transition Block:')
        for v in transition_block.variables:
            print('  ', v.name, v.shape)
            all_variables.append(v)
        
    print('Model variables:')
    for v in model.variables:
        print(' ', v.name, v.shape)
        all_variables.append(v)

    return all_variables

def main(args):
    print(args) 
    # Get crop size from input_dim and downsample
    crop_size = int(args.input_dim / args.downsample)

    # Build the dataset
    dataset = Dataset(
        record_path = args.dataset,
        crop_size = crop_size,
        downsample = args.downsample,
        n_threads = args.n_threads,
        batch = args.batch_size,
        prefetch_buffer=args.prefetch_buffer
    )

    # Test batch:
    x, y = dataset.iterator.next()
    print('Test batch:')
    print('x: ', x.get_shape())
    print('y: ', y.get_shape())

    with tf.device('/gpu:0'):
        model = Model(n_classes=args.n_classes)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        grad_fn = tfe.implicit_gradients(loss_fn)

    # Process once to print network sizes and initialize the variables:
    yhat = model(x, verbose=True)
    print('yhat: ', yhat.get_shape())

    all_variables = get_nested_variables(model)

    global_step = tf.train.get_or_create_global_step()
    if os.path.exists(args.save_dir): shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    save_prefix = os.path.join(args.save_dir, args.snapshot_prefix)
    saver = tfe.Saver(all_variables)
    saver.save(save_prefix, global_step)

    print('\nStart training...')
    for k in range(args.iterations):
        train_step(model, dataset, optimizer, grad_fn)
        tf.assign(global_step, global_step+1)

        if k % args.save_every == 0:
            print('Saving step ', global_step)
            saver.save(save_prefix, global_step.numpy())

        if k % args.print_every == 0:
            loss_ = loss_fn(model, dataset)
            print('STEP [{:07d}] LOSS = [{:3.4f}]'.format(k, loss_))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=25000, type=int)
    parser.add_argument('--save_dir', default='./trained')
    parser.add_argument('--dataset', default='./data')
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--input_dim', default=128, type=int)
    parser.add_argument('--downsample', default=0.25, type=float)
    parser.add_argument('--n_threads', default=8, type=int)
    parser.add_argument('--prefetch_buffer', default=3096, type=int)
    parser.add_argument('--save_every', default=1000, type=int)
    parser.add_argument('--print_every', default=25, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--snapshot_prefix', default='eager_classifier')

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tfe.enable_eager_execution(config=config)

    main(args)