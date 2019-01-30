from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import sys
import shutil
import argparse

from milk.utilities import ClassificationDataset
from milk.classifier import Classifier
from milk.utilities import model_utils
from milk.utilities import training_utils

def main(args):
    print(args) 
    # Get crop size from input_dim and downsample
    crop_size = int(args.input_dim / args.downsample)

    # Build the dataset
    dataset = ClassificationDataset(
        record_path = args.dataset,
        crop_size = crop_size,
        downsample = args.downsample,
        n_threads = args.n_threads,
        batch = args.batch_size,
        prefetch_buffer=args.prefetch_buffer,
        shuffle_buffer=args.shuffle_buffer,
    )

    # Test batch:
    x, y = dataset.iterator.next()
    print('Test batch:')
    print('x: ', x.get_shape())
    print('y: ', y.get_shape())

    with tf.device('/gpu:0'):
        model = Classifier(n_classes=args.n_classes)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        grad_fn = tfe.implicit_gradients(model_utils.classifier_loss_fn)

        # Process once to print network sizes and initialize the variables:
        yhat = model(x, verbose=True)
        print('yhat: ', yhat.get_shape())

    all_variables = model.variables ## dont need get_nested_variables if TF > 1.10

    global_step = tf.train.get_or_create_global_step()
    if os.path.exists(args.save_dir): shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    save_prefix = os.path.join(args.save_dir, args.snapshot_prefix)
    saver = tfe.Saver(all_variables)
    saver.save(save_prefix, global_step)

    print('\nStart training...')
    for k in range(args.iterations):
        training_utils.classifier_train_step(model, dataset, optimizer, grad_fn, 
                              global_step=global_step)

        if k % args.save_every == 0:
            print('Saving step ', global_step)
            saver.save(save_prefix, global_step.numpy())

        if k % args.print_every == 0:
            loss_ = model_utils.classifier_loss_fn(model, dataset)
            print('STEP [{:07d}] LOSS = [{:3.4f}]'.format(k, loss_))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=20000, type=int)
    parser.add_argument('--save_dir', default='./trained')
    parser.add_argument('--dataset', default='../dataset/gleason_grade_train_ext.75pct.tfrecord')
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--input_dim', default=96, type=int)
    parser.add_argument('--image_channels', default=3, type=int)
    parser.add_argument('--downsample', default=0.25, type=float)
    parser.add_argument('--n_threads', default=8, type=int)
    parser.add_argument('--prefetch_buffer', default=2048, type=int)
    parser.add_argument('--shuffle_buffer', default=512, type=int)
    parser.add_argument('--save_every', default=500, type=int)
    parser.add_argument('--print_every', default=50, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--snapshot_prefix', default='classifier')

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tfe.enable_eager_execution(config=config)

    main(args)