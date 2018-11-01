"""
Train a classifier in graph mode

The classifier will be reused as initialization for the MIL encoder
so it must have at least a subset with the same architecture.
Currently, it's basically required that the entier encoding architecture
is kept constant.

"""
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

def main(args, sess):
    print(args) 
    # Get crop size from input_dim and downsample
    crop_size = int(args.input_dim / args.downsample)

    # Build the dataset
    dataset = ClassificationDataset(
        record_path = args.dataset,
        crop_size = crop_size,
        downsample = args.downsample,
        n_classes = args.n_classes,
        n_threads = args.n_threads,
        batch = args.batch_size,
        prefetch_buffer=args.prefetch_buffer,
        shuffle_buffer=args.shuffle_buffer,
        eager = False
    )
    sess.run(dataset.iterator.initializer)

    # Test batch:
    with tf.device('/gpu:0'):
        model = Classifier(n_classes=args.n_classes)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        x = tf.placeholder_with_default(dataset.x_op, 
            shape=[args.batch_size, args.input_dim, args.input_dim, args.image_channels], 
            name='x_in')
        ytrue = tf.placeholder_with_default(dataset.y_op,
            shape=[args.batch_size, args.n_classes],
            name='ytrue')
        yhat = model(x)
        all_variables = model.variables ## dont need get_nested_variables if TF > 1.10
        loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ytrue, logits=yhat)
        train_op = optimizer.minimize(loss_op)

    model.summary()

    sess.run(tf.global_variables_initializer())

    if os.path.exists(args.save_dir): shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    save_prefix = os.path.join(args.save_dir, args.snapshot_prefix)
    saver = tf.train.Saver(all_variables)
    saver.save(sess, save_prefix, 0)

    print('\nStart training...')
    try:
        for k in range(args.iterations):
            _, loss_ = sess.run([train_op, loss_op])
            if k % args.save_every == 0:
                print('Saving step ', k)
                saver.save(sess, save_prefix, k)

            if k % args.print_every == 0:
                # print(k, loss_)
                print('STEP [{:07d}] LOSS = [{:3.4f}]'.format(
                    k, np.mean(loss_)
                ))
    except KeyboardInterrupt:
        print('Stop signal')
    finally:
        print('Saving one last time')
        saver.save(sess, save_prefix, k)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=20000, type=int)
    parser.add_argument('--save_dir', default='./trained')
    parser.add_argument('--dataset', default='../dataset/gleason_grade_train_ext.75pct.tfrecord')
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
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
    sess = tf.Session(config=config)

    try:
        main(args, sess)
    except KeyboardInterrupt:
        print('Stopping early.')
    finally:
        print('Done.')
        sess.close()