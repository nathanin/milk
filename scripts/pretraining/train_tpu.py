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
# from milk.utilities import model_utils
# from milk.utilities import training_utils

def main(args):
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

    # Test batch:
    encoder_args = {
        'depth_of_model': 32,
        'growth_rate': 64,
        'num_of_blocks': 4,
        'output_classes': 2,
        'num_layers_in_each_block': 8,
    }
    model = Classifier(input_shape=(args.input_dim, args.input_dim, 3), 
                       n_classes=args.n_classes, encoder_args=encoder_args)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['categorical_accuracy'])
    model.summary()

    print('\nStart training...')
    try:
        model.fit(dataset.dataset.make_one_shot_iterator(),
                  steps_per_epoch=args.iterations,
                  epochs=args.epochs)

    except KeyboardInterrupt:
        print('Stop signal')
    finally:
        print('Saving one last time')
        model.save(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_path', default='./pretrained.h5')
    parser.add_argument('--dataset', default='../dataset/gleason_grade_train_ext.tfrecord')
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
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
    parser.add_argument('--tpu', default=False, action='store_true')

    args = parser.parse_args()

    main(args)