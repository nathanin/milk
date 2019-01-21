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

from encoder_config import encoder_args
from data_util import CifarRecords
from milk.classifier import Classifier

def main(args):
    print(args) 
    # Build the dataset
    assert args.dataset is not None
    dataset = CifarRecords(src=args.dataset, xsize=args.input_dim,
      ysize=args.input_dim, batch=args.batch_size, buffer=args.prefetch_buffer,
      parallel=args.threads)

    # Test batch:
    model = Classifier(input_shape=(args.input_dim, args.input_dim, 3), 
                       n_classes=args.n_classes, encoder_args=encoder_args)
    #optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate, decay=1e-7)
    optimizer = tf.train.AdamOptimizer(args.learning_rate)

    if args.gpus > 1:
      model = tf.keras.utils.multi_gpu_model(model, args.gpus, cpu_relocation=True)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['categorical_accuracy'])
    model.summary()

    try:
        model.fit(dataset.make_one_shot_iterator(),
                  steps_per_epoch=20000,
                  epochs=args.epochs)

    except KeyboardInterrupt:
        print('Stop signal')
    finally:
        print('Saving one last time')
        model.save(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--dataset', default='cifar-10-tfrecord', type=str)
    parser.add_argument('--input_dim', default=32, type=int)
    parser.add_argument('--threads', default=4, type=int)
    parser.add_argument('--save_path', default='./cifar_10_model.h5')
    parser.add_argument('--n_classes', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--prefetch_buffer', default=2048, type=int)

    args = parser.parse_args()

    main(args)
