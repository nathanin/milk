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

sys.path.insert(0, '../experiment')
from encoder_config import big_encoder_args as encoder_args

def main(args):
  print(args) 
  crop_size = int(args.input_dim / args.downsample)

  # Build the dataset
  dataset = ClassificationDataset(
    record_path = args.dataset,
    crop_size = crop_size,
    downsample = args.downsample,
    n_classes = args.n_classes,
    n_threads = args.n_threads,
    batch = args.batch_size,
    prefetch_buffer = args.prefetch_buffer,
    shuffle_buffer = args.shuffle_buffer,
    eager = False
  )

  # Test batch:
  model = Classifier(input_shape=(args.input_dim, args.input_dim, 3), 
                     n_classes=args.n_classes, 
                     encoder_args=encoder_args, 
                     deep_classifier=True)

  ## Need tf.train for TPU's
  # optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate, decay=1e-6)

  model.compile(optimizer=optimizer,
          loss=tf.keras.losses.categorical_crossentropy,
          metrics=['categorical_accuracy'])
  model.summary()

  try:
    model.fit(dataset.dataset.make_one_shot_iterator(),
          steps_per_epoch=args.iterations,
          epochs=args.epochs)

  except KeyboardInterrupt:
    print('Stop signal')
  finally:
    print('Saving')
    model.save(args.save_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--tpu', default=False, action='store_true')
  parser.add_argument('--epochs', default=50, type=int)
  parser.add_argument('--dataset', default='../dataset/gleason_grade_train_ext.tfrecord')
  parser.add_argument('--n_classes', default=5, type=int)
  parser.add_argument('--save_path', default='./pretrained.h5')
  parser.add_argument('--n_threads', default=12, type=int)
  parser.add_argument('--input_dim', default=96, type=int)
  parser.add_argument('--downsample', default=0.25, type=float)
  parser.add_argument('--iterations', default=2500, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--learning_rate', default=1e-4, type=float)
  parser.add_argument('--shuffle_buffer', default=128, type=int)
  parser.add_argument('--prefetch_buffer', default=512, type=int)

  args = parser.parse_args()

  main(args)
