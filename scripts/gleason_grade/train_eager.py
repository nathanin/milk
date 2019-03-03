from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import shutil
import sys
import os

from milk.utilities import ClassificationDataset
from milk.eager import ClassifierEager

from milk.utilities import model_utils
from milk.utilities import training_utils

from tensorflow.keras.layers import Input

from milk.encoder_config import get_encoder_args

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
    prefetch_buffer = args.prefetch_buffer,
    shuffle_buffer = args.shuffle_buffer,
    device = args.device,
    device_buffer = args.device_buffer,
    eager = True
  )

  # Test batch:
  batchx, batchy = next(dataset.iterator)
  print('Test batch:')
  print('batchx: ', batchx.get_shape())
  print('batchy: ', batchy.get_shape())

  ## Working on this way -- hoping for a speed up with the keras.Model.fit() functions..
  # input_tensor = Input(name='images', shape=[args.input_dim, args.input_dim, args.image_channels] )
  # logits = ClassifierEager(encoder_args=encoder_args, n_classes=args.n_classes)(input_tensor)
  # model = tf.keras.Model(inputs=input_tensor, outputs=logits)

  encoder_args = get_encoder_args(args.encoder)
  model = ClassifierEager(encoder_args=encoder_args, n_classes=args.n_classes)
  yhat = model(batchx, training=True, verbose=True)
  print('yhat: ', yhat.get_shape())

  if args.snapshot is not None and os.path.exists(args.snapshot):
    model.load_weights(args.snapshot)

  # optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate, decay=1e-5)
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

  model.summary()

  # model.compile(optimizer = optimizer,
  #   loss = 'categorical_crossentropy',
  #   metrics = ['categorical_accuracy'])

  # model.fit_generator(generator=tf.contrib.eager.Iterator(dataset),
  #   steps_per_epoch = args.steps_per_epoch,
  #   epochs = args.epochs)

  try:
    running_avg = []
    for k in range(args.steps_per_epoch * args.epochs):
      with tf.GradientTape() as tape:
        batchx, batchy = next(dataset.iterator)
        yhat = model(batchx)

        loss = tf.keras.losses.categorical_crossentropy(y_true=batchy, y_pred=yhat)
        running_avg.append(np.mean(loss))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

      if k % 50 == 0:
        print('{:05d} loss={:3.3f}'.format(k, np.mean(running_avg)))
        running_avg = []

  except:
    print('Caught exception')

  finally:  
    print('Saving')
    model.save_weights(args.saveto)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--epochs', default=50, type=int)
  parser.add_argument('--steps_per_epoch', default=1000, type=int)
  parser.add_argument('--learning_rate', default=1e-4, type=float)
  parser.add_argument('--saveto', default='eager_classifier.h5')
  parser.add_argument('--encoder', default='big', type=str)

  # Usually don't change these
  parser.add_argument('--dataset', default='../dataset/gleason_grade_train_ext.75pct.tfrecord')
  parser.add_argument('--n_threads', default=8, type=int)
  parser.add_argument('--input_dim', default=96, type=int)
  parser.add_argument('--n_classes', default=5, type=int)
  parser.add_argument('--downsample', default=0.25, type=float)
  parser.add_argument('--image_channels', default=3, type=int)
  parser.add_argument('--shuffle_buffer', default=512, type=int)
  parser.add_argument('--prefetch_buffer', default=2048, type=int)
  parser.add_argument('--device', default='/gpu:0', type=str)
  parser.add_argument('--device_buffer', default=64, type=int)

  parser.add_argument('--snapshot', default=None, type=str)
  parser.add_argument('--encoder', default='big', type=str)

  args = parser.parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  tf.enable_eager_execution(config=config)

  main(args)
