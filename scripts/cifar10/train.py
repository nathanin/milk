"""
Train a classifier with keras API

The classifier will be reused as initialization for the MIL encoder
so it must have at least a subset with the same architecture.
Currently, it's basically required that the entier encoding architecture
is kept constant.

python 3.x
"""
import tensorflow as tf
import sys
import argparse

from data_util import CifarRecords
from milk.classifier import Classifier

#sys.path.insert(0, '../experiment')
from cifar_encoder_config import encoder_args

def main(args):
  print(args) 
  # Build the dataset
  assert args.dataset is not None

  dataset = CifarRecords(src=args.dataset, xsize=args.input_dim,
    ysize=args.input_dim, batch=args.batch_size, buffer=args.prefetch_buffer,
    parallel=args.threads)

  # Test batch:
  model = Classifier(input_shape=(args.input_dim, args.input_dim, 3), 
                     n_classes=args.n_classes, encoder_args=encoder_args,
                     deep_classifier=True)

  ## BUG tf.keras.optimizers.Adam breaks in multi-gpu mode ?
  optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate, decay=1e-6)
  #optimizer = tf.train.AdamOptimizer(args.learning_rate)

  if args.gpus > 1:
    model = tf.keras.utils.multi_gpu_model(model, args.gpus, cpu_relocation=True)

  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['categorical_accuracy'])
  model.summary()

  try:
    model.fit(dataset.make_one_shot_iterator(),
              steps_per_epoch = 10000,
              epochs = args.epochs)
              # steps_per_epoch=60000 // args.batch_size,
              # epochs=args.epochs)
  except KeyboardInterrupt:
    print('Stop signal')
  except Exception as e:
    print('Exception')
    print(e)
  finally:
    print('Saving')
    model.save(args.save_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpus', default=1, type=int)
  parser.add_argument('--epochs', default=50, type=int)
  parser.add_argument('--dataset', default='data/cifar-10-tfrecord', type=str)
  parser.add_argument('--threads', default=8, type=int)
  parser.add_argument('--input_dim', default=96, type=int)
  parser.add_argument('--save_path', default='./cifar_10_model.h5')
  parser.add_argument('--n_classes', default=10, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--learning_rate', default=1e-4, type=float)
  parser.add_argument('--prefetch_buffer', default=4096, type=int)

  args = parser.parse_args()

  main(args)
