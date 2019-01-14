"""
Classic MNIST classifier
"""

from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import argparse
import os

from milk import make_encoder
from milk import Classifier

from encoder_config import encoder_args

def generate_batch(x, y, N):
    while True:
        n_x = x.shape[0]
        idx = np.random.choice(range(n_x), N)

        batch_x = x[idx, ...] / 255.
        batch_y = np.eye(N, 10)[y[idx]]

        yield np.expand_dims(batch_x, -1).astype(np.float32), batch_y


def main(args):
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    generator = generate_batch(train_x, train_y, args.batch)
    test_generator = generate_batch(test_x, test_y, args.batch)
    batch_x, batch_y = next(generator)
    print('batch:', batch_x.shape, batch_y.shape, batch_x.min(), batch_x.max())

    model = Classifier(input_shape=(28,28,1), n_classes=10, encoder_args=encoder_args)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['categorical_accuracy'])

    model.fit_generator(generator, 
                        steps_per_epoch=args.steps_per_epoch, 
                        epochs=args.epochs,
                        validation_data=test_generator,
                        validation_steps=50)
    model.save(args.o)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', default='pretrained_model.h5')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch', default=96, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--steps_per_epoch', default=int(1e3))
    args = parser.parse_args()

    main(args)
