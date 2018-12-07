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
    batch_x, batch_y = next(generator)
    print('batch:', batch_x.shape, batch_y.shape, batch_x.min(), batch_x.max())

    encoder_args = {
        'depth_of_model': 16,
        'growth_rate': 32,
        'num_of_blocks': 2,
        'output_classes': 2,
        'num_layers_in_each_block': 8,
    }

    model = Classifier(input_shape=(28,28,1), n_classes=10, encoder_args=encoder_args)
    # model.summary()
    
    if os.path.exists(args.pretrained_model):
        print('Pulling weights from pretrained model')
        print(args.pretrained_model)
        pretrained = load_model(args.pretrained_model)
        pretrained_layers = {l.name: l for l in pretrained.layers if 'encoder' in l.name}
        for l in model.layers:
            if 'encoder' not in l.name:
                continue
            try:
                w = pretrained_layers[l.name].get_weights()
                print('setting layer {}'.format(l.name))
                l.set_weights(w)
            except:
                print('error setting layer {}'.format(l.name))

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['categorical_accuracy'])

    model.fit_generator(generator, steps_per_epoch=args.steps_per_epoch, 
                        epochs=args.epochs)
    model.save(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=96)
    parser.add_argument('--save_path', default='pretrained_model.h5')
    parser.add_argument('--pretrained_model', default='pretrained_model.h5')
    parser.add_argument('--steps_per_epoch', default=int(1e3))
    parser.add_argument('--epochs', default=10)
    args = parser.parse_args()

    main(args)
