"""
Classic MNIST classifier
"""

from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.datasets import mnist
import numpy as np
import argparse

from milk.classifier import Classifier

def generate_batch(x, y, N):
    n_x = x.shape[0]
    idx = np.random.choice(range(n_x), N)

    batch_x = x[idx, ...] / 255.
    batch_y = np.eye(N, 10)[y[idx]]

    return np.expand_dims(batch_x, -1).astype(np.float32), batch_y


def main(args):
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    encoder_args = {
        'depth_of_model': 16,
        'growth_rate': 32,
        'num_of_blocks': 2,
        'output_classes': 2,
        'num_layers_in_each_block': 8,
    }
    model = Classifier(n_classes=10, encoder_args=encoder_args)
    batch_x, batch_y = generate_batch(train_x, train_y, args.batch)
    print('batch:', batch_x.shape, batch_y.shape, batch_x.min(), batch_x.max())

    yhat = model(tf.constant(batch_x), verbose=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    saver = tfe.Saver(model.variables)
    
    for k in range(args.steps):
        with tf.GradientTape() as tape:
            batch_x, batch_y = generate_batch(train_x, train_y, args.batch)
            yhat = model(tf.constant(batch_x))

            loss = tf.losses.softmax_cross_entropy(onehot_labels=batch_y, logits=yhat)
            print('{:06d}\t{}'.format(k, loss))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        if k % 500 == 0:
            print('Testing')
            batch_x, batch_y = generate_batch(test_x, test_y, args.batch)
            yhat = model(tf.constant(batch_x))
            
            test_loss = tf.losses.softmax_cross_entropy(onehot_labels=batch_y, logits=yhat)
            accuracy = (np.argmax(batch_y, -1) == np.argmax(yhat, -1)).mean()
            print('test loss = {} accuracy ~ {}'.format(test_loss, accuracy))

        if k % 1000 == 0:
            print('Saving step {}'.format(k))
            saver.save( file_prefix=args.save_prefix, global_step=k )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=32)
    parser.add_argument('--save_prefix', default='.trained/classifier')
    parser.add_argument('--steps', default=int(1e4))
    args = parser.parse_args()

    tf.enable_eager_execution()
    main(args)