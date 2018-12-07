"""
Minimal example with bagged-MNIST

## bagged-MNIST
To fit the Multiple Instance Learning problem statement, we consider bags of MNIST images.
First, pick a digit, or rule, to consider "positive".
Choose "9".
Draw a bag of size N. If B contains at least one digit "9", label B positive.

Then, learn a MIL model using this target.

To load up pretrained weights copy them over by name:
https://github.com/keras-team/keras/issues/1873

"""
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np

import argparse
import os 

from milk import Milk

def rearrange_bagged_mnist(x, y, positive_label):
    """
    For simplicity, rearrange the mnist digits from
    x ~ all classes 
    
    into

    x_pos ~ positive class(es)
    x_neg ~ positive class(es)
    """
    positive_mask = y == positive_label
    negative_mask = y != positive_label
    x_pos = x[positive_mask, ...] / 255.
    x_neg = x[negative_mask, ...] / 255.

    return x_pos, x_neg

def generate_negative_bag(x_neg, N):
    n_x_neg = x_neg.shape[0]
    neg_indices = np.random.choice(range(n_x_neg), N, replace=False)
    xbag = x_neg[neg_indices,...]
    np.random.shuffle(xbag)

    return xbag

def generate_positive_bag(x_pos, x_neg, N):
    n_x_pos = x_pos.shape[0]
    n_pos = int(np.random.uniform(low=1, high=int(N * 0.2)))
    # print('Generating bag with {} positive instances'.format(n_pos))
    pos_indices = np.random.choice(range(n_x_pos), n_pos, replace=False)
    xbag = [x_pos[pos_indices,...]]

    n_neg = N - n_pos
    xbag.append(generate_negative_bag(x_neg, n_neg))

    xbag = np.concatenate(xbag, axis=0)
    np.random.shuffle(xbag)
    return xbag

def generate_bagged_mnist(x_pos, x_neg, N, batch):
    """
    x ~ (samples, h, w)
    y ~ (samples)
    N ~ int. the size of the bag 

    return:
    bag_x ~ (1, N, h, w, (c))
    bag_y ~ (1, 2)
    """
    # Coin flip for generating a positive or negative bag:
    while True:
      batch_x, batch_y = [], []
      for _ in range(batch):
        y = np.random.choice([0,1])
        y_onehot = np.zeros((1,2), dtype=np.float32)
        y_onehot[0,y] = 1

        if y == 0:
            xbag = generate_negative_bag(x_neg, N)
            xbag = np.expand_dims(xbag, axis=0)
            xbag = np.expand_dims(xbag, axis=-1)
        else:
            xbag = generate_positive_bag(x_pos, x_neg, N)
            xbag = np.expand_dims(xbag, axis=0)
            xbag = np.expand_dims(xbag, axis=-1)
          
        batch_x.append(xbag.astype(np.float32))
        batch_y.append(y_onehot)
        
      yield np.concatenate(batch_x, axis=0), np.concatenate(batch_y, axis=0)

def main(args):
    if args.mnist is not None:
        (train_x, train_y), (test_x, test_y) = mnist.load_data(args.mnist)
    else:
        (train_x, train_y), (test_x, test_y) = mnist.load_data()

    print('train_x:', train_x.shape, train_x.dtype, train_x.min(), train_x.max())
    print('train_y:', train_y.shape)
    print('test_x:', test_x.shape)
    print('test_y:', test_y.shape)
    
    positive_label = np.random.choice(range(10))
    print('using positive label = {}'.format(positive_label))

    train_x_pos, train_x_neg = rearrange_bagged_mnist(train_x, train_y, positive_label)
    test_x_pos, test_x_neg = rearrange_bagged_mnist(test_x, test_y, positive_label)
    print('rearranged training set:')
    print('\ttrain_x_pos:', train_x_pos.shape, train_x_pos.dtype, 
        train_x_pos.min(), train_x_pos.max())
    print('\ttrain_x_neg:', train_x_neg.shape)
    print('\ttest_x_pos:', test_x_pos.shape)
    print('\ttest_x_neg:', test_x_neg.shape)

    generator = generate_bagged_mnist(train_x_pos, train_x_neg, args.N, 1)
    batch_x, batch_y = next(generator)
    print('batch_x:', batch_x.shape, 'batch_y:', batch_y.shape)

    encoder_args = {
        'depth_of_model': 16,
        'growth_rate': 32,
        'num_of_blocks': 2,
        'output_classes': 2,
        'num_layers_in_each_block': 8,
    }
    model = Milk(input_shape=(args.N, 28, 28, 1), encoder_args=encoder_args)
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

        del pretrained

    model.summary()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

    model.compile(optimizer=optimizer,
                  loss = tf.keras.losses.categorical_crossentropy,
                  metrics = ['categorical_accuracy'])

    model.fit_generator(generator=generator, steps_per_epoch=1000, epochs=10)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=25, type=int)
    parser.add_argument('--tpu', default=False, action='store_true')
    parser.add_argument('--mnist', default=None)
    parser.add_argument('--ntest', default=25, type=int)
    parser.add_argument('--save_prefix', default='./positive_bag/model', type=str)
    parser.add_argument('--initial_weights', default=None, type=str)
    parser.add_argument('--max_fraction_positive', default=0.1, type=int)
    parser.add_argument('--pretrained_model', default='pretrained_model.h5')

    args = parser.parse_args()
    tf.enable_eager_execution()
    main(args)