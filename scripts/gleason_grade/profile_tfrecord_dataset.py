"""
Print to console some statistics about the contents of a tfrecord dataset
"""

import tensorflow as tf
import numpy as np
import argparse
import time
import os

from matplotlib import pyplot as plt
from milk.utilities import ClassificationDataset

def main(args, sess):
    crop_size = int(args.input_dim / args.downsample)
    class_instances = {k: 0 for k in range(args.n_classes)}

    dataset = ClassificationDataset(
        record_path     = args.test_data,
        crop_size       = crop_size,
        downsample      = args.downsample,
        n_classes       = args.n_classes,
        batch           = args.batch_size,
        prefetch_buffer = args.prefetch_buffer,
        eager           = False,
        repeats         = args.repeats)
    sess.run(dataset.iterator.initializer)

    ytrue = dataset.y_op

    n_examples = 0
    n_batches = 0
    tstart_loop = time.time()
    batch_times = []
    print('Checking dataset {}'.format(args.test_data))
    while True:
        try:
            tstart = time.time()
            ynext = np.argmax(sess.run(ytrue), axis=-1)
            deltaT = time.time() - tstart
            n_next = len(ynext)
            n_examples += n_next
            for c in ynext:
                class_instances[c] += 1

            batch_times.append(deltaT)
            n_batches += 1
            if n_batches > args.n_batches:
                print('Breaking after {} batches'.format(n_batches))
                break

        except tf.errors.OutOfRangeError:
            print('Done')
            break

    tend_loop = time.time()
    print('Finished {} batches of {}'.format(n_batches, args.batch_size))
    print('Done in {}s'.format(tend_loop - tstart_loop))
    print('Class counts:')
    print('Total instances: {}'.format(n_examples))
    for c in range(args.n_classes):
        ccount = class_instances[c]
        cfreq = ccount / float(n_examples)
        print('{} = {} ~ {:3.3f}'.format(c, ccount, cfreq))

    plt.plot(range(n_batches), batch_times)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', 
        default='../dataset/gleason_grade_val_ext.tfrecord', type=str)
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--input_dim', default=96, type=int)
    parser.add_argument('--downsample', default=0.25, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_batches', default=1e5, type=int)
    parser.add_argument('--repeats', default=3, type=int)
    parser.add_argument('--prefetch_buffer', default=512, type=int)

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    main(args, sess)
    sess.close()
    
