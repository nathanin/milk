"""
Print to console some statistics about the contents of a tfrecord dataset

"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import argparse
import os

from milk.utilities import ClassificationDataset

def main(args, sess):
    crop_size = int(args.input_dim / args.downsample)
    class_instances = {k: 0 for k in range(args.n_classes)}

    dataset = ClassificationDataset(
        record_path     = args.test_data,
        crop_size       = crop_size,
        downsample      = args.downsample,
        n_classes       = args.n_classes,
        batch           = args.batch,
        prefetch_buffer = args.prefetch_buffer,
        eager           = False,
        repeats         = args.repeats)
    sess.run(dataset.iterator.initializer)

    ytrue = dataset.y_op

    n_examples = 0
    n_batches = 0
    print('Checking dataset {}'.format(args.test_data))
    while True:
        try:
            ynext = np.argmax(sess.run(ytrue), axis=-1)
            n_next = len(ynext)
            n_examples += n_next
            for c in ynext:
                class_instances[c] += 1

            n_batches += 1
            if n_batches > args.n_batches:
                print('Breaking after {} batches'.format(n_batches))
                break

        except tf.errors.OutOfRangeError:
            print('Done')
            break

    print('Class counts:')
    print('Total instances: {}'.format(n_examples))
    for c in range(args.n_classes):
        ccount = class_instances[c]
        cfreq = ccount / float(n_examples)
        print('{} = {} ~ {:3.3f}'.format(c, ccount, cfreq))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', 
        default='../dataset/gleason_grade_val_ext.tfrecord', type=str)
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--input_dim', default=96, type=int)
    parser.add_argument('--downsample', default=0.25, type=float)
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--n_batches', default=100, type=int)
    parser.add_argument('--repeats', default=3, type=int)
    parser.add_argument('--prefetch_buffer', default=512, type=int)

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    main(args, sess)
    sess.close()
    
