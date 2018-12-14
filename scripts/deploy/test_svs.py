"""
Test classifier on svs's

"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import datetime
import shutil
import pickle
import time
import glob
import cv2
import os
import re

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from svs_reader import Slide

from milk.utilities import data_utils
from milk.utilities import model_utils
from milk.utilities import training_utils
from milk.utilities import model_utils
from milk import Milk, MilkEncode, MilkPredict, MilkAttention


def get_wrapped_fn(svs):
    def wrapped_fn(idx):
        coords = svs.tile_list[idx]
        img = svs._read_tile(coords)
        return img, idx

    return wrapped_fn


def get_img_idx(svs, batch_size, prefetch):
    wrapped_fn = get_wrapped_fn(svs)
    def read_region_at_index(idx):
        return tf.py_func(func = wrapped_fn,
                            inp  = [idx],
                            Tout = [tf.float32, tf.int64],
                            stateful = False)            

    dataset = tf.data.Dataset.from_generator(generator=svs.generate_index,
        output_types=tf.int64)
    dataset = dataset.map(read_region_at_index, num_parallel_calls=4)
    dataset = dataset.prefetch(prefetch)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    img, idx = iterator.get_next()
    return img, idx
        

def transfer_to_ramdisk(src, ramdisk):
    basename = os.path.basename(src)
    dst = os.path.join(ramdisk, basename)
    shutil.copyfile(src, dst)

    return dst

def vect_to_tile(x, target):
    batch = x.shape[0]
    dim = x.shape[-1]
    return np.tile(x, target*target).reshape(batch, target, target, dim)


def main(args, sess):
    ## Search for slides
    # slide_list = []
    # with open('test_lists/{}.txt'.format(args.timestamp), 'r') as f:
    #     for l in f:
    #         slide_list.append(l.replace('\n', ''))

    slide_list = sorted(glob.glob(os.path.join(args.slide_dir, '*.svs')))
    print('Found {} slides'.format(len(slide_list)))

    if args.shuffle:
        np.random.shuffle(slide_list)

    snapshot = os.path.join('save', '{}.h5'.format(args.timestamp))
    trained_model = load_model(snapshot)
    encoder_args = {
        'depth_of_model': 32,
        'growth_rate': 64,
        'num_of_blocks': 4,
        'output_classes': 2,
        'num_layers_in_each_block': 8,
    }
    if args.mcdropout:
        encoder_args['mcdropout'] = True

    encode_model = MilkEncode(input_shape=(args.input_dim, args.input_dim, 3), 
                 encoder_args=encoder_args)
    encode_shape = list(encode_model.output.shape)
    predict_model = MilkPredict(input_shape=[512], mode=args.mil, use_gate=args.gated_attention)
    attention_model = MilkAttention(input_shape=[512], use_gate=args.gated_attention)

    models = model_utils.make_inference_functions(encode_model,
                                                  predict_model,
                                                  trained_model,
                                                  attention_model=attention_model)
    encode_model, predict_model, attention_model = models

    z_pl = tf.placeholder(shape=(None, 512), dtype=tf.float32)
    y_op = predict_model(z_pl)
    att_op = attention_model(z_pl)

    ## Loop over found slides:
    for src in slide_list[:5]:
        ramdisk_path = transfer_to_ramdisk(src, args.ramdisk)  # never use the original src
        svs = Slide(slide_path = ramdisk_path, 
                    preprocess_fn = lambda x: (x/255.).astype(np.float32) ,
                    process_mag=args.mag,
                    process_size=args.input_dim,
                    oversample_factor=1.5)
        # svs.initialize_output(name='prob', dim=args.n_classes, mode='tile')
        svs.initialize_output(name='attention', dim=1, mode='tile')
        # svs.initialize_output(name='rgb', dim=3, mode='full')

        n_tiles = len(svs.tile_list)
        prefetch = min(512, n_tiles)

        # Get tensors for image an index
        img, idx = get_img_idx(svs, args.batch_size, prefetch)
        z_op = encode_model(img)

        batches = 0
        zs = []
        indices = []

        print('Processing {} tiles'.format(len(svs.tile_list)))
        while True:
            try:
                batches += 1
                z, idx_ = sess.run([z_op, idx])
                zs.append(z)
                indices.append(idx_)
                if batches % 25 == 0:
                    print('batch {:04d}\t{}'.format(batches, z.shape))
            except tf.errors.OutOfRangeError:
                print('Done')
                break

        zs = np.concatenate(zs, axis=0)
        indices = np.concatenate(indices)
        print('zs:', zs.shape)
        print('indices:', indices.shape)

        rnd = np.random.choice(indices, 10)
        print(rnd)

        yhat = sess.run(y_op, feed_dict={z_pl: zs})
        print('yhat:', yhat)

        att = sess.run(att_op, feed_dict={z_pl: zs})
        att = np.squeeze(att)
        print('att:', att.shape)
        print(att[rnd])

        plt.hist(att, bins=100); plt.show()

        # svs.place_batch(att, indices, 'attention', mode='tile')
        # attention_img = svs.output_imgs['attention']
        # print('attention image:', attention_img.shape, 
        #       attention_img.dtype, attention_img.min(),
        #       attention_img.max())

        # basename = os.path.basename(src).replace('.svs', '')
        # dst = os.path.join(args.save_dir, '{}.jpg'.format(basename))
        # cv2.imwrite(dst, attention_img * 255.)

        os.remove(ramdisk_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_dir', default='../dataset/svs/', type=str)
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--mcdropout', default=False, action='store_true')
    parser.add_argument('--gated_attention', default=True, action='store_false')
    parser.add_argument('--timestamp', default=None, type=str)
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--input_dim', default=96, type=int)
    parser.add_argument('--mag', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_dir', default='processed_svs', type=str)
    parser.add_argument('--ramdisk', default='/dev/shm', type=str)
    parser.add_argument('--mil', default='attention', type=str)
    args = parser.parse_args()

    assert args.timestamp is not None

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        main(args, sess)
    
