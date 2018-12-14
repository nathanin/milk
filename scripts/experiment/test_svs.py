"""
Test classifier on svs's

We need two files to translate between obfuscated filenames, and the 
original svs files:

1. case_dict_obfuscated.pkl is a pickled dictionary with the form:
   {uid: label, ...}
   We ust it to go from the uid's saved in the test list for a particular run to labels

2. uid2slide.pkl is a pickled dictionary with the form:
   {uid: [/path/to/1.svs, /path/to/2.svs,...], ...}
   We use it to translate from the uid given in the test list to a set of slides
    for that particular case. There may be more than one slide, so we'll just choose one.

   Be aware that the slide lists in uid2slide need to point somewhere on the local filesystem.
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

# uid2label = pickle.load(open('../dataset/case_dict_obfuscated.pkl', 'rb'))
uid2label = pickle.load(open('../dataset/cases_md5.pkl', 'rb'))
uid2slide = pickle.load(open('../dataset/uid2slide.pkl', 'rb'))

def get_wrapped_fn(svs):
    def wrapped_fn(idx):
        coords = svs.tile_list[idx]
        img = svs._read_tile(coords)
        return img, idx

    return wrapped_fn

def auc_curve(ytrue, yhat, savepath=None):
    plt.figure(figsize=(2,2), dpi=300)
    yhat_c = yhat[:,1]; print(yhat_c.shape)
    ytrue = np.squeeze(ytrue); print(ytrue.shape)
    auc_c = roc_auc_score(y_true=ytrue, y_score=yhat_c)
    fpr, tpr, _ = roc_curve(y_true=ytrue, y_score=yhat_c)

    plt.plot(fpr, tpr, 
        label='M1 AUC={:3.3f}'.format(auc_c))
        
    plt.legend(loc='lower right', frameon=True)
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - specificity')

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')

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

def read_test_list(test_file):
    test_list = []
    with open(test_file, 'r') as f:
        for l in f:
            test_list.append(l.replace('\n', ''))
    
    return test_list

def get_slidelist_from_uids(uids):
    slide_list_out = []
    labels = []
    for uid in uids:
        try:
            slide_list = uid2slide[uid]
            if len(slide_list) > 0:
                slide_list_out.append(np.random.choice(slide_list))
                labels.append(uid2label[uid])
            else:
                print('WARNING UID: {} found no associated slides'.format(uid))
        except:
            print('WARNING UID: {} not found in slide list'.format(uid))

    print('Testing on slides:')
    for s, l in zip(slide_list_out, labels):
        print('{}\t\t{}'.format(s, l))
    return slide_list_out, labels

def main(args, sess):
    ## Search for slides
    # slide_list = []
    # with open('test_lists/{}.txt'.format(args.timestamp), 'r') as f:
    #     for l in f:
    #         slide_list.append(l.replace('\n', ''))

    # Translate obfuscated file names to paths if necessary
    test_list = os.path.join(args.testdir, '{}.txt'.format(args.timestamp))
    test_list = read_test_list(test_list)
    test_unique_ids = [os.path.basename(x).replace('.npy', '') for x in test_list]
    slide_list, slide_labels = get_slidelist_from_uids(test_unique_ids)

    print('Found {} slides'.format(len(slide_list)))

    snapshot = os.path.join(args.savedir, '{}.h5'.format(args.timestamp))
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
    yhats = []
    ytrues = []
    fig = plt.figure(figsize=(2,2), dpi=180)
    for src, lab in zip(slide_list, slide_labels):
        ramdisk_path = transfer_to_ramdisk(src, args.ramdisk)  # never use the original src

        # Check for a background image, and if found, use it to speed things up.
        try:
            background_mode = 'accurate'
            basename = os.path.basename(src).replace('.svs', '')
            fgdst = os.path.join(args.odir, '{}_fg.png'.format(basename))
            if os.path.exists(fgdst):
                print('Found fg img at: {}'.format(fgdst))
                fgimg = cv2.imread(fgdst,-1)
                background_mode = 'image'
                svs = Slide(slide_path        = ramdisk_path, 
                            background_speed  = 'image',
                            background_image  = fgimg,
                            preprocess_fn     = lambda x: (x/255.).astype(np.float32) ,
                            process_mag       = args.mag,
                            process_size      = args.input_dim,
                            oversample_factor = 1.5)
            else:
                print('No background found')
                svs = Slide(slide_path        = ramdisk_path, 
                            background_speed  = 'accurate',
                            preprocess_fn     = lambda x: (x/255.).astype(np.float32) ,
                            process_mag       = args.mag,
                            process_size      = args.input_dim,
                            oversample_factor = 1.5)
            # svs.initialize_output(name='prob', dim=args.n_classes, mode='tile')
            svs.initialize_output(name='attention', dim=1, mode='tile')
            # svs.initialize_output(name='rgb', dim=3, mode='full')
        except:
            print('Error making slide')
            continue

        n_tiles = len(svs.tile_list)
        prefetch = min(512, n_tiles)

        # Get tensors for image and index
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
                    print('batch {:04d}\t{}\t{}'.format(
                           batches, z.shape, batches*args.batch_size))
            except tf.errors.OutOfRangeError:
                print('Done')
                break

        zs = np.concatenate(zs, axis=0)
        indices = np.concatenate(indices)
        print('zs:', zs.shape)
        print('indices:', indices.shape)


        yhat = sess.run(y_op, feed_dict={z_pl: zs})
        print('yhat:', yhat)

        rnd = np.random.choice(indices, 10)
        att = sess.run(att_op, feed_dict={z_pl: zs})
        att = np.squeeze(att)
        print('att:', att.shape)
        print(att[rnd])

        yhats.append(yhat)
        ytrues.append(lab)
        print('Slide label: {} predicted: {}'.format(lab, yhat))

        svs.place_batch(att, indices, 'attention', mode='tile')
        attention_img = svs.output_imgs['attention']
        print('attention image:', attention_img.shape, 
              attention_img.dtype, attention_img.min(),
              attention_img.max())
        attention_img = attention_img * (255. / attention_img.max())

        dst = os.path.join(args.odir, '{}.jpg'.format(basename))
        cv2.imwrite(dst, attention_img)

        fgdst = os.path.join(args.odir, '{}_fg.png'.format(basename))
        fgimg = (svs.ds_tile_map > 0).astype(np.uint8)
        cv2.imwrite(fgdst, fgimg * 255)

        dst = os.path.join(args.odir, '{}.png'.format(basename))
        plt.clf()
        plt.hist(att, bins=100); 
        plt.title('Attention distribution\n{} ({} tiles)'.format(basename, n_tiles))
        plt.xlabel('Attention score')
        plt.ylabel('Tile count')
        plt.savefig(dst, bbox_inches='tight')

        os.remove(ramdisk_path)

    dst = os.path.join(args.odir, '{}.png'.format(args.timestamp))
    yhats = np.concatenate(yhats, axis=0)
    ytrue = np.array(ytrues)
    for i, yt in enumerate(ytrue):
        print(yt, yhats[i, :])

    auc_curve(ytrue, yhats, savepath=dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', default=None, type=str)
    parser.add_argument('--testdir',   default='test_lists', type=str)
    parser.add_argument('--savedir',   default='save', type=str)
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--input_dim', default=96, type=int)
    parser.add_argument('--mag',       default=5, type=int)
    parser.add_argument('--batch_size',default=64, type=int)
    parser.add_argument('--odir',      default='attention_svs', type=str)
    parser.add_argument('--ramdisk',   default='/dev/shm', type=str)

    parser.add_argument('--mcdropout', default=False, action='store_true')
    parser.add_argument('--mil',       default='attention', type=str)
    parser.add_argument('--gated_attention', default=True, action='store_false')
    args = parser.parse_args()

    assert args.timestamp is not None

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        main(args, sess)
    
