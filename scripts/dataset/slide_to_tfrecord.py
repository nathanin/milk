from __future__ import print_function
import re
import os
import cv2
import sys
import glob
import shutil
import argparse
import numpy as np
import tensorflow as tf

sys.path.insert(0, '/home/nathan/svs_reader')
from slide import Slide 

CASE_PATT = r'(?P<cname>SP.\d+-\d+).+'
def get_case_names(slide_list):
    slide_bases = [os.path.basename(x) for x in slide_list]
    case_names = [re.findall(CASE_PATT, x)[0] for x in slide_bases]
    return case_names

RAM_DISK = '/dev/shm'
def transfer_to_ramdisk(src, ramdisk = RAM_DISK):
    base = os.path.basename(src)
    dst = os.path.join(ramdisk, base)
    shutil.copyfile(src, dst)
    return dst

def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value]))

def dump_slide(slide_path, args):
    tmp_path = transfer_to_ramdisk(slide_path)
    print(tmp_path)
    basename = os.path.basename(slide_path)

    try:
        svs = Slide(slide_path = tmp_path,
                    preprocess_fn = lambda x: x, 
                    normalize_fn = lambda x: x, 
                    process_mag = 5, 
                    process_size = 128,
                    oversample_factor = 1.)
        print(len(svs.tile_list))

        # Save the foreground image for reference
        fg = svs.foreground
        fg_fn = os.path.join(args.out_dir, basename.replace('.svs', '_fg.jpg'))
        cv2.imwrite(fg_fn, fg*255)
        
        # Dump tiles sequentially
        height = _int64_feature(args.size)
        width = _int64_feature(args.size)
        record_path = os.path.join(args.out_dir, basename.replace('.svs', '.tfrecord'))
        writer = tf.python_io.TFRecordWriter(record_path)
        for img, ix in svs.generator():
            img_raw = img.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': height,
                'width': width,
                'img': _bytes_feature(img_raw) }))
            writer.write(example.SerializeToString())
            if ix % 500 == 0:
                print('Writing [{} / {}]'.format(ix, len(svs.tile_list)))
        writer.close()
    except Exception as e:
        print(e)
    finally:
        os.remove(tmp_path)
        print('Done. {} cleaned.'.format(tmp_path))

def dump_dir(args):
    slide_list = glob.glob(os.path.join(args.svs_dir, '*.svs'))
    # case_names = get_case_names(slide_list)
    # print(case_names)

    for slide in slide_list:
        dump_slide(slide, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('svs_dir', default='.')
    parser.add_argument('out_dir', default='.')
    parser.add_argument('--magnification', default=10, type=int)
    parser.add_argument('--size', default=256, type=int)

    args = parser.parse_args()
    dump_dir(args)