from __future__ import print_function
import pandas as pd
import numpy as np
import argparse
import pickle
import shutil
import glob
import cv2
import os

from svs_reader import Slide
from matplotlib import pyplot as plt

def transfer_to_ramdisk(src, ramdisk='/dev/shm'):
    basename = os.path.basename(src)
    dst = os.path.join(ramdisk, basename)
    if os.path.exists(dst):
        print('{} exists'.format(dst))

    else:
        print('Transferring {} --> {}'.format(src, dst))
        shutil.copyfile(src, dst)

    return dst

def load_slide(slide, fgpth, args):
    fgimg = cv2.imread(fgpth, 0)
    svs = Slide(slide_path=slide,
                background_speed='image',
                background_image=fgimg,
                process_mag=args.mag,
                process_size=args.input_dim,
                oversample_factor=1.5)
    print('loaded slide with {} tiles'.format(len(svs.tile_list)))
    return svs

def get_high_att(att, index_img, n=10):
    indices = index_img.ravel()
    uatt = np.argsort(att.ravel())[::-1]
    high_n =  uatt[:n]
    return indices[high_n]

def main(matched_slides, args):
    slide = matched_slides[1]
    basename = os.path.basename(slide).replace('.svs', '')
    fgpath = os.path.join(args.srcdir, '{}_fg.png'.format(basename))
    attpath = os.path.join(args.srcdir, '{}_att.png'.format(basename))
    assert os.path.exists(slide)
    assert os.path.exists(fgpath)
    assert os.path.exists(attpath)
    print(slide, fgpath, attpath)

    slide_ramdisk = transfer_to_ramdisk(slide, args.localdisk)
    svs = load_slide(slide_ramdisk, fgpath, args)
    index_img = svs.ds_tile_map

    att = cv2.imread(attpath, -1)
    print('loaded attention image: {}'.format(att.shape))
    high_attention_indices = get_high_att(att, index_img, n=args.n_imgs)
    print(high_attention_indices)

    for idx in high_attention_indices:
        coords = svs.tile_list[idx]
        img = svs._read_tile(coords)
        plt.imshow(img); plt.show()

    # os.remove(slide_ramdisk)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', default='processed_slides', type=str)
    parser.add_argument('--dstdir', default='processed_slides', type=str)
    parser.add_argument('--localdisk', default='/dev/shm', type=str)

    ## Needs to match settings used to generate attention map
    parser.add_argument('--mag', default=5, type=int)  
    parser.add_argument('--input_dim', default=96, type=int)

    ## For plottings
    parser.add_argument('--n_imgs', default=10, type=int)

    ## Grab a list of svs files
    with open('../dataset/uid2slide.pkl', 'rb') as f:
        uid2slide = pickle.load(f)
    svs_paths = []
    for k, v in uid2slide.items():
        for v_ in v:
            if os.path.exists(v_):
                svs_paths.append(v_)
    print('Got {} slides'.format(len(svs_paths)))
    args = parser.parse_args()

    # Grab the slides with output
    processed_files = glob.glob(os.path.join(args.srcdir, '*_att.png'))
    processed_bases = [os.path.basename(x).replace('_att.png', '') for x in processed_files]

    matched_slides = []
    for s in svs_paths:
        b = os.path.basename(s).replace('.svs', '')
        if b in processed_bases:
            print('\t{}'.format(s))
            matched_slides.append(s)
    print('Got {} matched slides'.format(len(matched_slides)))

    main(matched_slides, args)