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
    svs.initialize_output('tiles', dim=3, mode='full')
    return svs

def get_high_att(att, index_img, n=10):
    indices = index_img.ravel()
    uatt = np.argsort(att.ravel())[::-1]
    high_n =  uatt[:n]
    return indices[high_n]

UPSAMPLE = 8
BORDER = 10
def place_imgs(outimg, hi_att, svs, tile_size, border):
    psize = tile_size * UPSAMPLE
    idxmap = svs.ds_tile_map
    for idx in hi_att:
        coords = svs.tile_list[idx]
        img = svs._read_tile(coords)
        pl_mask = idxmap == idx

        # The center
        x,y = np.where(pl_mask)  
        x=x[0]*tile_size; y=y[0]*tile_size

        # Shift  
        x -= int(psize/2)
        y -= int(psize/2)

        img = cv2.resize(img, dsize=(psize-(2*BORDER), psize-(2*BORDER)))
        img = np.pad(img, pad_width=((BORDER, BORDER), (BORDER, BORDER), (0,0)),
                     mode='constant', constant_values=border)
        try:
            outimg[x:psize+x, y:psize+y, :] = img
        except Exception as e:
            print(e)
            continue
    return outimg

def main(matched_slides, args):
    slide = matched_slides[0]
    for slide in matched_slides:
        basename = os.path.basename(slide).replace('.svs', '')
        fgpath = os.path.join(args.fgdir, '{}_fg.png'.format(basename))
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
        outimg = cv2.resize(att, dsize=(0,0), fx=args.tile_size, fy=args.tile_size)
        outimg = np.stack([outimg]*3, axis=2)
        print('output image: {}'.format(outimg.shape))
        hi_att = get_high_att(att, index_img, n=args.n_imgs)

        outimg = place_imgs(outimg, hi_att, svs, args.tile_size, 255)

        outname = os.path.join(args.dstdir, '{}_imprinted.jpg'.format(basename))
        print('saving {}'.format(outname))
        cv2.imwrite(outname, outimg[:,:,::-1])
        os.remove(slide_ramdisk)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', default='attention_images', type=str)
    parser.add_argument('--dstdir', default=None, type=str)
    parser.add_argument('--fgdir',  default='../usable_area/inference', type=str)
    parser.add_argument('--localdisk', default='/dev/shm', type=str)

    ## Needs to match settings used to generate attention map
    parser.add_argument('--mag', default=5, type=int)  
    parser.add_argument('--input_dim', default=96, type=int)

    ## For plottings
    parser.add_argument('--n_imgs', default=50, type=int)
    parser.add_argument('--tile_size', default=96, type=int)

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

    if args.dstdir is None:
        args.__dict__['dstdir'] = args.srcdir

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