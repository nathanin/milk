from __future__ import print_function
import re
import os
import cv2
import sys
import glob
import shutil
import argparse
import numpy as np

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

def dump_slide(slide_path, args):
    tmp_path = transfer_to_ramdisk(slide_path)
    print(tmp_path)
    basename = os.path.basename(slide_path)

    try:
        svs = Slide(slide_path = tmp_path,
                    preprocess_fn = lambda x: x, 
                    normalize_fn = lambda x: x, 
                    process_mag = args.magnification, 
                    process_size = args.size,
                    oversample_factor = args.oversample)
        n_tiles = len(svs.tile_list)
        print(n_tiles)

        # Save the foreground image for reference
        fg = svs.foreground
        fg_fn = os.path.join(args.out_dir, basename.replace('.svs', '_fg.jpg'))
        cv2.imwrite(fg_fn, fg*255)
        
        # Dump tiles sequentially
        record_path = os.path.join(args.out_dir, basename.replace('.svs', '.npy'))

        img_stack = np.zeros((n_tiles, args.size, args.size, 3), dtype=np.uint8)
        insert_to = 0
        for img, ix in svs.generator():
            ## skip missed white space
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            white = (img_gray > 230).sum()
            if white > 0.75 * np.square(args.size):
                continue

            img_stack[insert_to, ...] = img
            insert_to += 1
            if ix % 500 == 0:
                print('Writing [{} / {}]'.format(ix, len(svs.tile_list)))

        print('Trimming stack : {} -- {}'.format(n_tiles, insert_to))
        img_stack = img_stack[:insert_to-1, ...]
        np.save(record_path, img_stack)
        img_stack = [] ## Clear the big array

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
    
def dump_list(args):

    slide_list = []
    with open(args.svs_list, 'r') as f:
        for L in f:
            if os.path.exists(L):
                slide_list.append(L)
            else:
                print('WARNING {} not found'.format(L))

    for slide in slide_list:
        dump_slide(slide, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('svs_dir', default='.')
    parser.add_argument('out_dir', default='.')
    parser.add_argument('--svs_list', default=None)
    parser.add_argument('--magnification', default=10, type=int)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--oversample', default=1.5, type=float)

    args = parser.parse_args()
    if args.svs_list is not None:
        dump_list(args)
    else:
        dump_dir(args)