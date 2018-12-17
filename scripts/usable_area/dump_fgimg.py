"""
Dump the 'accurate' mode fg image, and compare with the
selection of the trained classifier

"""

import numpy as np
import pickle
import shutil
import cv2
import os

from svs_reader import Slide

OUTDIR = 'accurate_fgimg'

def transfer_to_ramdisk(src, ramdisk = '/dev/shm'):
    base = os.path.basename(src)
    dst = os.path.join(ramdisk, base)
    print('Transferring: {} --> {}'.format(src, dst))
    shutil.copyfile(src, dst)
    return dst

with open('../dataset/uid2slide.pkl', 'rb') as f:
    df = pickle.load(f)
slide_list = []
for k, v in df.items():
    for v_ in v:
        if os.path.exists(v_):
            slide_list.append(v_)
print('Slide list: {}'.format(len(slide_list)))

for i, src in enumerate(slide_list):
    print('\nSlide {}'.format(i))
    basename = os.path.basename(src).replace('.svs', '')
    fgpth = os.path.join(OUTDIR, '{}_fg.png'.format(basename))
    if os.path.exists(fgpth):
        print('{} exists'.format(fgpth))
        continue

    ramdisk_path = transfer_to_ramdisk(src)  # never use the original src
    print('Using fg image at : {}'.format(fgpth))
    try:
        svs = Slide(slide_path        = ramdisk_path, 
                    # background_speed  = 'accurate',
                    background_speed  = 'accurate',
                    preprocess_fn     = lambda x: (x/255.).astype(np.float32) ,
                    process_mag       = 5,
                    process_size      = 96,
                    oversample_factor = 1.5,
                    verbose           = True)
        print('calculated foregroud: ', svs.foreground.shape)
        print('calculated ds_tile_map: ', svs.ds_tile_map.shape)
        cv2.imwrite(fgpth, (svs.ds_tile_map > 0).astype(np.uint8) * 255)
    except:
        print('Slide error')
    finally:
        os.remove(ramdisk_path)