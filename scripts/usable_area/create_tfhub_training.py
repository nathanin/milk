from __future__ import print_function
import numpy as np
import cv2
import os
import sys
import glob
import argparse

IMGSIZE = 128
N_SUBIMGS = 2
def split_subimgs(img):
    x, y = img.shape[:2]
    target_x = int(x / IMGSIZE) * N_SUBIMGS
    target_y = int(y / IMGSIZE) * N_SUBIMGS
    x_vect = np.linspace(0, x-IMGSIZE, target_x, dtype=np.int)
    y_vect = np.linspace(0, y-IMGSIZE, target_y, dtype=np.int)

    subimgs = []
    for x_ in x_vect:
        for y_ in y_vect:
            try:
                subimgs.append(img[x_:x_+IMGSIZE, y_:y_+IMGSIZE, :])
            except:
                subimgs.append(img[x_:x_+IMGSIZE, y_:y_+IMGSIZE])

    return subimgs

def initialize_output_dirs(output_base):
    os.makedirs(output_base)

    for k in range(5):
        pth = os.path.join(output_base, '{}'.format(k))
        os.makedirs(pth)

N_CLASSES = 2
CLASSES = range(N_CLASSES)
def main(args):
    imglist = sorted(glob.glob(os.path.join(args.source_dir, '*.jpg')))
    
    totalimgs = 0
    for imgpath in imglist:
        img = cv2.imread(imgpath, -1)
        img = cv2.resize(img, dsize=(0,0), fx=0.125, fy=0.125)
        subimgs = split_subimgs(img)

        print(imgpath, len(subimgs))
        for subimg in subimgs:
            imgname = os.path.join(args.output_dir, '{:06d}_435.jpg'.format(totalimgs))
            cv2.imwrite(imgname, subimg)
            totalimgs += 1
            # print(imgname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')
    parser.add_argument('output_dir')

    args = parser.parse_args()

    main(args)
