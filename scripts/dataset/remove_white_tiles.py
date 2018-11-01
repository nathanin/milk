from __future__ import print_function
import numpy as np
import cv2
import os
import glob
import argparse

def main(args):
    # Check input and output
    npy_list = sorted(glob.glob(os.path.join(args.in_dir, '*.npy')))

    if os.path.exists(args.out_dir):
        print('WARNING {} exists.'.format(args.out_dir))
        return 0
    else:
        os.makedirs(args.out_dir)

    for npy_file in npy_list:
        fout = os.path.basename(npy_file)
        fout = os.path.join(args.out_dir, fout)
        print(fout, end=': ')

        xout = []
        x = np.load(npy_file)
        print(x.shape, x.dtype, end=' --> ')
        
        for img in x:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            white = gray > args.threshold

            if (white.sum() / float(x.shape[1] * x.shape[2])) > args.pct_threshold:
                continue
            else:
                xout.append(np.expand_dims(img, 0))

        xout = np.concatenate(xout, axis = 0)
        np.save(fout, xout)
        print(xout.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--threshold', default=210, type=int)
    parser.add_argument('--pct_threshold', default=0.5, type=float)

    args = parser.parse_args()

    main(args)

