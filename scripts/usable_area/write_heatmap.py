from __future__ import print_function
import os
import cv2
import glob
import argparse
import numpy as np

POS_CLASS = 0
def main(npydir, outdir):
    print(npydir)
    npylist = sorted(glob.glob(os.path.join(npydir, '*.npy')))

    for npy in npylist:
        outfile = npy.replace('.npy', '.jpg')
        x = np.load(npy)
        xpos = x[..., POS_CLASS]
        print(npy, outfile, x.shape, xpos.shape, xpos.min(), xpos.max())
        cv2.imwrite(outfile, xpos)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npydir')
    parser.add_argument('--outdir', default=None)

    args = parser.parse_args()
    if args.outdir is None:
        outdir = args.npydir
    else:
        outdir = args.outdir
    
    main(args.npydir, outdir)
    