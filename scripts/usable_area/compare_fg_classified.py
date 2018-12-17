"""
Gather and print some stats about the foreground area in the slide data
"""
import os
import cv2
import glob
import numpy as np

from matplotlib import pyplot as plt

accurate_fg_list = glob.glob('./accurate_fgimg/*_fg.png')
classified_fg_list = glob.glob('./inference/*_fg.png')
print('Got {} "accurate" images'.format(len(accurate_fg_list)))
print('Got {} "classified" images'.format(len(classified_fg_list)))

accurate_fg_base = [os.path.basename(x).replace('_fg.png', '') for x in accurate_fg_list]
classified_fg_base = [os.path.basename(x).replace('_fg.png', '') for x in classified_fg_list]

print('Matches:')
matches = []
for ab in accurate_fg_base:
    if ab in classified_fg_base:
        print(ab)
        matches.append(ab)

print(len(matches))

def print_stats(lst):
    mn = np.mean(lst)
    md = np.median(lst)
    sd = np.std(lst)

    print('\tMean: ', mn)
    print('\tMedian: ', md)
    print('\tstd: ', sd)
    print('\tmin: ', np.min(lst))
    print('\tmax: ', np.max(lst))

aareas = []
careas = []
diffs = []
for m in matches:
    a = os.path.join('accurate_fgimg', '{}_fg.png'.format(m))
    c = os.path.join('inference', '{}_fg.png'.format(m))
    assert os.path.exists(a)
    assert os.path.exists(c)

    aimg = (cv2.imread(a, 0) > 0).astype(np.uint8)
    cimg = (cv2.imread(c, 0) > 0).astype(np.uint8)
    
    asum = float(aimg.sum())
    csum = float(cimg.sum())


    dsum = asum - csum
    print('Slide {}\ta = {: 5.0f}\tc = {: 5.0f}\tdelta = {: 3.0f}'.format(m, asum, csum, dsum))
    aareas.append(asum)
    careas.append(csum)
    diffs.append(dsum)

print('\n"Accurate" areas:')
print_stats(aareas)
print('\n"Classified" areas:')
print_stats(careas)
print('\nDifference:')
print_stats(diffs)