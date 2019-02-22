import numpy as np
import sys
import cv2
import os

import seaborn as sns


def draw_attention(attimg, n_bins=100, backgroud=1):
  palette = sns.color_palette("RdBu", n_colors=n_bins)
  bins = np.linspace(-10, 5, n_bins)

  attimg = cv2.resize(attimg, fx=3., fy=3., dsize=(0,0),
    interpolation=cv2.INTER_LINEAR)

  imgout = np.dstack([np.zeros_like(attimg)]*3)
  background_mask = attimg == 0

  ## now that we have the real background, 
  ## we have to level out a low baseline:
  # attimg -= attimg[1-background_mask].min()
  # attimg *= (1. / attimg.max())
  # attimg = -np.log(1-attimg)

  # and bin it
  attimg = np.digitize(attimg, bins)

  for b in range(n_bins):
    b_mask = attimg == b
    imgout[b_mask, ...] = palette[b]

  imgout[background_mask] = backgroud
  imgout *= 255
  imgout = imgout.astype(np.uint8)

  return imgout

if __name__ == '__main__':
  sampleimg = np.random.uniform(low=0, high=255, size=(96,96))
  imgout = draw_attention(sampleimg / 255.)
