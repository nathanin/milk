import numpy as np
import glob
import cv2

def get_images():
  pth = '../dataset/tile_images/*.jpg'
  img_list = glob.glob(pth)
  images = [cv2.imread(x) for x in img_list]
  return img_list

def main():
  images = get_images()

if __name__ == '__main__':
  main()