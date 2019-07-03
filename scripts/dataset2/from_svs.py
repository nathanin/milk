from MILDataset import MILDataset, create_dataset
from svsutils import PythonIterator, cpramdisk, Slide

import numpy as np
import traceback
import argparse
import cv2
import sys
import os

def stack_tiles(slide, args):
  print('Streaming from slide with {} tiles'.format(len(slide.tile_list)))
  stack = []
  skipped = 0
  it_factory = PythonIterator(slide, args)
  for k, (img, idx) in enumerate(it_factory.yield_one()):
    # Skip white and black tiles
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    white = gray > args.thresh_white
    black = gray < args.thresh_black
    thresh = np.prod(gray.shape) * args.thresh_pct
    if white.sum() > thresh or black.sum() > thresh:
      skipped += 1
      continue
    if k % 1000 == 0:
      print('Tile {:04d}: {} \t skipped: {}'.format(k, img.shape, skipped))
    stack.append(img[:,:,::-1])
  del it_factory
  stack = np.stack(stack, axis=0)

  print('Finished {:04d}: {} \t skipped: {}'.format(k, img.shape, skipped))
  print('Returning stack: {} ({})'.format(stack.shape, stack.dtype))
  return stack


def test_write_read(mildataset):
  ## Testing
  name = 'fakedata'
  data = np.abs(np.random.normal(size=(128, 256, 256, 3)) * 255).astype(np.uint8)
  attr_type = 'label'
  attr_value  = 'class1'
  mildataset.new_dataset(name, data, attr_type, attr_value) 
  print(mildataset.data_group_names)

  read_data, read_label = mildataset.read_data(name, attr_type)
  print('Read back:')
  print(read_data.shape, read_data.dtype)
  print(read_label)


def main(args):
  ## Read in the slides
  with open(args.slides, 'r') as f:
    slides =  [line.strip() for line in f if os.path.exists(line.strip())]

  ## Initialize the dataset; store the metadata file
  with open(args.meta_file, 'r') as f:
    lines = [line for line in f]
  meta_string = ''.join(lines)
  print(meta_string)
  create_dataset(args.data_h5, meta_string, clobber=args.clobber)

  mildataset = MILDataset(args.data_h5)
  print(mildataset.data_group_names)

  for i, src in enumerate(slides):
    print('\n[\t{}/{}\t]'.format(i, len(slides)))
    print('File {:04d} {} --> {}'.format(i, src, args.ramdisk))
    basename = os.path.splitext(os.path.basename(src))[0]
    print(basename)
    rdsrc = cpramdisk(src, args.ramdisk)
    try:
      slide = Slide(rdsrc, args)
      tile_stack = stack_tiles(slide, args)
      mildataset.new_dataset(basename, tile_stack, 'label', 'placeholder')

    except Exception as e:
      print('Breaking')
      traceback.print_tb(e.__traceback__)
    finally:
      print('Removing {}'.format(rdsrc))
      os.remove(rdsrc)

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('data_h5')
  p.add_argument('meta_file')
  p.add_argument('slides')
  p.add_argument('-r', dest='ramdisk', default='/dev/shm', type=str)

  p.add_argument('--clobber', default=False, action='store_true')
  p.add_argument('--thresh_white', default=230, type=int)
  p.add_argument('--thresh_black', default=20,  type=int)
  p.add_argument('--thresh_pct',   default=0.3, type=float)

  # Slide options
  p.add_argument('-b', dest='batchsize', default=1, type=int)
  p.add_argument('--mag',   dest='process_mag', default=5, type=int)
  p.add_argument('--chunk', dest='process_size', default=256, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.05, type=float)

  args = p.parse_args()

  main(args)