from __future__ import print_function
import os
import cv2
import sys
import glob
import shutil
import pickle
import argparse
import numpy as np

from svs_reader import Slide 

'''
Update to use case list, and account for multiple slides per case.


'''


RAM_DISK = '/dev/shm'
def transfer_to_ramdisk(src, ramdisk = RAM_DISK):
  base = os.path.basename(src)
  dst = os.path.join(ramdisk, base)
  shutil.copyfile(src, dst)
  return dst

def dump_slide(slide_path, args, return_stack=False):
  tmp_path = transfer_to_ramdisk(slide_path)
  print(tmp_path)
  basename = os.path.basename(slide_path)

  try:
    fgpath = os.path.join(args.fgdir, '{}_fg.png'.format(basename.replace('.svs', '')))
    if os.path.exists(fgpath):
      fgimg = cv2.imread(fgpath, 0)
    else:
      print('Foreground image {} not found'.format(fgpath))
      return None

    svs = Slide(slide_path = tmp_path,
                background_speed  = 'image',
                background_image  = fgimg,
                preprocess_fn = lambda x: x, 
                normalize_fn = lambda x: x, 
                process_mag = args.magnification, 
                process_size = args.size,
                oversample_factor = args.oversample)
    n_tiles = len(svs.tile_list)
    print(n_tiles)

    max_tiles = min(n_tiles, args.max)

    # Save the foreground image for reference
    fg = svs.foreground
    fg_fn = os.path.join(args.out_dir, basename.replace('.svs', '.jpg'))
    cv2.imwrite(fg_fn, fg*255)
    
    # Dump tiles sequentially
    record_path = os.path.join(args.out_dir, basename.replace('.svs', '.npy'))

    img_stack = np.zeros((max_tiles, args.size, args.size, 3), dtype=np.uint8)
    insert_to = 0

    # choice_tiles = np.random.choice(svs.tile_list, max_tiles, replace=False)
    shuffled_tiles = svs.tile_list
    np.random.shuffle(shuffled_tiles)
    # for img, ix in svs.generator():
    for ix, coords in enumerate(shuffled_tiles):
      ## skip missed white space
      img = svs._read_tile(coords)
      img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      white = (img_gray > 230).sum()
      if white > 0.75 * np.square(args.size):
        continue

      # Control for the skipped white tiles
      img_stack[insert_to, ...] = img
      insert_to += 1
      if insert_to % 250 == 0:
        print('Writing [{} / {}]'.format(insert_to, max_tiles))
      
      if insert_to+1 == max_tiles:
        print('Done with {} tiles'.format(insert_to+1))
        break

    print('Trimming stack : {} -- {}'.format(n_tiles, insert_to))
    img_stack = img_stack[:insert_to-1, ...]

    if return_stack:
      return img_stack
    else:
      print('Stack size in memory: {}'.format(img_stack.nbytes))
      np.save(record_path, img_stack)
      img_stack = [] ## Clear the big array

  except Exception as e:
    print(e)
  finally:
    os.remove(tmp_path)
    print('Done. {} cleaned.'.format(tmp_path))

  return None

def dump_sets(args):
  case_dict = {}
  svs_set = pickle.load(open(args.svs_set, 'rb'))
  for caseid, (svslist, stage) in svs_set.items():
    if stage > 1:
      print('Case {} stage {}. skipping.'.format(caseid, stage))
      continue 

    # obfuscated = hashlib.md5(caseid.encode()).hexdigest()
    dst = os.path.join(args.out_dir, '{}.npy'.format(caseid))
    print('dumping case {} to {}'.format(caseid, dst))
    stack = []
    for slide in svslist:
      substack = dump_slide(slide, args, return_stack=True)
      if substack is not None:
        stack.append(substack)

    if len(stack) > 1:
      stack = np.concatenate(stack, axis=0)
    elif len(stack) == 1:
      stack = stack[0]
    else:
      continue
      
    print('case {} got stack {}. Saving {}'.format(caseid, stack.shape, dst))
    np.save(dst, stack)

def dump_dir(args):
  slide_list = glob.glob(os.path.join(args.svs_dir, '*.svs'))
  # case_names = get_case_names(slide_list)
  print(len(slide_list))

  for slide in slide_list:
    status = dump_slide(slide, args)
  
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
  parser.add_argument('--svs_set', default=None)
  parser.add_argument('--max', default=2500, type=int)
  parser.add_argument('--magnification', default=10, type=int)
  parser.add_argument('--size', default=128, type=int)
  parser.add_argument('--oversample', default=1.25, type=float)
  parser.add_argument('--fgdir', default='../usable_area/inference', type=str)

  args = parser.parse_args()
  if args.svs_list is not None:
    dump_list(args)
  elif args.svs_set is not None:
    dump_sets(args)
  else:
    dump_dir(args)
