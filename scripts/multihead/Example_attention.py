#!/usr/env/bin python
"""
This example deploys a classifier to a list of SVS slides

Utilities demonstrated here:

cpramdisk      - manages and copies data between slow and fast media
Slide          - core object for managing slide data read/write
PythonIterator - hooks for creating generators out of a Slide
xx
TensorflowIterator - A wrapped PythonIterator with multithreading
                     and direct integration with TensorFlow graphs

This script takes advantage of model constructors defined in 
https://github.com/nathanin/milk


Usage
-----
```
python Example_classifier.py [slides.txt] [model/snapshot.h5] [encoder type] [options]
```

June 2019
"""
from svsutils import repext
from svsutils import cpramdisk
from svsutils import Slide
from svsutils import reinhard
from svsutils import PythonIterator
from svsutils import TensorflowIterator

import tensorflow as tf
import numpy as np
import traceback

from milk.eager import MilkEager
from milk.encoder_config import get_encoder_args

import argparse
import os

def main(args):


  # Define a compute_fn that should do three things:
  # 1. define an iterator over the slide's tiles
  # 2. compute an output with a given model / arguments
  # 3. return a reconstructed slide
  def compute_fn(slide, args, model=None, n_dropout=10 ):
    assert tf.executing_eagerly()
    print('Slide with {}'.format(len(slide.tile_list)))

    # In eager mode, we return a tf.contrib.eager.Iterator
    eager_iterator = TensorflowIterator(slide, args).make_iterator()

    # The iterator can be used directly. Ququeing and multithreading
    # are handled in the backend by the tf.data.Dataset ops
    features, indices = [], []
    for k, (img, idx) in enumerate(eager_iterator):
      # img = tf.expand_dims(img, axis=0)
      features.append( model.encode_bag(img, training=False, return_z=True) )
      indices.append(idx.numpy())

      img, idx = img.numpy(), idx.numpy()
      if k % 50 == 0:
        print('Batch #{:04d}\t{}'.format(k, img.shape))

    features = tf.concat(features, axis=0)

    ## Sample-dropout
    # features = features.numpy()
    # print(features.shape)
    # n_instances = features.shape[0]
    # att = np.zeros(n_instances)
    # n_choice = int(n_instances * 0.7)
    # all_heads = list(range(args.heads))
    # for j in range(n_dropout):
    #   idx = np.random.choice(range(n_instances), n_choice, replace=False)
    #   print(idx)
    #   fdrop = features[idx, :]

    z_att, att = model.mil_attention(features,
                                     training=False, 
                                     return_raw_att=True)

    # att[idx] += np.squeeze(attdrop)
    yhat_multihead = model.apply_classifier(z_att, heads=all_heads, 
      training=False)
    print('yhat mean {}'.format(np.mean(yhat_multihead, axis=0)))

    indices = np.concatenate(indices)
    att = np.squeeze(att)
    slide.place_batch(att, indices, 'att', mode='tile')
    ret = slide.output_imgs['att']
    print('Got attention image: {}'.format(ret.shape))

    return ret, features.numpy()




  ## Begin main script:
  # Set up the model first
  encoder_args = get_encoder_args(args.encoder)
  model = MilkEager(encoder_args=encoder_args,
                    mil_type=args.mil,
                    deep_classifier=args.deep_classifier,
                    batch_size=args.batchsize,
                    temperature=args.temperature,
                    heads = args.heads)
  
  x = tf.zeros((1, 1, args.process_size,
                args.process_size, 3))
  all_heads = [0,1,2,3,4,5,6,7,8,9]
  _ = model(x, verbose=True, heads=all_heads, training=True)
  model.load_weights(args.snapshot, by_name=True)

  # keras Model subclass
  model.summary()

  # Read list of inputs
  with open(args.slides, 'r') as f:
    slides = [x.strip() for x in f]

  # Loop over slides
  for src in slides:
    # Dirty substitution of the file extension give us the
    # destination. Do this first so we can just skip the slide
    # if this destination already exists.
    # Set the --suffix option to reflect the model / type of processed output
    dst = repext(src, args.suffix)
    featdst = repext(src, args.suffix+'.feat.npy')

    # Loading data from ramdisk incurs a one-time copy cost
    rdsrc = cpramdisk(src, args.ramdisk)
    print('\n\nFile:', rdsrc)

    # Wrapped inside of a try-except-finally.
    # We want to make sure the slide gets cleaned from 
    # memory in case there's an error or stop signal in the 
    # middle of processing.
    try:
      # Initialze the side from our temporary path, with 
      # the arguments passed in from command-line.
      # This returns an svsutils.Slide object
      slide = Slide(rdsrc, args)

      # This step will eventually be included in slide creation
      # with some default compute_fn's provided by svsutils
      # For now, do it case-by-case, and use the compute_fn
      # that we defined just above.
      slide.initialize_output('att', args.n_classes, mode='tile',
        compute_fn=compute_fn)

      # Call the compute function to compute this output.
      # Again, this may change to something like...
      #     slide.compute_all
      # which would loop over all the defined output types.
      ret, features = slide.compute('att', args, model=model)
      print('{} --> {}'.format(ret.shape, dst))
      print('{} --> {}'.format(features.shape, featdst))
      np.save(dst, ret)
      np.save(featdst, features)
    except Exception as e:
      print(e)
      traceback.print_tb(e.__traceback__)
    finally:
      print('Removing {}'.format(rdsrc))
      os.remove(rdsrc)

if __name__ == '__main__':
  """
  standard __name__ == __main__
  how to make this nicer
  """
  p = argparse.ArgumentParser()
  p.add_argument('slides') 
  p.add_argument('snapshot') 
  p.add_argument('encoder') 
  p.add_argument('--iter_type', default='tf', type=str) 
  p.add_argument('--suffix', default='.att.npy', type=str) 

  # common arguments with defaults
  p.add_argument('-b', dest='batchsize', default=64, type=int)
  p.add_argument('-r', dest='ramdisk', default='/dev/shm', type=str)
  p.add_argument('-j', dest='workers', default=8, type=int)
  p.add_argument('-c', dest='n_classes', default=1, type=int)

  # Slide options
  p.add_argument('--mag',   dest='process_mag', default=10, type=int)
  p.add_argument('--chunk', dest='process_size', default=128, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.0, type=float)
  p.add_argument('--verbose', dest='verbose', default=False, action='store_true')

  # Keywords for the model
  p.add_argument('--mil',   dest='mil', default='attention', type=str)
  p.add_argument('--heads',   dest='heads', default=10, type=int)
  p.add_argument('--temperature',   dest='temperature', default=0.5, type=float)
  p.add_argument('--deep_classifier', dest='deep_classifier', default=False, type=bool)

  args = p.parse_args()

  # Functionals for later:
  args.__dict__['preprocess_fn'] = lambda x: (reinhard(x) / 255.).astype(np.float32)

  tf.enable_eager_execution()
  main(args)