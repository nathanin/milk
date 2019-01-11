"""
Like the builder from the tensorflow examples without all the frills.
For instance we dont care about thinks like bounding boxes or human readable labels

Just read in a tsv with

<class>\t<path> 

and encode the data into `n` tfrecords

The hope is streaming from tfrecords will dramatically improve training speed
and reduce the hard drive hit.

"""
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import shutil
import six
import sys
import os


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=0)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    # assert len(image.shape) == 3
    # assert image.shape[2] == 3
    return image


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(value, six.text_type):           
    value = six.binary_type(value, encoding='utf-8') 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  height = image.shape[0]
  width = image.shape[1]

  if len(image.shape) != 3 or image.shape[2] != 3:
    print('\t\tgot grayscale image; skipping')
    return 0, 0, 0
  else: 
    return image_data, height, width


def _convert_to_example(image_buffer, height, width, label):
  """Build an Example proto for an example.

  Args:
    image_buffer: string, JPEG encoding of RGB image
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  example = tf.train.Example(features=tf.train.Features(feature={
      'label': _int64_feature(label),
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'encoded': _bytes_feature(image_buffer)}))
  return example


blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
              'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
              'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
              'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
              'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
              'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
              'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
              'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
              'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
              'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
              'n07583066_647.JPEG', 'n13037406_4650.JPEG',
              'n02105855_2933.JPEG' ]
def write_record(record_path, shard, coder):
  writer = tf.python_io.TFRecordWriter(record_path)
  print('\tgot {} files'.format(shard.shape[0]))
  written = 0
  for i in range(shard.shape[0]):
    label = shard.iloc[i,0]
    file_path = shard.iloc[i,1]
    
    if file_path.split('/')[-1] in blacklist:
      print('\t\tSkipping {}'.format(file_path))
      continue

    image_buffer, height, width = _process_image(file_path, coder)
    if image_buffer == 0:
      print('\t\tSkipping {}'.format(file_path))
      continue 

    example = _convert_to_example(image_buffer, height, width, label)
    writer.write(example.SerializeToString())
    written += 1

  writer.close()
  sys.stdout.flush()
  print('\twrote {} files'.format(written))


def main(args):
  if not os.path.exists(args.dst):
    os.makedirs(args.dst)

  filenames = pd.read_csv(args.src, sep='\t', header=None, index_col=None)
  filenames = filenames.sample(frac=args.pct)
  print('Sampled {} file names'.format(filenames.shape[0]))

  n_images = filenames.shape[0]
  # shards = int(n_images // args.n)
  sharded_filenames = np.array_split(filenames, args.n)

  coder = ImageCoder()

  print(len(sharded_filenames))
  for i, shard in enumerate(sharded_filenames):
    record_path = os.path.join(args.dst, 'train-{:04d}'.format(i))
    print(record_path, shard.shape)
    write_record(record_path, shard, coder)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('src')
  parser.add_argument('dst')
  parser.add_argument('--n', default=1024, type=int)
  parser.add_argument('--pct', default=1.0, type=float)

  args = parser.parse_args()
  main(args)