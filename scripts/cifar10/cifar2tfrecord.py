"""
Converts the cifar-10 data into tfrecords

Make use of ImageCoder class from tensorflow examples
"""
import tensorflow as tf
import pickle
import six
import cv2

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

    # Initializes function that decodes JPEG data in the native channels.
    self._decode_jpeg_native_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg_native = tf.image.decode_jpeg(self._decode_jpeg_native_data, channels=0)

    # Initializes function that decodes JPEG data in the native channels.
    self._decode_jpeg_rgb_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg_rgb = tf.image.decode_jpeg(self._decode_jpeg_rgb_data, channels=3)

    # Initializes function that decodes RGB JPEG data.
    self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
    self._encode_jpeg = tf.image.encode_jpeg(self._encode_jpeg_data, format='rgb')

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg_native(self, image_data):
    image = self._sess.run(self._decode_jpeg_native,
                           feed_dict={self._decode_jpeg_native_data: image_data})
    # assert len(image.shape) == 3
    # assert image.shape[2] == 3
    return image

  def decode_jpeg_rgb(self, image_data):
    image = self._sess.run(self._decode_jpeg_rgb,
                           feed_dict={self._decode_jpeg_rgb_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def encode_jpeg(self, image_data):
    image = self._sess.run(self._encode_jpeg,
                           feed_dict={self._encode_jpeg_data: image_data})
    # assert len(image.shape) == 3
    # assert image.shape[2] == 3
    return image

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(value, six.text_type):           
    value = six.binary_type(value, encoding='utf-8') 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(image_buffer, label):
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
      'image': _bytes_feature(image_buffer)}))
  return example

FILENAMES = [
  'cifar-10-batches-py/data_batch_1',
  'cifar-10-batches-py/data_batch_2',
  'cifar-10-batches-py/data_batch_3',
  'cifar-10-batches-py/data_batch_4',
  'cifar-10-batches-py/data_batch_5',
]
RECORD_PATH = './cifar-10-tfrecord'

def translate_image(img):
  img = img.reshape((32, 32, 3), order='F').transpose([1, 0, 2])
  return img

def main():
  coder = ImageCoder()
  writer = tf.python_io.TFRecordWriter(RECORD_PATH)

  i = 0
  for f in FILENAMES:
    with open(f, 'rb') as fo:
      d = pickle.load(fo, encoding='bytes')
      labs = d[b'labels']
      imgs = d[b'data']
      print(f, imgs.shape, imgs.dtype, len(labs))

      for lab, img in zip(labs, imgs):
        i += 1
        img_ = translate_image(img)
        
        if i % 1000 == 0:
          o = 'debug/{}.png'.format(i)
          cv2.imwrite(o, img_[:,:,::-1])

        img_buffer = coder.encode_jpeg(img_)
        example = _convert_to_example(img_buffer, lab)

        writer.write(example.SerializeToString())
  
  writer.close()

if __name__ == '__main__':
  main()
