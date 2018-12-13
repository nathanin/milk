from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe

class ClassificationDataset(object):
    """ Eager-enabled dataset for image/mask pairs

    With on the fly augmentation functions

    TODO add a switch for eager mode.
    Test if prefetch_to_device in graph mode is significantly faster.
    Prefetch buffer doesn't seem to work
    """
    def __init__(self, 
                 record_path, 
                 crop_size = 512, 
                 downsample = 0.25, 
                 n_classes = 5, 
                 n_threads = 6, 
                 batch = 16, 
                 prefetch_buffer = 4096, 
                 shuffle_buffer = 2048,
                 eager = True, 
                 repeats = None):

        # Get a preprocessing function that uses tensorflow ops only
        preprocessing = self._build_preprocessing(crop_size, downsample, n_classes)

        # Build the dataset object
        if repeats:
            print('Setting up dataset with {} repeats'.format(repeats))
            self.dataset = (tf.data.TFRecordDataset(record_path)
                            .repeat(repeats)
                            .map(preprocessing, 
                                num_parallel_calls=n_threads)
                            .prefetch(buffer_size=prefetch_buffer)
                            .batch(batch))
        else:
            print('Setting up dataset with infinite repeats')
            self.dataset = (tf.data.TFRecordDataset(record_path)
                            .repeat()
                            .shuffle(buffer_size=shuffle_buffer)
                            .prefetch(buffer_size=int(prefetch_buffer/2))
                            .map(preprocessing, 
                                num_parallel_calls=n_threads)
                            .prefetch(buffer_size=prefetch_buffer)
                            .batch(batch))
        
        if eager:
            # Eager iterator
            self.iterator = tfe.Iterator(self.dataset)
        else:
            self.iterator = self.dataset.make_initializable_iterator()
            self.x_op, self.y_op = self.iterator.get_next()

    def _decode(self, example):
        features = {'height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'width': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'img': tf.FixedLenFeature((), tf.string, default_value=''),
                    'mask': tf.FixedLenFeature((), tf.string, default_value=''), }
        pf = tf.parse_single_example(example, features)

        height = tf.squeeze(pf['height'])
        width = tf.squeeze(pf['width'])

        img = pf['img']
        mask = pf['mask']
        img = tf.decode_raw(img, tf.uint8)  ## Warning on hardcoded image types
        mask = tf.decode_raw(mask, tf.uint8)
        # img = tf.image.decode_image(img)
        # mask = tf.image.decode_image(mask)

        img = tf.cast(img, tf.float32)
        mask = tf.cast(mask, tf.float32)

        return height, width, img, mask

    def _build_preprocessing(self, crop_size, downsample, n_classes):
        """ Returns a function that reads the example

        For classification, read the mask and decide a class
        """
        def preproc_fn(example):
            h, w, img, mask = self._decode(example)
            img_shape = tf.stack([h, w, 3], axis=0)
            mask_shape = tf.stack([h, w, 1], axis=0)

            img = tf.reshape(img, img_shape)
            mask = tf.reshape(mask, mask_shape)

            # Glue them together to preserve orientation during rotate/flip
            image_mask = tf.concat([img, mask], axis=-1)
            image_mask = tf.random_crop(image_mask,
                [crop_size, crop_size, 4])
            image_mask = tf.image.random_flip_left_right(image_mask)
            image_mask = tf.image.random_flip_up_down(image_mask)
            img, mask = tf.split(image_mask, [3, 1], axis=-1)

            # Do color augmentations on image only
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.4, upper=0.7)
            img = tf.image.random_hue(img, max_delta=0.1)
            img = tf.image.random_saturation(img, lower=0.5, upper=0.7)

            # Get resizing args
            target_h = tf.cast(crop_size*downsample, tf.int32)
            target_w = tf.cast(crop_size*downsample, tf.int32)
            img = tf.image.resize_images(img, [target_h, target_w])
            ## method 1 = nearest neighbor
            ## Classification don't bother to resize the mask
            # mask = tf.image.resize_images(mask, [target_h, target_w], method=1) 

            # move to [0,1]
            img = tf.multiply(img, 1/255.)

            # change mask to a class
            uniques, _ , counts = tf.unique_with_counts(
                tf.squeeze(tf.reshape(mask, (1, -1))))
            majority = uniques[tf.argmax(counts)]

            # onehot mask
            majority = tf.cast(majority, tf.uint8)
            mask = tf.one_hot(majority, depth=n_classes)

            # Return ops
            return img, mask
        
        return preproc_fn