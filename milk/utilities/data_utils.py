"""
To replace 'MultiNucleusData' class

https://stackoverflow.com/questions/47086599/parallelising-tf-data-dataset-from-generator

All written in numpy (and OpenCV)

To use:

```
import data_utils
transform_fn = data_utils.make_transform_fn(...)
train_list, test_list = data_utils.list_data(path, test_pct=0.2)

train_generator = lambda: data_utils.generator(train_list)
test_generator = lambda: data_utils.generator(test_list)

def load_bag(path):
    data, label = data_utils.load(path, transform_fn=transform_fn, min_bag=25, max_bag=50)
    transformed_data = data_utils.transform(subsetted_data)
    return transformed_data, label

def pyfunc_wrapper(path):
    return tf.contrib.eager.py_func(
        func = load_bag,
        inp  = [path],
        Tout = [tf.float32, tf.float32])
```

"""
from __future__ import print_function
import numpy as np
import cv2
import os
import glob
import re
import time


def as_one_hot(y):
    onehot = np.zeros((len(y), 2), dtype=np.float32)
    onehot[np.arange(len(y)), y] = 1
    return onehot

def split_train_test(data_list, test_pct=0.2):
    """ Stochastically split a list into two groups

    For each sample draw from a binomial distribution paramaterized by
    `test_pct`. If 1, then it's sorted into the test list. Otherwise
    it's sorted into the training list.

    Args:
        data_list: A list
        test_pct: float, default 0.2

    Returns:
        [train_list, test_list]

    """
    total_cases = len(data_list)
    aim_test = int(total_cases * test_pct)
    print('Aiming for {} train / {} test'.format(
        total_cases - aim_test,
        aim_test ))
    train_list = []
    test_list = []
    for val in data_list:
        if np.random.binomial(1, test_pct):
            test_list.append(val)
        else:
            train_list.append(val)

    print('Ranomly split {} train / {} test'.format(
        len(train_list), len(test_list)))
    return train_list, test_list

def split_train_test_balanced(data_list, test_pct=0.2):
    """ Split list into two preserving the class composition

    """
    pass

def split_train_val_test(data_list, val_pct=0.2, test_pct=0.2):
    total_cases = len(data_list)
    aim_test = int(total_cases * test_pct)
    print('Aiming for {} train / {} test'.format(
        total_cases - aim_test,
        aim_test ))
    train_list = []
    test_list = []
    for tst in data_list:
        if np.random.binomial(1, test_pct):
            test_list.append(tst)
        else:
            train_list.append(tst)

    print('Ranomly split {} train / {} test'.format(
        len(train_list), len(test_list)))

    print('Splitting train --> train/val')
    aim_val = int(len(train_list) * val_pct)
    print('Aiming for {} validation cases'.format(aim_val))

    train_list_out = []
    val_list = []
    for val in train_list:
        if np.random.binomial(1, val_pct):
            val_list.append(val)
        else:
            train_list_out.append(val)

    print('Randomly split {} train / {} val'.format(
        len(train_list_out), len(val_list)))

    return train_list_out, val_list, test_list

def list_data(data_patt, val_pct=0.2, test_pct=0.2):
    data_list = glob.glob(data_patt)
    return split_train_val_test(data_list, val_pct=val_pct, test_pct=test_pct)

"""
generator enforces supports batch_size = 1 even if batch_size is set
"""
def generator(data_list, batch_size=1):
    while True:
        yield np.random.choice(data_list, batch_size, replace=False)[0]

def exhaustable_generator(data_list):
    for data in data_list:
        yield data

def apply_transform(transform_fn, batch_x):
    batch_x = [transform_fn(x_) for x_ in batch_x]
    batch_x = np.stack(batch_x, axis=0)
    return batch_x

def load(data_path, 
         transform_fn=None, 
         min_bag=3, 
         max_bag=None, 
         const_bag=None,
         use_mmap=True,
         case_label_fn=None):
    """ Read a bag of examples from npy file 

    Assume the image-like examples for one data point are stored as a (bag, h, w, c) tensor.
    Determine the number of examples to use, between [min_bag, max_bag], (or const_bag)
    then return a random sample from the bag.

    Args:
        data_path: (str) Required. path to .npy for reading.
        transform_fn: (function) Default None
            A preprocessing function, returned by `make_transform_fn`
        min_bag: (int) Default 3
        max_bag: (int) Default None
        const_bag: (int) Default None
        use_mmap: (bool) Default True
            Whether or not to use numpy's mmap_mode argument for reading large arrays.
            see: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.load.html
        case_label_fn: (function) Default None


    Returns:
        numpy tensor: 
    """
    if not isinstance(data_path, str):
        data_path = data_path.decode('utf-8') ## New after update to py3.6 & TF 1.11
    if case_label_fn is not None:
        y_ = case_label_fn(data_path)
    else:
        raise Exception('Must supply `case_label_fn')

    # case = re.findall(CASE_PATT, data_path)[0]
    # y_ = CASE_LABEL_DICT[case]

    if use_mmap:
        batch_x = np.load(data_path, mmap_mode='r')
    else: 
        batch_x = np.load(data_path)

    bag_limit = batch_x.shape[0]
    if max_bag is None:
        max_bag = bag_limit

    # First check the upper limit is less than the number of examples
    if max_bag > bag_limit:
        # print('data_utils.load Warning. MAX_BAG ({}) > data size ({})'.format(max_bag, bag_limit), end='\t')
        # print('Setting MAX_BAG = data size/2 ({})'.format(data_path))
        max_bag = int(bag_limit/2)

    # Then make sure the lower limit is somewhere less than max. 
    if min_bag >= max_bag:
        # print('data_utils.load Warning. MIN_BAG ({}) >= MAX_BAG ({})'.format(min_bag, max_bag), end='\t')
        # print('Setting MIN_BAG = MAX_BAG/2 ({})'.format(data_path))
        min_bag = int(max_bag / 2)

    if const_bag is not None:
        if const_bag > bag_limit:
            # print('Warning. CONST_BAG > BAG_LIMIT')
            const_bag = bag_limit
        else:
            n_sub = const_bag
    else:
        n_sub = np.random.randint(min_bag, max_bag)

    sub_indices = np.random.choice(range(bag_limit), n_sub, replace=False)

    # If using mmap_mode, here data is actually read from disk
    batch_x = batch_x[sub_indices, ...]
    batch_x = apply_transform(transform_fn, batch_x)
    ## Add batch dimension:
    batch_x = np.expand_dims(batch_x, 0)
    y_ = np.expand_dims(y_, 0)
    return batch_x, as_one_hot(y_)

def make_transform_fn(height, width, crop_size, scale=1.0, normalize=False):
    """ Return a series of chained numpy and open-cv functions
    
    """
    # batch_x = np.squeeze(batch_x)
    # height, width = batch_x.shape[1:3]
    max_crop_h = height-crop_size
    max_crop_w = width-crop_size

    def _center_crop(x_):
        hh = int((height / 2) - (crop_size / 2))
        ww = int((width / 2) - (crop_size / 2))
        return x_[hh:hh+crop_size, ww:ww+crop_size, :]

    def _random_crop(x_):
        hh = int(np.random.uniform(low=0, high=max_crop_h))
        ww = int(np.random.uniform(low=0, high=max_crop_w))
        return x_[hh:hh+crop_size, ww:ww+crop_size, :]

    def _resize_fn(x_):
        dtarget = int(crop_size * scale)
        x_ = cv2.resize(x_, dsize=(dtarget, dtarget))
        return x_

    def _flip_fn(x_):
        flip_type = np.random.choice([-1,0,1])
        flipped_x = cv2.flip(x_, flip_type)
        return flipped_x

    def _rotate_fn(x_):
        theta = int(np.random.uniform(low=0, high=90))
        M = cv2.getRotationMatrix2D((crop_size/2,crop_size/2),theta,1)
        rotated_x = cv2.warpAffine(x_, M,(crop_size, crop_size))
        return rotated_x

    def _rotate_90(x_):
        theta = np.random.choice([0., 90., 180.])
        M = cv2.getRotationMatrix2D((crop_size/2,crop_size/2),theta,1)
        rotated_x = cv2.warpAffine(x_, M,(crop_size, crop_size))
        return rotated_x

    def _zero_center(x_):
        return (x_ * (1/255.)).astype(np.float32)
        # return (x_ * (2/255.) - 1).astype(np.float32)

    def _chained_fns(x_):
        x_ = _random_crop(x_)
        x_ = _rotate_90(x_)
        x_ = _resize_fn(x_)
        x_ = _flip_fn(x_)
        if normalize:
            x_ = reinhard(x_)
        x_ = _zero_center(x_)
        return x_

    return _chained_fns

def reinhard(image, target=None):
    """ Reinhard color normalization
    Color transfer towards a reference mean & standard deviation.

    @article{reinhard2001color,
    title={Color transfer between images},
    author={Reinhard, Erik and Adhikhmin, Michael and Gooch, Bruce and Shirley, Peter},
    journal={IEEE Computer graphics and applications},
    volume={21},
    number={5},
    pages={34--41},
    year={2001},
    publisher={IEEE}
    }

    Args:
        image: 3-channel RGB space image
        target: 3x2 np.ndarray float64

    Returns: 
        returnimage: uint8 np.ndarray
    """
    ## Default target, as in Gertych, et al 2015
    if target is None:
        target = np.array([[148.60, 41.56], [169.30, 9.01], [105.97, 6.67]])

    M, N = image.shape[:2]

    whitemask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    whitemask = whitemask > 215 ## TODO: Hard code threshold; replace with Otsu

    if whitemask.sum() > 0.8*M*N:
        ## All whilte; skip
        return image

    imagelab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    imageL, imageA, imageB = cv2.split(imagelab)

    # mask is valid when true
    imageLM = np.ma.MaskedArray(imageL, whitemask)
    imageAM = np.ma.MaskedArray(imageA, whitemask)
    imageBM = np.ma.MaskedArray(imageB, whitemask)

    ## Sometimes STD is near 0, or 0; add epsilon to avoid div by 0 -NI
    epsilon = 1e-12

    imageLMean = imageLM.mean()
    imageLSTD = imageLM.std() + epsilon

    imageAMean = imageAM.mean()
    imageASTD = imageAM.std() + epsilon

    imageBMean = imageBM.mean()
    imageBSTD = imageBM.std() + epsilon

    # normalization in lab
    imageL = (imageL - imageLMean) / imageLSTD * target[0][1]  + target[0][0]
    imageA = (imageA - imageAMean) / imageASTD * target[1][1]  + target[1][0]
    imageB = (imageB - imageBMean) / imageBSTD * target[2][1]  + target[2][0]

    imagelab = cv2.merge((imageL, imageA, imageB))
    imagelab = np.clip(imagelab, 0, 255)
    imagelab = imagelab.astype(np.uint8)

    # Back to RGB space
    returnimage = cv2.cvtColor(imagelab, cv2.COLOR_LAB2RGB)
    # Replace white pixels
    returnimage[whitemask] = image[whitemask]

    return returnimage
