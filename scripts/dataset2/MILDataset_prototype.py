from tqdm import tqdm
import numpy as np
import h5py

import os
import sys
import glob
import contextlib

import tensorflow as tf

# https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
@contextlib.contextmanager
def temp_seed(seed):
  state = np.random.get_state()
  np.random.seed(seed)
  try:
    yield
  except:
    np.random.set_state(state)


def create_dataset(data_path, meta_string, mode=None, seed=999, clobber=False):
  """
  Initialize the dataset and the keys dataset.
  - Create the metadata dataset and write the `meta_string`
  - Create the a test dataset we can use for data integrity.
  """
  # mode = 'w' if clobber else 'w-'
  # Detect existing file and selectively handle it
  # prefer appending.
  meta_string_enc = np.string_(meta_string)
  np.random.seed(seed)
  rand_data = np.random.normal(size=(5,5))

  mode = mode if mode is not None else 'a'
  with h5py.File(data_path, mode) as data_f:
    data_f.create_dataset("metadata", data=meta_string_enc)
    intg = data_f.create_dataset("integrity", data=rand_data)
    intg.attrs['seed'] = seed

  print('Done.')


class MILDataset:
  def __init__(self, data_path, meta_string='', preproc_fn=lambda x: x,
               mode='a', crop=128, n_classes=2 ):
    """
      - set the dataset and keys
      - check dataset and keys for reading
      - check dataset and keys for matching groups
      - parse a mode string
      - build iterators

    Note about building libhdf5 with thread-safety for concurrent reading:
    https://portal.hdfgroup.org/display/knowledge/Questions+about+thread-safety+and+concurrent+access

    Therefore, we'll load data in serial and perform the preprocessing in parallel.
    This may create some overhead but hey it is.
    """
    # Some defaults
    self.data_path = data_path
    self.preproc_fn = preproc_fn 
    self.meta_string = meta_string
    self.default_fields = ['integrity', 'metadata']

    # Settings for preprocessing
    self.crop = crop
    self.n_classes = n_classes

    self.pct_tr, self.pct_v, self.pct_te = 0.6, 0.2, 0.2

    if not os.path.exists(data_path):
      print('Creating dataset at {}'.format(data_path))
      create_dataset(data_path, meta_string)

    self.data_h5 = h5py.File(data_path, mode)

    self.get_groups()
    self.check_integrity()

  def get_groups(self):
    """
      - pulls out list of matching groups from sources
      - populate two lists:
        1. data_group_names 
        2. key_group_names
      - populate two dictionaries:
        1. data_groups = { 'keys': hooks to data }
        2. key_groups = { 'keys: hooks to key values }
    """
    self.data_group_names = [name for name in self.data_h5 \
        if name not in self.default_fields]
    self.data_groups = {name: self.data_h5[name] for name in self.data_h5}

    print('group names: {} groups: {}'.format(
      len(self.data_group_names),
      len(self.data_groups)))

  def split_train_val(self, seed1=999, seed2=111):
    """
    split_train_val:
      - partition the dictionary keys into 3 groups
    
    After calling this method we will have lists to 
    generate from.

    Favor adding extras to training.
    
    Only have to work with names, everything gets looked up
    in self.data_groups anyway.
    """
    print('\nSplitting train / val / test')
    n = len(self.data_group_names)
    n_tr = np.int(n * self.pct_tr)
    n_v = np.int(n * self.pct_v)
    print(n, n_tr, n_v)

    # Shuffle for splitting
    # np.random.seed(seed1)
    names = sorted(self.data_group_names)
    with temp_seed(seed1):
      np.random.shuffle(names)

    self.tr_datasets = sorted(names[:n_tr])
    self.v_datasets  = sorted(names[n_tr:n_tr+n_v])
    self.te_datasets = sorted(names[n_tr+n_v:])
    print('Train: {} Val: {} Test: {}'.format(
          len(self.tr_datasets),
          len(self.v_datasets),
          len(self.te_datasets)))
    
    # Shuffle order
    # np.random.seed(seed2)
    with temp_seed(seed2):
      print('Train first 3:', self.tr_datasets[:3])
      np.random.shuffle(self.tr_datasets)
      np.random.shuffle(self.v_datasets)
      np.random.shuffle(self.te_datasets)
      print('After shuffle', self.tr_datasets[:3])

  def read_data(self, dataset, attr, subset=None):
    """
    ( Reads happen in serial to not require thread-safe libhdf5 )
    read_group:
      args: group
      - read the stack data into memory according to rules
      - apply the preproc_fn() to the stack
      - return the 4D stack as np.float32
      - raise an alarm if anything goes wrong        
    """
    data_ = self.data_h5[dataset]
    n = data_.shape[0]
    idx = np.arange(n)
    if subset is not None:
      s = subset if subset < n else n
      idx = sorted(np.random.choice(idx, size=s, replace=False))

    x_ = data_[idx, ...]
    label_ = data_.attrs[attr]
    return x_, label_

  def python_iterator(self, mode='train', subset=100, attr='stage_code', seed=999):
    """
    python_iterator:
      - return an iterator over the contents of the open dataset 
      - use separate lists for train-test-val

    ! these are in random order
    """
    self.split_train_val(seed1=seed, seed2=np.random.randint(low=100, high=99999))
    
    if mode == 'train':
      lst = self.tr_datasets
    elif mode == 'val':
      lst = self.v_datasets
    else:
      lst = self.te_datasets

    for i, name in enumerate(lst):
      data, label = self.read_data(name, attr, subset=subset)
      yield data, label


  def tensorflow_iterator(self, preproc_fn=None, batch_size=1, 
                          buffer_size=32, threads=8, eager=True,    # iterator arg .. for what? 
                          mode='train', subset=100, seed=999, 
                          attr='stage_code'):
    """
    tensorflow_iterator:
      - build a tensorflow dataset iterator using some options
        --> copy from svsutils.iterators
        --> also copy from old data_utils...

    """

    @tf.contrib.eager.defun
    def tf_preproc_fn(x4d):
      # multiprocessing for individual images - we can apply any tf / python code here.
      def _internal(x3d):
        x = tf.image.random_crop(x3d, (self.crop, self.crop, 3))
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_flip_left_right(x)
        return x
      xproc = tf.map_fn(_internal, x4d, parallel_iterations=threads, back_prop=False)
      return xproc

    if preproc_fn is None:
      preproc_fn = tf_preproc_fn

    py_it = lambda : self.python_iterator(mode=mode, subset=subset, attr=attr, seed=seed)
    def map_fn(x,y):
      x = tf.cast(x, tf.float32) / 255.
      # x = tf.py_function(preproc_fn, inp=[x], Tout=(tf.float32))
      x = preproc_fn(x)
      y = tf.one_hot(tf.squeeze(y), self.n_classes)
      return x, y

    dataset = tf.data.Dataset.from_generator(py_it, output_types=(tf.uint8, tf.int64))
    dataset = dataset.prefetch(buffer_size)
    dataset = dataset.map(map_fn, num_parallel_calls=threads)
    dataset = dataset.prefetch(buffer_size)
    dataset = dataset.batch(batch_size)

    # if eager:
    tf_it = dataset.make_one_shot_iterator()
    return tf_it
    # else:
    #   return dataset

    
  def new_dataset(self, dataset_name, data, attr_type, attr_value, 
                  chunks=True, compression='gzip'):
    """
    new_dataset:
      - write a dataset from memory 
      - write the corresponding key entry 

    arguments:
      dataset_name: unique ID for this dataset
      data: the data
      attr_type: the element in data_.attr to store the value of key 
        (TODO : multi-attr add)
      attr_value: some label
      chunks: chunk size for the data
        by default let h5py guess how to chunk it.
        maybe it's best to set chunk size to (1, img_h, img_w, channels)
        since we'll always want to read the whole image anyway
    """
    data_ = self.data_h5.create_dataset(dataset_name, data=data, 
      chunks=chunks, compression=compression)
    
    if isinstance(attr_type, list):
      for at, av in zip(attr_type, attr_value):
        data_.attrs[at] = av
    else:
      data_.attrs[attr_type] = attr_value

    # Update the groups
    self.get_groups()


  def check_integrity(self, seed=999): 
    """
    check_integrity: 
      - read the integrity group from both sources using read_group
      - raise an alarm if something goes wrong
    """
    print('Checking sample data')
    np.random.seed(seed)
    chk = np.random.normal(size=(5,5))
    data_integ = self.data_groups['integrity'][:]
    assert np.isclose(chk, data_integ).all()
    print('Check passed.')
