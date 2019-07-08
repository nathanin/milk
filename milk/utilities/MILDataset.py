"""
Implements an HDF5 based dataset

The HDF5 file will be organized with the root group
and datasets corresponding to sets

Provide reference iterators in vanilla python
and tf.data.Dataset based loading / preprocessing.

assume libhdf5 is not thread-safe, and we can only have one 
thread read from the h5 file at a time.

"""
from tqdm import tqdm
import numpy as np
import h5py

import os
import sys
import glob
import contextlib
import itertools
import gc

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

    self.pct_tr, self.pct_v, self.pct_te = 0.7, 0.1, 0.2

    if not os.path.exists(data_path):
      print('Creating dataset at {}'.format(data_path))
      create_dataset(data_path, meta_string)

    self.data_h5 = h5py.File(data_path, mode)

    self.get_groups()
    self.check_integrity()

    ## Define a few functions for downstream use
    # @tf.contrib.eager.defun
    def tf_preproc_fn(x4d):
      # multiprocessing for individual images - we can apply any tf / python code here.
      def _internal(x3d):
        x = tf.image.random_crop(x3d, (self.crop, self.crop, 3))
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_flip_left_right(x)
        return x
      xproc = tf.map_fn(_internal, x4d, parallel_iterations=8)
      return xproc

    @tf.contrib.eager.defun
    def map_fn(x,y):
      with tf.device('/cpu:0'):
        x = tf.cast(x, tf.float32) / 255.
        x = tf_preproc_fn(x)
        y = tf.one_hot(tf.squeeze(y), self.n_classes)
      return x, y

    self.map_fn = map_fn




  def get_groups(self):
    """
      - pulls out list of matching groups from sources
      - populate two lists:
        1. data_group_names 
        2. key_group_names
      - populate two dictionaries:
        1. data_hooks = { 'keys': hooks to data }
        2. key_groups = { 'keys: hooks to key values }
    """
    self.matching_datasets = [name for name in self.data_h5 \
        if name not in self.default_fields]
    self.data_hooks = {name: self.data_h5[name] for name in self.data_h5 \
       if name not in self.default_fields}
    self.special_hooks = {name: self.data_h5[name] for name in self.default_fields}

    print('group names: {} groups: {}'.format(
      len(self.matching_datasets),
      len(self.data_hooks)))


  def split_train_val(self, groupby, seed=999):
    """
    split_train_val:
      - partition the dictionary keys into 3 groups
    
    After calling this method we will have lists to 
    generate from.

    Favor adding extras to training.
    
    Only have to work with names, everything gets looked up
    in self.data_hooks anyway.
    """
    print('\nSplitting train / val / test by field {}'.format(groupby))
    ds_attr_vals = { name: ds.attrs[groupby] for name, ds in self.data_hooks.items() }
    unique_vals = np.unique([v for _, v in ds_attr_vals.items()])
    # print(unique_vals)

    grouped_sets = {u: [] for u in unique_vals}

    for name, v in ds_attr_vals.items():
      grouped_sets[v].append(name)

    # for name, s in grouped_sets.items():
    #   print(name)
    #   print(s)

    n = len(unique_vals)
    n_tr = np.int(n * self.pct_tr)
    n_v = np.int(n * self.pct_v)
    # print(n, n_tr, n_v)

    # Shuffle for splitting
    # np.random.seed(seed1)
    names = sorted(unique_vals)
    with temp_seed(seed):
      np.random.shuffle(names)

    self.tr_datasets = [grouped_sets[n] for n in names[:n_tr]] 
    self.v_datasets  = [grouped_sets[n] for n in names[n_tr:n_tr+n_v]] 
    self.te_datasets = [grouped_sets[n] for n in names[n_tr+n_v:]] 
    self.tr_datasets = list(itertools.chain.from_iterable(self.tr_datasets))
    self.v_datasets = list(itertools.chain.from_iterable(self.v_datasets))
    self.te_datasets = list(itertools.chain.from_iterable(self.te_datasets))

    self.dataset_lengths = {
      'train': len(self.tr_datasets),
      'val': len(self.v_datasets),
      'test': len(self.te_datasets),
    }
    # print('Train: {} Val: {} Test: {}'.format(
    #       len(self.tr_datasets),
    #       len(self.v_datasets),
    #       len(self.te_datasets)))
    

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
    # data_ = self.data_h5[dataset]
    n = self.data_hooks[dataset].shape[0]
    idx = np.arange(n)
    if subset is not None:
      s = subset if subset < n else n
      idx = sorted(np.random.choice(idx, size=s, replace=False))

    x_ = self.data_hooks[dataset][...]
    x_ = x_[idx, ...]
    label_ = self.data_hooks[dataset].attrs[attr]
    return x_, label_


  def python_iterator(self, mode='train', subset=100, attr='stage_code', seed=999,
                      data_fn = lambda x: x , label_fn = lambda x: x):
    """
    python_iterator:
      - return an iterator over the contents of the open dataset 
      - use separate lists for train-test-val

    ! these are in random order
    """
    # Shuffle order
    # np.random.seed(seed2)
    with temp_seed(seed):
      # print('Train first 3:', self.tr_datasets[:3])
      np.random.shuffle(self.tr_datasets)
      np.random.shuffle(self.v_datasets)
      np.random.shuffle(self.te_datasets)
      # print('After shuffle', self.tr_datasets[:3])
    
    if mode == 'train':
      lst = self.tr_datasets
    elif mode == 'val':
      lst = self.v_datasets
    else:
      lst = self.te_datasets

    for name in lst:
      data, label = self.read_data(name, attr, subset=subset)
      try:
        label = label.astype(np.uint8)
      except:
        label = label

      data = data_fn(data)
      label = label_fn(label)
      yield data, label


  def python_name_iterator(self, mode='train', subset=100, attr='stage_code', seed=999):
    with temp_seed(seed):
      np.random.shuffle(self.tr_datasets)
      np.random.shuffle(self.v_datasets)
      np.random.shuffle(self.te_datasets)
    
    if mode == 'train':
      lst = self.tr_datasets
    elif mode == 'val':
      lst = self.v_datasets
    else:
      lst = self.te_datasets

    for name in lst:
      yield name

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

    def py_it():
      return self.python_iterator(mode=mode, subset=subset, attr=attr, seed=seed)

    # self.tf_dataset = self.tf_dataset.prefetch(buffer_size)
    self.tf_dataset = (tf.data.Dataset.from_generator(py_it, output_types=(tf.uint8, tf.uint8))
                       .map(self.map_fn, num_parallel_calls=threads)
                      #  .prefetch(threads*2)
                       .batch(batch_size))
    self.len_tf_ds = self.dataset_lengths[mode]
    # self.tf_dataset = self.tf_dataset.take(self.len_tf_ds)

  def clear_mem(self):
    self.tf_dataset = []
    self.len_tf_ds = []
    tf.reset_default_graph()
    gc.collect()

  def new_dataset(self, dataset_name, data, attr_type, attr_value, 
                  chunks=True, compression='lzf'):
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
    data_integ = self.special_hooks['integrity'][:]
    assert np.isclose(chk, data_integ).all()
    print('Check passed.')

  def close_refs(self):

    self.data_h5.close()

    for k, v in self.__dict__.items():
      self.__dict__[k] = None

    self.clear_mem()