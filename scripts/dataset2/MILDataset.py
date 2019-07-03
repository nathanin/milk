import numpy as np
import h5py

import os
import sys
import glob

def create_dataset(data_path, meta_string, seed=999, clobber=False):
  """
  Initialize the dataset and the keys dataset.
  - Create the metadata dataset and write the `meta_string`
  - Create the a test dataset we can use for data integrity.
  """
  mode = 'w' if clobber else 'w-'
  meta_string_enc = np.string_(meta_string)
  np.random.seed(seed)
  rand_data = np.random.normal(size=(5,5))

  with h5py.File(data_path, mode) as data_f:
    data_f.create_dataset("metadata", data=meta_string_enc)
    intg = data_f.create_dataset("integrity", data=rand_data)
    intg.attrs['seed'] = seed

  print('Done. Created:')
  print(data_path)


class MILDataset:
  def __init__(self, data_path, preproc_fn=lambda x: x,
               mode='a'):
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
    self.data_group_names = [name for name in self.data_h5]
    self.data_groups = {name: self.data_h5[name] for name in self.data_h5}

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
      idx = np.random.choice(idx, size=subset, replace=False)

    x_ = data_[idx, ...]
    label_ = data_.attrs[attr]
    return x_, label_

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

  def tensorflow_iterator(self):
    """
    tensorflow_iterator:
      - build up a tensorflow dataset iterator using some options
        --> copy from svsutils.iterators
    """
    pass
    
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
        (TODO : multi-key / group add)
      attr_value: some label
      chunks: chunk size for the data
        by default let h5py guess how to chunk it.
        maybe it's best to set chunk size to (1, img_h, img_w, channels)
        since we'll always want to read the whole image anyway
    """
    data_ = self.data_h5.create_dataset(dataset_name, data=data, 
      chunks=chunks, compression=compression)
    data_.attrs[attr_type] = attr_value

    # Update the groups
    self.get_groups()

