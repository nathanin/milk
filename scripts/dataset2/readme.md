## New data classes based on hdf5 

Data is kept in stacks extracted from slides, according to the case ID

### The structure is:

``` markdown
# Descriptive filename
pnbx-10x

# description of the dataset
/metadata/...

# cases listed; append cases as needed
/group 1/image stack
/group 2/image stack
...

```

### Keep the keys in a separate file:
``` markdown
# Matching filename + keys
pnbx-10x-keys

# Key type / cases match cases in main data set
/label 1/group 1/value
        /group 2/value
        /group 3/value

/label 2/group 1/value
        /group 2/value
        /group 3/value
```


### And write a python wrapper to interface:

``` python
class Dataset:
  """
  args:
    data_path
    key_path
    mode: read or write

  attributes:
    data_groups (see get_groups)
    key_groups (see get_groups)

    preproc_fn (supplied, with default):
      - apply this function when loading each stack
      - args: image_stack (stack, h, w, c)
      - returns: image_stack (stack, h, w, c)

  methods:
    init: 
      - set the dataset and keys
      - check dataset and keys for reading
      - check dataset and keys for matching groups
      - parse a mode string
      - build iterators

    get_groups:
      - pulls out list of matching groups from sources
      - populate two dictionaries:
        1. data_groups = { 'keys': hooks to data }
        2. key_groups = { 'keys: hooks to key values }

    read_group:
      args: group
      - read the stack data into memory according to rules
      - apply the preproc_fn() to the stack
      - return the 4D stack as np.float32
      - raise an alarm if anything goes wrong        

    check_reading: 
      - read a sample group from both sources using read_group
      - raise an alarm if something goes wrong
      - include a test data piece as an integrity check ?

    tensorflow_iterator:
      - build up a tensorflow dataset iterator using some options
        --> copy svsutils
    
    append_group:
      - append data from memory to the open file as a new dataset/group 
      - append the corresponding key too. 

    write_group:
      - replace data in a group with data from memory 
      - replace the corresponding key too. 
  """
```

