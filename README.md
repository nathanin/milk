Code for "Deep Multiple Instance Learning identifies histological features of aggressive prostate cancer in needle biopsies"

In progress.

#### Multiple Instance Learning Kit (MILK)

Structure
```
milk/
|____utilities/
      |____ __init__.py
      |____ drawing_utils.py
      |____ data_utils.py
      |____ model_utils.py
      |____ training_utils.py
|____ __init__.py
|____ classifier.py
|____ densenet.py
|____ encoder.py
|____ ops.py
|____ mil.py

scripts/
|_____dataset/
      |____ __init__.py
      |____ data/
      |____ remove_white_tiles.py
      |____ slide_to_npy.py
      |____ slide_to_tfrecord.py
|____ debugging/
|____ deploy/
|____ evaluate/
|____ pretraining/
|____ training/
|____ usable_area/

```

#### Requirements
Please refer to `requirements.txt`

#### Goal
- **Make the whole thing run on TPU.** With the current support for Keras models on TPU, this might require a rewrite away from the subclassing pattern into the `Sequential()` pattern or `keras.Model(inputs=[...], outputs=[...])` pattern. ([REF](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/tpu/python/tpu/keras_support.py)). I'm not sure the DenseNet constructor is fully compatable with keras-on-TPUs. Regardless, we probably need to restrict tensors to constant size (just set `const_bag` and reject bags with too few instances -- we can use them for testing), but can take advantage of the innate TPU graph duplication to run batches of 8, instead of 1, without further alterations. Input might get tricky; Will probably have to host images in a google cloud bucket, although the keras support doc says to use numpy arrays.

Date: October 25, 2018
Contact: Nathan.Ing@cshs.org , ing.nathany@gmail.com
