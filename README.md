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
|____ experiment/
      |____ batch_train.sh
      |____ batch_test.sh
      |____ train_tpu.py
      |____ test_npy.py
|____ usable_area/

```

#### Requirements
Please refer to `requirements.txt`

#### Goal
- **Make the whole thing run on TPU.** 
  - Progress: see branch `functional-api` for a version compatible with TPU execution. See it run on [Google Colab](https://colab.research.google.com/drive/1eOcZaqQG01fS16ckn9x94ivW-k12fbcg). The next thing to do is update the main experiment.

Date: October 25, 2018
Contact: Nathan.Ing@cshs.org , ing.nathany@gmail.com
