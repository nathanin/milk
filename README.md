Code for "Deep Multiple Instance Learning identifies histological features of aggressive prostate cancer in needle biopsies"

In progress.

#### Multiple Instance Learning Kit (MILK)

Structure
```
# The main module
milk/
|____ utilities/
      |____ __init__.py
      |____ classification_dataset.py
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
|____ test_classifier.py
|____ test_densenet.py
|____ test_mil.py

# Scripts to run our experiments
scripts/
|____ attention/
      |____ attention_maps.py
      |____ compare_attention.py

|____ dataset/
      |____ __init__.py
      |____ remove_white_tiles.py
      |____ slide_to_npy.py
      |____ slide_to_tfrecord.py

|____ debugging/
      |____ test_classifier_class.py
      |____ test_data_utils.py
      |____ test_milk_class.py

|____ deploy/
      |____ deploy.py
      |____ deploy.sh
      |____ test_svs.py

|____ pretraining/
      |____ pretrained_reference.h5
      |____ profile_tfrecord_dataset.py
      |____ projection.py
      |____ test_svs.py
      |____ test.py
      |____ train_graph.py
      |____ train_tpu.py

|____ experiment/
      |____ batch_train.sh
      |____ batch_test.sh
      |____ train_tpu.py
      |____ test_npy.py

|____ usable_area/
      |____ classify_folder.py
      |____ create_tfhub_training.py
      |____ deploy_retrained.py
      |____ dump_fgimg.py
      |____ retrain.py
      |____ run_deploy.sh
      |____ run_retrain.sh
      |____ write_heatmap.py
      
|____ mnist/
      |____ pretrain_mnist.py
      |____ readme.md
      |____ train_mnist.py

```

#### Requirements
Please refer to `requirements.txt`

#### Milestones
- **Make the whole thing run on TPU.** 
  - Progress: see branch `functional-api` for a version compatible with TPU execution. See it run the "bagged MNIST" example on [Google Colab](https://colab.research.google.com/drive/1eOcZaqQG01fS16ckn9x94ivW-k12fbcg). The next thing to do is update the main experiment. 

Date: October 25, 2018
Contact: Nathan.Ing@cshs.org , ing.nathany@gmail.com
