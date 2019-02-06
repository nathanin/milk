Code for "Deep Multiple Instance Learning identifies histological features of aggressive prostate cancer in needle biopsies"

In progress.

### Multiple Instance Learning Kit (MILK)

Structure
```
.
├── milk
│   ├── classifier.py
│   ├── densenet.py
│   ├── eager
│   │   ├── classifier.py
│   │   ├── densenet.py
│   │   ├── encoder.py
│   │   ├── __init__.py
│   │   └── mil.py
│   ├── encoder_config.py
│   ├── encoder.py
│   ├── __init__.py
│   ├── mil.py
│   ├── ops.py
│   ├── test_classifier.py
│   ├── test_densenet.py
│   ├── test_mil.py
│   └── utilities
│       ├── classification_dataset.py
│       ├── data_utils.py
│       ├── drawing_utils.py
│       ├── __init__.py
│       ├── mil_dataset.py
│       ├── model_utils.py
│       └── training_utils.py
├── README.md
├── requirements.txt
├── scripts
│   ├── attention
│   │   ├── attention_maps.py
│   │   ├── attimg.py
│   │   ├── batch_attention.sh
│   │   └── compare_attention.py
│   ├── cifar10
│   │   ├── cifar2tfrecord.py
│   │   ├── data_util.py
│   │   └── train.py
│   ├── dataset
│   │   ├── combine_npy_by_case.py
│   │   ├── __init__.py
│   │   ├── reduce_and_obfuscate.py
│   │   ├── remove_white_tiles.py
│   │   ├── slide_to_npy.py
│   │   └── slide_to_tfrecord.py
│   ├── debugging
│   │   ├── test_classifier_class.py
│   │   ├── test_data_utils.py
│   │   └── test_milk_class.py
│   ├── deploy
│   │   ├── attention_maps.py
│   │   ├── attimg.py
│   │   ├── deploy.py
│   │   ├── deploy.sh
│   │   ├── deploy_usable_area.py
│   │   ├── notebooks
│   │   │   └── km_plot.ipynb
│   │   ├── prad_clinical_data.csv
│   │   ├── README.md
│   │   ├── run_usable_area.sh
│   │   ├── test_svs.py
│   │   └── workup_tcga_clinical_data.py
│   ├── experiment
│   │   ├── batch_hyperparams.sh
│   │   ├── batch_test.py
│   │   ├── batch_test.sh
│   │   ├── batch_train.sh
│   │   ├── encoder_config.py
│   │   ├── gather_results.py
│   │   ├── readme.md
│   │   ├── searchargs.sh
│   │   ├── test_eager.py
│   │   ├── test_npy.py
│   │   ├── test_svs.py
│   │   ├── train_eager.py
│   │   ├── train_tpu_inmemory.py
│   │   ├── train_tpu.py
│   │   └── train_v0.py
│   ├── imagenet
│   │   ├── build_imagenet_data.py
│   │   ├── data_util.py
│   │   ├── encoder_config.py
│   │   ├── make_imagenet_tfrecords.py
│   │   ├── preprocess_imagenet.py
│   │   └── train.py
│   ├── misc
│   │   ├── image_cloud.ipynb
│   │   └── image_cloud.py
│   ├── mnist
│   │   ├── encoder_config.py
│   │   ├── pretrain_mnist.py
│   │   ├── readme.md
│   │   ├── train_mnist_eager.py
│   │   └── train_mnist.py
│   ├── pretraining
│   │   ├── profile_tfrecord_dataset.py
│   │   ├── projection.py
│   │   ├── test.py
│   │   ├── test_svs.py
│   │   ├── train_eager.py
│   │   ├── train_graph.py
│   │   └── train_tpu.py
│   ├── README.md
│   ├── usable_area
│   │   ├── classify_folder.py
│   │   ├── compare_fg_classified.py
│   │   ├── create_tfhub_training.py
│   │   ├── deploy_retrained.py
│   │   ├── dump_fgimg.py
│   │   ├── retrain.py
│   │   ├── run_deploy.sh
│   │   ├── run_retrain.sh
│   │   └── write_heatmap.py
│   └── visualizations
│       ├── attention_maps.py
│       ├── imprint_high_attention.py
│       ├── projections.py
│       ├── recolor_projections.py
│       └── z_space_projections.py
└── setup.py

```

#### Requirements
Please refer to `requirements.txt`

#### Setup
```
pip install -e .
```

#### MNIST Bags
MNIST Bags demonstrates the "positive bag" problem.
We have sets of `n` MNIST digits, with one particular digit designated as the "positive" one.
If >=1 examples of this digit is present in a bag, we consider the bag positive.
Else, the bag is negative.
The task is to build a classifier using only the **bag level** annotations of positive and negative.
That is, we have no knowledge of the labels for the individual examples composing each bag.
See [Ilse, et al (2018)](https://arxiv.org/abs/1802.04712).

We can make the problem easier by initializing the `encoder` CNN to be a straight up MNIST classifier.

Run:

```
scripts/mnist/pretrain_mnist.py -o scripts/mnist/pretrained.h5
scripts/mnist/train.py -o scripts/mnist/bagged_mnist.h5 --pretrained scripts/mnist/pretrained.h5
```


#### Milestones
- **Make the whole thing run on TPU.** 
  - Progress: see branch `functional-api` for a version compatible with TPU execution. 
  - Update: reworked `master` to use the functional api. Branch `functional-api` is now defunct.
  - See it run the bagged MNIST example on [Google Colab](https://colab.research.google.com/drive/1eOcZaqQG01fS16ckn9x94ivW-k12fbcg). 

 
Contact: ing.nathany@gmail.com
