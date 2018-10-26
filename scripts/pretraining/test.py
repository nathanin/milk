from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import os
import glob

config=tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

import seaborn as sns
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '.')
from segmentation_model import EagerSegmentation
from train import get_nested_variables

DATA = '../../dataset/tiles/pruned_no_white/*.npy'
data_list = glob.glob(DATA)
print(len(data_list))

ENCODER_SNAPSHOT = './trained/eager_segmentation-13001'

X_SIDE = 128

sample_data = np.load(data_list[0])

x = sample_data[:16, :X_SIDE, :X_SIDE, :]
print(x.shape)

with tf.device('/gpu:0'):
    model = EagerSegmentation()
    # Run once to initialize variables
    x = x * (1./255)
    yhat = model(tf.constant(x), verbose=True)
    
variables = get_nested_variables(model)
saver = tfe.Saver(variables)

saver.restore(ENCODER_SNAPSHOT)