from __future__ import print_function
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (Dropout, Dense, Input)

from .encoder import make_encoder

def deep_feedforward(features, n_layers=5, width=256, dropout_rate=0.3):
  for k in range(n_layers):
    features = Dense(width, activation=tf.nn.relu, name='deep_{}'.format(k))(features)
    features = Dropout(dropout_rate, name='deep_drop_{}'.format(k))(features)

  return features

def Classifier(input_shape, n_classes=5, deep_classifier=True, encoder_args=None):
  image = Input(shape=input_shape)

  #input_shape needs to make its way into the encoder initialization:
  features = make_encoder(image=image, 
                          input_shape=input_shape, 
                          encoder_args=encoder_args)

  if deep_classifier:
    features = deep_feedforward(features)

  features = Dropout(0.3, name='classifier_dropout')(features)
  features = Dense(n_classes, activation=tf.nn.softmax, name='classifier')(features)

  model = tf.keras.Model(inputs=[image], outputs=[features])

  return model
