from __future__ import print_function
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (Dropout, Dense, Input, BatchNormalization)
from .encoder import make_encoder

""" Same fn as in mil.py """
def deep_feedforward(features, n_layers=5, width=256, dropout_rate=0.3):
  for k in range(n_layers):
    features = Dense(width, activation=tf.nn.relu, name='classifier_{}'.format(k))(features)
    features = Dropout(dropout_rate, name='classifier_drop_{}'.format(k))(features)
  return features

def Classifier(input_shape, n_classes=5, encoder_args=None, deep_classifier=True):
  image = Input(shape=input_shape)

  #input_shape needs to make its way into the encoder initialization:
  features = make_encoder(image=image, 
                          input_shape=input_shape, 
                          encoder_args=encoder_args)
  features = BatchNormalization(momentum=0.99, trainable=True, axis=-1, name='classifier_bn')(features)
  features = Dropout(0.3, name='classifier_dropout')(features)

  if deep_classifier:
    features = deep_feedforward(features)

  logits = Dense(n_classes, activation=tf.nn.softmax, name='classifier')(features)
  model = tf.keras.Model(inputs=[image], outputs=[logits])

  return model
