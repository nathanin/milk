from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from .encoder import make_encoder_eager

class ClassifierEager(tf.keras.Model):
  def __init__(self, n_classes = 5, deep_classifier=True, hidden_dim=256, 
               encoder_args=None, dropout_rate=0.3):

    super(ClassifierEager, self).__init__()

    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout_rate

    self.encoder = make_encoder_eager( encoder_args = encoder_args )

    # This can be anything
    self.classifier_layers = []
    self.ff_dropout = Dropout(self.dropout_rate)
    if deep_classifier:
      depth = 5
    else:
      depth = 1

    for k in range(depth):
      self.classifier_layers.append(
        Dense(self.hidden_dim, activation=tf.nn.relu, use_bias=False, 
        name='cls_deep_{}'.format(k)))

    self.batchnorm1 = BatchNormalization(momentum=0.99, trainable=True, axis=-1, name='classifier_bn')
    self.dropout = Dropout(rate=self.dropout_rate, name='classifier_dropout')
    self.classifier = Dense(n_classes, activation=tf.nn.softmax, name='classifier')
    self.output_names = ['classifier']
    # self.output_shapes = [self.classifier.get_shape()]

  def call(self, x_in, verbose=False, return_features=False, training=True):
    features = self.encoder(x_in, training=training)
    features = self.batchnorm1(features)
    features = self.dropout(features, training=training)

    for layer in self.classifier_layers:
      features = layer(features) 
      features = self.ff_dropout(features, training=training)

    logits = self.classifier(features)

    if return_features:
      return logits, features
    else:
      return logits
