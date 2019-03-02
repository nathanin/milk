"""
MI-Net with convolutions
"""
from __future__ import print_function
import tensorflow as tf

from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization)

from .densenet import DenseNetEager
from .encoder import make_encoder_eager

from milk.utilities.model_utils import lr_mult

class MilkEager(tf.keras.Model):
  def __init__(self, z_dim=256, encoder_args=None, cls_normalize=True, mil_type='attention', deep_classifier=True, temperature=1):

    super(MilkEager, self).__init__()

    self.hidden_dim = z_dim
    self.deep_classifier = deep_classifier
    self.mil_type = mil_type
    self.built_fn = False
    self.temperature = temperature
    self.cls_normalize = cls_normalize

    self.densenet = make_encoder_eager( encoder_args = encoder_args )
    self.drop2  = Dropout(rate=0.3)

    if mil_type == 'attention':
      #self.att_batchnorm = BatchNormalization(trainable=True)
      self.attention = Dense(units=256, 
        activation=tf.nn.tanh, use_bias=False, name='attention')
      self.attention_gate = Dense(units=256, 
        activation=tf.nn.sigmoid, use_bias=False, name='attention_gate')
      self.attention_layer = Dense(units=1, 
        activation=None, use_bias=False, name='attention_layer')

    self.classifier_layers = []
    if self.deep_classifier:
      depth=5
    else:
      depth=1

    if self.cls_normalize:
      self.classifier_dropout_0 = Dropout(rate = 0.3)
      # self.classifier_dropout_1 = Dropout(rate = 0.3)
      self.classifier_bn = BatchNormalization(trainable=True, axis=-1) # Is this how we make it invariant to bag size?

    for i in range(depth):
      self.classifier_layers.append(
        Dense(units=self.hidden_dim, activation=tf.nn.relu, use_bias=False,
        name = 'mil_deep_{}'.format(i)))

    self.classifier = Dense(units=2, activation=tf.nn.softmax, use_bias=False, name='mil_classifier')

  def mil_attention(self, features, verbose=False, return_att=False, return_raw_att=False, training=True):
    # features = self.att_batchnorm(features, training=training)
    att = self.attention(features)
    att_gate = self.attention_gate(features)
    att = att * att_gate # tf.multiply()
    if verbose:
      print('attention:', att.shape)
    att = self.attention_layer(att)
    att = tf.transpose(att, perm=(1,0)) 
    if verbose:
      print('attention:', att.shape)
    if return_raw_att:
      att_ret = tf.identity(att)

    # https://en.wikipedia.org/wiki/Softmax_function#Smooth_arg_max
    att /= self.temperature

    # Question: WTF?
    # tensorflow.python.framework.errors_impl.InternalError: CUB segmented reduce errorinvalid configuration argument [Op:Softmax]
    # unless we put this op on CPU in eager mode.
    with tf.device('/cpu:0'):
      att = tf.nn.softmax(att, axis=1)

    if verbose:
      print('attention:', att.shape)
    z = tf.matmul(att, features)
    if verbose:
      print('features - attention:', z.shape)
    if return_att or return_raw_att:
      if not return_raw_att:
        att_ret = tf.identity(att)
      return z, att_ret
    else:
      return z

  def apply_mil(self, z, training=True, verbose=False):
    if self.mil_type == 'attention':
      z = self.mil_attention(z, training=training, verbose=verbose)
    # Add elif's here for more MIL or attention types
    else:
      z = tf.reduce_mean(z, axis=0, keep_dims=True)
    if verbose:
      print('z:', z.shape)
    return z

  def encode_bag(self, x_bag, batch_size=64, training=True, verbose=False, return_z=False):
    z_bag = []
    n_bags = x_bag.shape[0] // batch_size
    remainder = x_bag.shape[0] - n_bags*batch_size
    x_bag = tf.split(x_bag, tf.stack([batch_size]*n_bags + [remainder]), axis=0)
    for x_in in x_bag:
      z = self.densenet(x_in, training=training)
      if verbose:
        print('\t z: ', z.shape)
      z = self.drop2(z, training=training)
      if self.mil_type == 'instance':
        z = self.apply_classifier(z, verbose=verbose, training=training)
        if verbose:
          print('Instance yhat:', z.shape)
      z_bag.append(z)

    z = tf.concat(z_bag, axis=0)
    if verbose:
      print('\tz bag:', z.shape)
    #z = self.att_batchnorm(z, training=training)
    if return_z:
      return z
    else:
      if self.mil_type == 'instance':
        z = tf.reduce_mean(z, axis=0, keep_dims=True)
      else:
        z = self.apply_mil(z, verbose=verbose)
      return z

  def apply_classifier(self, features, verbose=False, training=True):
    if self.cls_normalize:
      features = self.classifier_bn(features, training=training)
      features = self.classifier_dropout_0(features, training=training)
    for layer in self.classifier_layers:
      features = layer(features)

    # features = self.classifier_dropout_1(features, training=training)
    yhat = self.classifier(features)
    return yhat

  #@tf.contrib.eager.defun
  def call(self, x_in, T=20, batch_size=64, 
           training=True, verbose=False,):
    """
    `training` controls the use of dropout and batch norm, if defined
    `return_embedding`
      prediction
      attention
      raw embedding (batch=num instances)
      embedding after attention (batch=1)
      classifier hidden layer (batch=1)
    """
    if verbose:
      print(x_in.shape)
    assert len(x_in.shape) == 5
    n_x = list(x_in.shape)[0]
    if verbose:
      print('Encoder Call:')
      print('n_x: ', n_x)
    zs = []
    # This loop is over the batch dimension;
    x_in_split = tf.split(x_in, n_x, axis=0)
    for x_bag in x_in_split:
      if verbose:
        print('x_bag:', x_bag.shape)
      x_bag = tf.squeeze(x_bag, 0)
      z = self.encode_bag(x_bag, batch_size=batch_size, training=training, verbose=verbose)
      zs.append(z)
    z_batch = tf.concat(zs, axis=0) #(batch, features)
    if verbose:
      print('z_batch: ', z_batch.shape)
    if self.mil_type != 'instance':
      yhat = self.apply_classifier(z_batch, training=training, verbose=verbose)
    else:
      yhat = z_batch
    return yhat
