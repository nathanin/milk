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
  def __init__(self, z_dim=256, encoder_args=None, mil_type='attention', deep_classifier=True):

    super(MilkEager, self).__init__()

    self.hidden_dim = z_dim
    self.deep_classifier = deep_classifier
    self.mil_type = mil_type
    self.built_fn = False

    self.densenet = make_encoder_eager( encoder_args = encoder_args )
    # self.densenet = DenseNet(
    #   depth_of_model=36,
    #   growth_rate=32,
    #   num_of_blocks=3,
    #   output_classes=2,
    #   num_layers_in_each_block=12,
    #   data_format='channels_last',
    #   dropout_rate=0.3,
    #   pool_initial=True
    # )

    self.drop2  = Dropout(rate=0.3)

    if mil_type == 'attention':
      self.att_batchnorm = BatchNormalization()
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

    for i in range(depth):
      self.classifier_layers.append(
        Dense(units=self.hidden_dim, activation=tf.nn.relu, use_bias=False,
        name = 'mil_deep_{}'.format(i)))

    self.classifier = Dense(units=2, activation=tf.nn.softmax, use_bias=False, name='mil_classifier')

  def mil_attention(self, features, verbose=False, return_att=False, training=True):
    features = self.att_batchnorm(features, training=training)
    att = self.attention(features)
    att_gate = self.attention_gate(features)
    att = att * att_gate # tf.multiply()
    if verbose:
      print('attention:', att.shape)

    att = self.attention_layer(att)
    att = tf.transpose(att, perm=(1,0)) 
    if verbose:
      print('attention:', att.shape)

    # Question: WTF?
    # tensorflow.python.framework.errors_impl.InternalError: CUB segmented reduce errorinvalid configuration argument [Op:Softmax]
    # unless we put this op on CPU in eager mode.
    with tf.device('/cpu:0'):
      att = tf.nn.softmax(att, axis=1)

    if verbose:
      print('attention:', att.shape)

    # Scale learning proportionally to bag size
    z = tf.matmul(att, features)

    if return_att:
      return z, att
    else:
      return z

  def encode_bag(self, x_bag, batch_size=64, 
    training=True, verbose=False, return_z=False):
    z_bag = []
    n_bags = x_bag.shape[0] // batch_size
    remainder = x_bag.shape[0] - n_bags*batch_size
    x_bag = tf.split(x_bag, [batch_size]*n_bags + [remainder], axis=0)
    for x_in in x_bag:
      z = self.densenet(x_in, training=training)
      if verbose:
        print('\t z: ', z.shape)
      z = self.drop2(z, training=training)
      z_bag.append(z)

    z = tf.concat(z_bag, axis=0)
    if verbose:
      print('\tz bag:', z.shape)

    if return_z:
      return z
    else:
      z = self.apply_mil(z, verbose=verbose)
      return z

  def apply_mil(self, z, training=True, verbose=False):
    if self.mil_type == 'attention':
      z = self.mil_attention(z, training=training, verbose=verbose)
    # Add elif's here
    else:
      z = tf.reduce_mean(z, axis=0, keep_dims=True)

    if verbose:
      print('z:', z.shape)

    return z

  def apply_classifier(self, features, verbose=False, training=True):
    for layer in self.classifier_layers:
      features = layer(features)
    if verbose:
      print('features - ff:', features.shape)

    yhat = self.classifier(features)
    if verbose:
      print('yhat:', yhat.shape)

    return yhat

  ## BUG for tf.contrib.eager.defun
  #def build_encode_fn(self, training=True, verbose=False, batch_size=64):
  #  print('building encode fn')
  #  def built_fn(x_bag):
  #    z_bag = self.encode_bag(x_bag, batch_size=batch_size, training=training, 
  #      verbose=verbose)
  #    return z_bag
  #
  #  @tf.contrib.eager.defun
  #  def func(x_bag):
  #    return tf.map_fn(built_fn, x_bag, parallel_iterations=4)

  #  self.built_encode_fn = func
  #  self.built_fn = True

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
    for x_bag in x_in:
      if verbose:
        print('x_bag:', x_bag.shape)
      z = self.encode_bag(x_bag, batch_size=batch_size, training=training, 
        verbose=verbose)
      zs.append(z)

    #if not self.built_fn:
    #  self.build_encode_fn(training=training, verbose=verbose, batch_size=batch_size)

    ##zs = tf.map_fn(self.built_encode_fn, x_in, parallel_iterations=n_x)
    #zs = self.built_encode_fn(x_in)

    # Gather
    z_concat = tf.concat(zs, axis=0) #(batch, features)
    if verbose:
      print('z_concat: ', z_concat.shape)

    ## Classifier 
    yhat = self.apply_classifier(z_concat, training=training, verbose=verbose)

    return yhat
