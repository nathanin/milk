"""
MI-Net with convolutions
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization)
# from tensorflow.layers import BatchNormalization

from .densenet import DenseNetEager
from .encoder import make_encoder_eager

from milk.utilities.model_utils import lr_mult

class MilkEager(tf.keras.Model):
  def __init__(self, z_dim=256, encoder_args=None, cls_normalize=True, mil_type='attention', 
               heads=1, batch_size=32, deep_classifier=True, temperature=1):

    super(MilkEager, self).__init__()

    self.hidden_dim = z_dim
    self.deep_classifier = deep_classifier
    self.mil_type = mil_type
    self.batch_size = batch_size
    self.temperature = temperature
    self.cls_normalize = cls_normalize
    self.heads = heads

    self.densenet = make_encoder_eager( encoder_args = encoder_args )
    self.drop2  = Dropout(rate=0.3)

    if mil_type == 'attention':
      # self.att_batchnorm = BatchNormalization(trainable=True)
      self.attention = Dense(units=256, 
        activation=tf.nn.tanh, use_bias=False, name='attention')
      self.attention_gate = Dense(units=256, 
        activation=tf.nn.sigmoid, use_bias=False, name='attention_gate')
      self.attention_layer = Dense(units=1, 
        activation=None, use_bias=False, name='attention_layer')

    self.classifier_heads = []
    if self.deep_classifier:
      depth=3
    else:
      depth=1

    for h in range(self.heads):
      # head_layers = []
      # for i in range(depth):
      #   head_layers.append(
      #     Dense(units=self.hidden_dim, activation=tf.nn.relu, use_bias=False,
      #     name = 'head_{}_{}'.format(h, i)))
      # head_layers.append(Dense(units=2, activation=tf.nn.softmax, use_bias=False, 
      #   name='classifier_{}'.format(h)))
      # self.classifier_heads.append(head_layers)
      self.classifier_heads.append(Dense(units=2, activation=tf.nn.softmax, use_bias=False, 
        name='classifier_{}'.format(h)))

  # @tf.contrib.eager.defun
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
    # with tf.device('/cpu:0'):
    att = tf.nn.softmax(att, axis=1)

    # TODO clean up logic
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

  # @tf.contrib.eager.defun
  def apply_mil(self, z, training=True, verbose=False):
    if self.mil_type == 'attention':
      z = self.mil_attention(z, training=training, verbose=verbose)
    # Add elif's here for more MIL or attention types
    else:
      z = tf.reduce_mean(z, axis=0, keep_dims=True)
    if verbose:
      print('z:', z.shape)
    return z

  # TODO unify encode_bag to work in return_z mode for all MIL types
  # @tf.contrib.eager.defun
  def encode_bag(self, x_bag, training=True, verbose=False, return_z=False):
    bag_size = x_bag.shape[0]
    n_bags = bag_size // self.batch_size
    remainder = bag_size - n_bags*self.batch_size
    x_bag = tf.split(x_bag, tf.stack([self.batch_size]*n_bags + [remainder]), axis=0)
    z_bag, z_enc = [None]*len(x_bag), [None]*len(x_bag)

    # This loop raises an AutographParseError if inside a tf.contrib.eager.defun
    # It's a syntax error, maybe we need to change the loop
    for i in range(len(x_bag)):
      x_in = x_bag[i]
      z = self.densenet(x_in, training=training)
      if verbose:
        print('\t z: ', z.shape)
      z = self.drop2(z, training=training)
      # if self.mil_type == 'instance':
        # z_enc.append(z)
        # z_enc[i] = z
        # z = self.apply_classifier(z, verbose=verbose, training=training)
        # if verbose:
        #   print('Instance yhat:', z.shape)
      # z_bag.append(z)
      z_bag[i] = z

    z = tf.concat(z_bag, axis=0)
    if verbose:
      print('\tz bag:', z.shape)

    # request to return zed should clobber the rest of these ops
    if return_z and self.mil_type == 'instance':
      # z_enc = tf.concat(z_enc, axis=0)
      # print('Returning zenc and z', z_enc.shape, z.shape)
      z = z
      # return z_enc, z
    elif return_z:
      z = z
    else:
      if self.mil_type == 'instance':
        z = tf.reduce_mean(z, axis=0, keepdims=True)
      else:
        z = self.apply_mil(z, verbose=verbose)
    return z

  # @tf.contrib.eager.defun
  def apply_classifier(self, features, heads=[0], verbose=False, training=True):
    """ Apply classifier layers

    In multihead mode, pick one of the heads at random.
    NOTE change to tf.random to handle graph mode
    """
    yout = []
    for h in heads:
      layer = self.classifier_heads[h]
      yout.append(layer(features))
    return yout

  @tf.contrib.eager.defun
  def call(self, x_in, heads=[0], training=True, verbose=False):
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

    # Make sure the gradeint stops here
    x_in = tf.stop_gradient(x_in)

    if verbose:
      print('Encoder Call:')
      print('n_x: ', n_x)
    # This loop is over the batch dimension
    x_in_split = tf.split(x_in, n_x, axis=0)
    zs = []
    for x_bag in x_in_split:
      if verbose:
        print('x_bag:', x_bag.shape)
      x_bag = tf.squeeze(x_bag, 0)
      z = self.encode_bag(x_bag, training=training, verbose=verbose)
      zs.append(z)
    z_batch = tf.concat(zs, axis=0) #(batch, features)
    if verbose:
      print('z_batch: ', z_batch.shape)

    # if self.mil_type == 'instance':
    #   # yhat = z_batch
    #   yhat = self.apply_classifier(z_batch, heads=heads, training=training, verbose=verbose)
    #   yhat = tf.reduce_mean(yhat, axis=0, keepdims=True)
    # else:
    #   yhat = self.apply_classifier(z_batch, heads=heads, training=training, verbose=verbose)

    yhat = []
    for h in heads:
      layer = self.classifier_heads[h]
      yhat.append(layer(z_batch))

    if self.mil_type == 'instance':
      yhat = tf.reduce_mean(yhat, axis=0, keepdims=True)
    
    if verbose:
      print('yhat: ', yhat)

    return yhat
