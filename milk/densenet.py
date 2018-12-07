# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Densely Connected Convolutional Networks.

Reference [
Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

https://github.com/tensorflow/tensorflow/blob/master/ ...
tensorflow/contrib/eager/python/examples/densenet/densenet.py

Rewritten into Keras' functional API

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import (
  Conv2D, Dense, Dropout, AveragePooling2D, 
  MaxPooling2D, GlobalAveragePooling2D, Flatten, 
  BatchNormalization, Input, Concatenate)
l2 = tf.keras.regularizers.l2

def ConvBlock(features, num_filters, data_format, bottleneck, weight_decay=1e-4,
              dropout_rate=0.3, name_suffix=''):
  """Convolutional Block consisting of (batchnorm->relu->conv).

  Arguments:
    features: tensor input
    num_filters: number of filters passed to a convolutional layer.
    data_format: "channels_first" or "channels_last"
    bottleneck: if True, then a 1x1 Conv is performed followed by 3x3 Conv.
    weight_decay: weight decay
    dropout_rate: dropout rate.
  """
  if bottleneck:
    inter_filter = num_filters * 4
    features = Conv2D(inter_filter,
                      (1, 1),
                      activation=tf.nn.relu,
                      padding="same",
                      use_bias=False,
                      data_format=data_format,
                      # kernel_initializer="he_normal",
                      kernel_regularizer=l2(weight_decay),
                      name='encoder_bottle{}'.format(name_suffix)
                      )(features)

  axis = -1 if data_format == "channels_last" else 1
  # don't forget to set use_bias=False when using batchnorm
  features = Conv2D(num_filters,
                    (3, 3),
                    activation=tf.nn.relu,
                    padding="same",
                    use_bias=False,
                    data_format=data_format,
                    # kernel_initializer="he_normal",
                    kernel_regularizer=l2(weight_decay),
                    name='encoder{}'.format(name_suffix)
                    )(features)

  # self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)
  features = tf.layers.Dropout(dropout_rate)(features)

  return features


def TransitionBlock(features, num_filters, data_format, weight_decay=1e-4,
                    dropout_rate=0.3, block_num=0):
  """Transition Block to reduce the number of features.

  Arguments:
    num_filters: number of filters passed to a convolutional layer.
    data_format: "channels_first" or "channels_last"
    weight_decay: weight decay
    dropout_rate: dropout rate.
  """

  axis = -1 if data_format == "channels_last" else 1

  # self.batchnorm = tf.keras.layers.BatchNormalization(axis=axis)
  features = Conv2D(num_filters,
                    (1, 1),
                    activation=tf.nn.relu,
                    padding="same",
                    use_bias=False,
                    data_format=data_format,
                    #  kernel_initializer="he_normal",
                    kernel_regularizer=l2(weight_decay),
                    name='encoder_trans{}'.format(block_num)
                    )(features)

  features = AveragePooling2D(pool_size=(2,2),
                              strides=(2,2),
                              data_format=data_format,
                              name='encoder_trans_pool{}'.format(block_num)
                              )(features)

  return features


def DenseBlock(features, num_layers, growth_rate, data_format, 
               bottleneck, weight_decay=1e-4, dropout_rate=0.3,
               block_num=0):
  """Dense Block consisting of ConvBlocks where each block's
  output is concatenated with its input.

  Arguments:
    num_layers: Number of layers in each block.
    growth_rate: number of filters to add per conv block.
    data_format: "channels_first" or "channels_last"
    bottleneck: boolean, that decides which part of ConvBlock to call.
    weight_decay: weight decay
    dropout_rate: dropout rate.
  """
  axis = -1 if data_format == "channels_last" else 1

  blocks = []
  # First convolution
  x = ConvBlock(features, 
                growth_rate,
                data_format,
                bottleneck,
                weight_decay,
                dropout_rate,
                name_suffix='{}_f'.format(block_num))
  for i in range(int(num_layers)-1):
    x_i = ConvBlock(x, 
                    growth_rate,
                    data_format,
                    bottleneck,
                    weight_decay,
                    dropout_rate,
                    name_suffix='{}_{}'.format(block_num, i))
    x = Concatenate(axis=axis)([x, x_i])

  return x

def DenseNet(image, input_shape, depth_of_model, growth_rate, num_of_blocks, 
             num_layers_in_each_block, data_format, bottleneck=True,
             compression=0.5, weight_decay=1e-4, dropout_rate=0.3,
             pool_initial=True, include_top=True, with_classifier=False,
             num_classes=2, return_model=False):
  """Creating the Densenet Architecture.
  All the same as before; except we require a fixed input shape

  Arguments:
    depth_of_model: number of layers in the model.
    growth_rate: number of filters to add per conv block.
    num_of_blocks: number of dense blocks.
    output_classes: number of output classes.
    num_layers_in_each_block: number of layers in each block.
                              If -1, then we calculate this by (depth-3)/4.
                              If positive integer, then the it is used as the
                                number of layers per block.
                              If list or tuple, then this list is used directly.
    data_format: "channels_first" or "channels_last"
    bottleneck: boolean, to decide which part of conv block to call.
    compression: reducing the number of inputs(filters) to the transition block.
    weight_decay: weight decay
    rate: dropout rate.
    pool_initial: If True add a 7x7 conv with stride 2 followed by 3x3 maxpool
                  else, do a 3x3 conv with stride 1.
    include_top: If true, GlobalAveragePooling Layer and Dense layer are
                 included.

    return_model: If true, gather itself and return a tf.keras.Model object.
                  else, return the final features tensor
  """
  # deciding on number of layers in each block
  if isinstance(num_layers_in_each_block, list) or isinstance(
      num_layers_in_each_block, tuple):
    num_layers_in_each_block = list(num_layers_in_each_block)
  else:
    if num_layers_in_each_block == -1:
      if num_of_blocks != 3:
        raise ValueError(
            "Number of blocks must be 3 if num_layers_in_each_block is -1")
      if (depth_of_model - 4) % 3 == 0:
        num_layers = (depth_of_model - 4) / 3
        if bottleneck:
          num_layers //= 2
        num_layers_in_each_block = [num_layers] * num_of_blocks
      else:
        raise ValueError("Depth must be 3N+4 if num_layer_in_each_block=-1")
    else:
      num_layers_in_each_block = [
          num_layers_in_each_block] * num_of_blocks

  axis = -1 if data_format == "channels_last" else 1

  # setting the filters and stride of the initial covn layer.
  if pool_initial:
    init_filters = (7, 7)
    stride = (2, 2)
  else:
    init_filters = (3, 3)
    stride = (1, 1)

  num_filters = 2 * growth_rate


  # Setup input layer
  if image is None:
    image = Input(shape=input_shape, name='encoder_input')

  # first conv and pool layer
  features = Conv2D(num_filters,
                    init_filters,
                    strides=stride,
                    padding="same",
                    use_bias=False,
                    data_format=data_format,
                    # kernel_initializer="he_normal",
                    kernel_regularizer=l2(weight_decay),
                    name='encoder_conv1'
                    )(image)
  if pool_initial:
    features = MaxPooling2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding="same",
                            data_format=data_format,
                            name='encoder_pool_init'
                            )(features)
  #   self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)

  # calculating the number of filters after each block
  num_filters_after_each_block = [num_filters]
  for i in range(1, num_of_blocks):
    temp_num_filters = num_filters_after_each_block[i-1] + (
        growth_rate * num_layers_in_each_block[i-1])
    # using compression to reduce the number of inputs to the
    # transition block
    temp_num_filters = int(temp_num_filters * compression)
    num_filters_after_each_block.append(temp_num_filters)

  # dense blocks
  for i in range(num_of_blocks):
    features = DenseBlock(features, 
                          num_layers_in_each_block[i],
                          growth_rate,
                          data_format,
                          bottleneck,
                          weight_decay,
                          dropout_rate, 
                          block_num=i)
    print(features.shape)

    if i+1 < num_of_blocks:
      features = TransitionBlock(features,
                          num_filters_after_each_block[i+1],
                          data_format,
                          weight_decay,
                          dropout_rate,
                          block_num=i)
      print('transition', features.shape)

  # last pooling and fc layer
  if include_top:
    # self.last_pool = tf.layers.AveragePooling2D(pool_size=(6,6), strides=(6,6))
    features = GlobalAveragePooling2D(data_format=data_format, name='encoder_glob_pool')(features)
    print(features.shape)
    # self.last_pool = tf.layers.Flatten()

  if with_classifier:
    features = Dense(num_classes, activation=tf.nn.softmax, name='encoder_classifier')(features)

  if return_model:
    model = tf.keras.Model(inputs=[image], outputs=[features])
    return model
  else:
    return features