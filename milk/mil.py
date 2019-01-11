"""
MI-Net with convolutions
"""
from __future__ import print_function
import tensorflow as tf

from tensorflow.keras.layers import (Input, Dense, Dropout, Average, Lambda, 
    Multiply, Permute, Softmax, Dot)

from .encoder import make_encoder
# from .utilities.model_utils import lr_mult

def squish_mean(features):
    # Squish the instances using a custom Keras layer wrapping tf.reduce_mean
    def reduce_mean_output_shape(input_shape):
        shape = list(input_shape)
        shape[0] = 1
        return tuple(shape)

    # Define and register a multiplier on this gradient
    features = Lambda(lambda x: tf.reduce_mean(x, axis=0, keepdims=True),  
                      output_shape=reduce_mean_output_shape)(features)

    return features

def instance_classifier(features, n_classes):
    print('Setting up instance classifier')
    logits = Dense(n_classes, activation=tf.nn.softmax, name='mil_classifier')(features)
    logits = squish_mean(logits)
    print('logits after reduce_mean', logits.shape)
    return logits

def deep_feedforward(features, n_layers=5, width=256, dropout_rate=0.3):
    for k in range(n_layers):
        features = Dense(width, activation=tf.nn.relu, name='deep_{}'.format(k))(features)
        features = Dropout(dropout_rate, name='deep_drop_{}'.format(k))(features)

    return features

def mil_features(features, n_classes, z_dim, dropout_rate):
    """ Caclulate the instance features then squish them """
    print('Features going into `mil_features`:', features.shape)
    features = Dropout(dropout_rate, name='mil_drop_1')(features)
    features = Dense(z_dim, activation=tf.nn.relu, name='mil_dense_1')(features)
    features = Dropout(dropout_rate, name='mil_drop_2')(features)
    features = Dense(int(z_dim/2), activation=tf.nn.relu, name='mil_dense_2')(features)
    features = Dropout(dropout_rate, name='mil_drop_3')(features)
    features = squish_mean(features)
    print('features after reduce_mean', features.shape)
    return features

def average_pooling(features, n_classes, z_dim, dropout_rate):
    print('Setting up average pooling MIL')
    features = mil_features(features, n_classes, z_dim, dropout_rate)
    logits = Dense(n_classes, activation=tf.nn.softmax, name='mil_classifier')(features)
    return logits

def attention_pooling(features, n_classes, z_dim, dropout_rate, use_gate=True, 
                      return_attention=False):
    """ Calculate attention then modulate the magnitude of the features

    Squish the attention-modulated features and return logits
    """
    attention = Dense(256, activation=tf.nn.tanh, use_bias=False,
                      name='att_0')(features)
    if use_gate:
        gate = Dense(256, activation=tf.nn.sigmoid, use_bias=False, 
                     name='att_gate')(features)
        print('Gate:', gate.shape)
        attention = Multiply(name='att_1')([attention, gate])
        print('Gated attention:', attention.shape)
    print('Embedded attention:', attention.shape)

    attention = Dense(1, activation=None, use_bias=False, name='att_2')(attention)
    print('Calculated attention:', attention.shape)

    attention = Lambda(lambda x: tf.transpose(x, perm=(1,0)))(attention)
    print('Transposed attention:', attention.shape)

    attention = Softmax(axis=1, name='att_sm')(attention)
    print('Softmaxed attention:', attention.shape)
    if return_attention:
        return attention

    features = Lambda(lambda x: tf.matmul(x[0], x[1]))([attention, features])
    # features = Dot(axes=1, name='feat_att')([attention, features])
    print('Scaled features:', features.shape)

    features = mil_features(features, n_classes, z_dim, dropout_rate)
    logits = Dense(n_classes, activation=tf.nn.softmax, name='mil_classifier')(features)
    return logits


def Milk(input_shape, 
         encoder=None, 
         z_dim=256, 
         n_classes=2, 
         dropout_rate=0.3, 
         encoder_args=None,
         mode="instance", 
         use_gate=True, 
         deep_classifier=False,
         freeze_encoder=False):

    """ Build the Multiple Instance Learning model

    Return a function that accepts batches of (batch, bag_size, h, w, ch)

    # When batch = 1 we can just squeeze the batch dimension out 
      and pad it back after the bag instances are combined at the end.
      if batch > 1, then we have to process batches separately, then concat the results.

    # for TPU exectution we force batch=8 so that we shard the batch into (1, bag, ...)
      to process each bag on its own TPU processor.

    # input_shape should be 4D, with (batch=1, num_instances, ...)
      The input layer pads on a batch dimension for us if none is given, which is perfect.

    # We have to have a single input layer on the outer-most Model level for TPU translation.
      So, during model construction, we need the nested 'models' to return tensors instead
      of tf.keras.Model instances.

    # `mode` (str) dictates the kind of MIL to use:
      "instance"  -- baseline
      "average"   -- average over the bag in latent space
      "attention" -- compute weighted average using learned weights
    """
    image = Input(shape=input_shape, name='image') #e.g. (None, 100, 96, 96, 3)
    print('image input:', image.shape)
    # Squeeze off the batch dimension
    # Assume the actual batch dimension = 1
    def squeeze_output_shape(input_shape):
        shape = list(input_shape)
        shape = shape[1:]
        return tuple(shape)

    image_squeezed = Lambda(lambda x: tf.squeeze(x, axis=0), 
                            output_shape=squeeze_output_shape)(image)
    print('image squeezed', image_squeezed.shape)
    if encoder is None:
        if freeze_encoder:
            print('NOTE: Initializing encoder with trainable = False')
            trainable = False
        else:
            trainable = True

        features = make_encoder(image=image_squeezed, 
                                input_shape=input_shape,  ## Unused
                                encoder_args=encoder_args,
                                trainable=trainable)
    else:
        features = encoder(image_squeezed)

    print('features after encoder', features.shape)
    # insert an identity layer to hook into later:
    features = Lambda(lambda x: x, name='feature_hook')(features)

    # if freeze_encoder:
    #     print('Stopping gradient from flowing to the encoder.')
    #     features = Lambda(lambda x: tf.stop_gradient(x))(features)

    if deep_classifier:
        features = deep_feedforward(features, n_layers=5)

    if mode == "instance":
        logits = instance_classifier(features, n_classes)
    elif mode == "average":
        logits = average_pooling(features, n_classes, z_dim, dropout_rate)
    elif mode == "attention":
        logits = attention_pooling(features, n_classes, z_dim, dropout_rate, use_gate=use_gate)
    else:
        print('Multiple-Instance mode {} not recognized'.format(mode))
        raise NotImplementedError

    model = tf.keras.Model(inputs=[image], outputs=[logits])
    return model


def MilkEncode(input_shape, encoder=None, dropout_rate=0.3, 
               encoder_args=None, deep_classifier=False):
    image = Input(shape=input_shape, name='image') #e.g. (None, 100, 96, 96, 3)
    print('image input:', image.shape)
    
    if encoder is None:
        features = make_encoder(image=image, 
                                input_shape=input_shape,  ## Unused
                                encoder_args=encoder_args)
    else:
        features = encoder(image)

    if deep_classifier:
        features = deep_feedforward(features, n_layers=5)

    return tf.keras.Model(inputs=[image], outputs=[features])


def MilkPredict(input_shape, z_dim=256, n_classes=2, dropout_rate=0.3, 
                mode="instance", use_gate=True):
    print('Setting up Predict model in {} mode'.format(mode))
    features = Input(shape=input_shape, name='feat_in')

    if mode == "instance":
        logits = instance_classifier(features, n_classes)
    elif mode == "average":
        logits = average_pooling(features, n_classes, z_dim, dropout_rate)
    elif mode == "attention":
        logits = attention_pooling(features, n_classes, z_dim, dropout_rate, 
                                   use_gate=use_gate)
    else:
        print('Multiple-Instance mode {} not recognized'.format(mode))
        raise NotImplementedError

    return tf.keras.Model(inputs=[features], outputs=[logits])


def MilkAttention(input_shape, z_dim=256, n_classes=2, dropout_rate=0.3, 
                  use_gate=True):
    
    features = Input(shape=input_shape, name='feat_in')
    attention = attention_pooling(features, n_classes, z_dim, dropout_rate, 
                                use_gate=use_gate, return_attention=True)

    return tf.keras.Model(inputs=[features], outputs=[attention])
