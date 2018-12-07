"""
MI-Net with convolutions
"""
from __future__ import print_function
import tensorflow as tf

from tensorflow.keras.layers import (Input, Dense, Dropout, Average, Lambda)

from .encoder import make_encoder
# from .utilities.model_utils import lr_mult

BATCH_SIZE = 10
def Milk(input_shape, encoder=None, z_dim=512, n_classes=2, 
         dropout_rate=0.3, use_attention=False, encoder_args=None):
    """
    We have to give batches of (batch, num_instances, h, w, ch)

    in the case where batch = 1 we can just squeeze the batch dimension out 
    and put it back after the multiple instances are combined at the end.

    if batch > 1, then we have to process batches separately, then concat the results

    # input_shape should be 4D, with (batch=1, num_instances, ...)
    . The input layer pads on a batch dimension for us if none is given, which is perfect.

    # We have to have a single input layer on the outer-most Model level for TPU translation.
    """
    image = Input(shape=input_shape) #e.g. (None, 100, 96, 96, 3)
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
        features = make_encoder(image=image_squeezed, 
                                input_shape=input_shape,  ## Unused
                                encoder_args=encoder_args)
    else:
        features = encoder(image_squeezed)

    print('features after encoder', features.shape)

    features = Dropout(dropout_rate, name='mil_drop_1')(features)
    features = Dense(z_dim, name='mil_dense_1')(features)
    features = Dropout(dropout_rate, name='mil_drop_2')(features)
    features = Dense(int(z_dim/2), name='mil_dense_2')(features)
    features = Dropout(dropout_rate, name='mil_drop_3')(features)
    print('features after classifier', features.shape)

    # Squish the instances
    # features = Average(axis=0)(features)
    def reduce_mean_output_shape(input_shape):
        shape = list(input_shape)
        shape[0] = 1
        return tuple(shape)
        
    features = Lambda(lambda x: tf.reduce_mean(x, axis=0, keepdims=True),  
                      output_shape=reduce_mean_output_shape)(features)
    print('features after reduce_mean', features.shape)
    logits = Dense(n_classes, activation=tf.nn.softmax, name='mil_classifier')(features)

    model = tf.keras.Model(inputs=[image], outputs=[logits])
    return model