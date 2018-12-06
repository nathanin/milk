from __future__ import print_function
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (Dropout, Dense, Input)

from encoder import make_encoder

def Classifier(input_shape, n_classes = 5, encoder_args=None):
    image = Input(shape=input_shape)

    #input_shape needs to make its way into the encoder initialization:
    encoder = make_encoder(input_shape=input_shape, encoder_args=encoder_args)
    features = encoder(image)
    features = Dropout(0.5)(features)
    features = Dense(n_classes, activation=tf.nn.softmax)(features)

    model = tf.keras.Model(inputs=[image], outputs=[features])

    return model