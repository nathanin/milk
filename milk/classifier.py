from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import sys

from milk.encoder import make_encoder

class Classifier(tf.keras.Model):
    def __init__(self, n_classes = 5):
        super(Classifier, self).__init__()

        # For pretraining these options need to match the 
        # eventual target -- use a dummy file to hold the default arguments
        self.densenet = make_encoder()

        # This can be anything
        self.squish = tf.layers.AveragePooling2D(pool_size=(6,6), strides=(1,1))
        self.dropout = tf.layers.Dropout(0.5)
        self.classifier = tf.layers.Dense(n_classes)

    def call(self, x_in, return_embedding=False, verbose=False, training=True):
        output = self.densenet(x_in, training=training, verbose=verbose)
        output = tf.squeeze(self.squish(output))
        if return_embedding:
            return output
        output = self.dropout(output, training=training)
        output = self.classifier(output)

        return output