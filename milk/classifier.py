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
        self.encoder = make_encoder()

        # This can be anything
        self.dropout = tf.layers.Dropout(0.5)
        self.classifier = tf.layers.Dense(n_classes)

    def call(self, 
             x_in, 
             return_embedding=False, 
             return_embedding_and_predict=False, 
             verbose=False, 
             training=True):
        embedding = self.encoder(x_in, training=training, verbose=verbose)
        if return_embedding and not return_embedding_and_predict:
            return embedding

        drop = self.dropout(output, training=training)
        prediction = self.classifier(drop)
        if return_embedding_and_predict:
            return output, prediction

        return output