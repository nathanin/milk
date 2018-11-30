"""
MI-Net with convolutions
"""
from __future__ import print_function
import tensorflow as tf

# from .densenet import DenseNet
from milk.encoder import make_encoder
from milk.utilities.model_utils import lr_mult

BATCH_SIZE = 10
class Milk(tf.keras.Model):
    def __init__(self, z_dim=512, encoder=None):
        super(Milk, self).__init__()
        self.encoder = make_encoder()
        # self.densenet = DenseNet(
        #     depth_of_model=15,
        #     growth_rate=32,
        #     num_of_blocks=3,
        #     output_classes=2,
        #     num_layers_in_each_block=5,
        #     data_format='channels_last',
        #     dropout_rate=0.3,
        #     pool_initial=True
        # )

        # self.dense1 = tf.layers.Dense(units=512, activation=tf.nn.relu, use_bias=True, name='dense1')
        self.drop2 = tf.layers.Dropout(rate=0.5)

        self.dense2 = tf.layers.Dense(units=z_dim, 
            activation=tf.nn.relu, 
            use_bias=False, 
            name='dense2') 
        self.drop3 = tf.layers.Dropout(rate=0.5)
        # self.uncertainty = self.track_layer(tf.layers.Dense(units=1, activation=None, use_bias=False)

        self.classifier_nonlinearity_1 = tf.layers.Dense(units=512,
            activation=tf.nn.relu, 
            use_bias=False,
            name='classifier_nonlin_1')
        self.classifier_nonlinearity_2 = tf.layers.Dense(units=256,
            activation=tf.nn.relu, 
            use_bias=False,
            name='classifier_nonlin_2')
        self.classifier = tf.layers.Dense(units=2, 
            activation=None, 
            use_bias=False, 
            name='classifier')

    def call(self, 
             x_in, 
             T=20, 
             batch_size = BATCH_SIZE, 
             training=True, 
             verbose=False,
             return_embedding=False,
             return_attention=False):
        """
        `training` controls the use of dropout and batch norm, if defined
        `return_embedding`
            prediction
            attention
            raw embedding (batch=num instances)
            embedding after attention (batch=1)
            classifier hidden layer (batch=1)

        """
        ## Like a manual tf.map()
        if verbose:
            print(x_in.get_shape())

        ## BUG sets of 1 should be handled
        n_x = x_in.get_shape().as_list()[0]
        if n_x == 1:
            x_in = tf.squeeze(x_in, 0)
            n_x = x_in.get_shape().as_list()[0]

        n_batches = int(n_x / batch_size)
        if n_x % batch_size == 0:
            batches = [batch_size]*n_batches
        else:
            batches = [batch_size]*n_batches+[n_x-(n_batches*batch_size)]
        x_split = tf.split(x_in, batches, axis=0)

        if verbose:
            print('Encoder Call:')
            print('n_x: ', n_x)
            print('n_batches', n_batches)
            print('batches', batches)
            print('x_split', len(x_split))

        zs = []
        for x_batch in x_split:
            # divide the learning rate since gradient accumulates
            # z = lr_mult(0.5)(self.encoder(x_batch, training=training))
            z = self.encoder(x_batch, training=training)
            if verbose:
                print('\t z: ', z.shape)
            # z = tf.squeeze(z, [1,2])
            z = self.drop2(z, training=training)
            z = self.dense2(z)
            z = self.drop3(z, training=training)
            zs.append(z)

        # Gather
        z_concat = tf.concat(zs, axis=0)
        if verbose:
            print('z_concat: ', z_concat.get_shape())

        ## MIL layer
        ## Note reduce_mean is tf.math.reduce_mean in TF>1.12
        z = tf.reduce_mean(z_concat, axis=0, keepdims=True)

        if verbose:
            print('z:', z.get_shape())

        ## Classifier 
        net = self.classifier_nonlinearity_1(z)
        net = self.classifier_nonlinearity_2(net)
        yhat = self.classifier(net)
        if verbose:
            print('yhat:', yhat.get_shape())

        if return_embedding:
            return yhat, att, z_concat, z, net
        if return_attention:
            return yhat, att
        else:
            return yhat