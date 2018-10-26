import tensorflow as tf
import numpy as np

def lr_mult(alpha):
    """
    @P-Gn stackoverflow user:
    https://stackoverflow.com/questions/34945554/ ...
    how-to-set-layer-wise-learning-rate-in-tensorflow/50388264#50388264
    """
    @tf.custom_gradient
    def _lr_mult(x):
        def grad(dy):
            return dy * alpha * tf.ones_like(x)
        return x, grad

    return _lr_mult

BATCH_SIZE = 8
def loss_function(model, dataset, batch_size=BATCH_SIZE, training=True, T=10):
    with tf.device('/cpu:0'):
        x, y = dataset.next()
        x = tf.squeeze(x, axis=0)

    ## Model-associated uncertainty during training
    # yhat_ = tf.expand_dims(model(x.gpu(), T=T, training=training), axis=0)
    # for _ in xrange(T):
    #     yhat_t = tf.expand_dims(model(x.gpu(), T=T, training=training), axis=0)
    #     yhat_ = tf.concat([yhat_, yhat_t], axis=0)
    # yhat = tf.reduce_mean(yhat_, axis=0)

    ## Just dropout
    # print('Running model forward, x:', x.get_shape(), end='...')
    yhat = model(x, batch_size = batch_size, T=T, training=training)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y.gpu(), logits=yhat.gpu())
    # print('Returning loss')
    return loss

def accuracy_function(model, dataset, T=10, N=25, batch_size=BATCH_SIZE, mc_dropout=False):
    acc = 0.
    for _ in range(N):
        with tf.device('/cpu:0'):
            x, y = dataset.next()
            x = tf.squeeze(x, axis=0)

        ## Model uncertainty -- use dropout at test time with training=True
        with tf.device('/gpu:0'):
            if mc_dropout:
                yhat_bar = tf.expand_dims(
                    model(x, batch_size=batch_size, training=mc_dropout), 
                    axis=0
                )

                for _ in range(T):
                    yhat_bar_t = model(x, batch_size = batch_size, training=mc_dropout)
                    yhat_bar_t = tf.expand_dims(yhat_bar_t, axis=0)
                    yhat_bar = tf.concat([yhat_bar, yhat_bar_t], axis=0)

                yhat_bar = tf.reduce_mean(yhat_bar, axis=[0,1])

            else:
                yhat_bar = model(x, batch_size=BATCH_SIZE, training=mc_dropout)

        yhat_bar = np.squeeze(yhat_bar)
        y_argmax = tf.squeeze(tf.argmax(y.gpu(), axis=1))
        yhat_argmax = tf.squeeze(tf.argmax(yhat_bar, axis=0))
        acc += (y_argmax.numpy() == yhat_argmax.numpy())
        # print('yhat_bar:', yhat_bar.numpy())
        # print('yhat_argmax', yhat_argmax.numpy())
        # print('acc', acc)
    acc /= N
    return acc