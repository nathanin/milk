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

  acc /= N
  return acc


def classifier_loss_fn(model, dataset):
  x, ytrue = dataset.iterator.next()

  with tf.device('/gpu:0'):
    yhat = model(x)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=ytrue, logits=yhat)
    loss = tf.reduce_mean(loss)
  return loss


def make_inference_functions(encode_model, predict_model, pretrained_model, attention_model=None):
  """ 
  Do not use this. It's so slow. 

  use model.load_weights(<path>, by_name=True) instead!

  """
  pretrained_layers = {l.name: l for l in pretrained_model.layers}

  print(encode_model.get_weights())
  
  encode_weights = []
  predict_weights = []
  attention_weights = []

  for l in encode_model.get_weights():
    lname = l.name
    try:
      encode_weights.append(pretrained_layers[lname].get_weights())
      print('Got encoder weight for layer {}'.format(lname))
    except:
      print('Encoder skipping layer {}'.format(lname))

  for l in predict_model.layers:
    lname = l.name
    try:
      predict_weights.append(pretrained_layers[lname].get_weights())
      print('Got predict weight for layer {}'.format(lname))
    except:
      print('Encoder skipping layer {}'.format(lname))

  if attention_model is not None:
    for l in attention_model.layers:
      lname = l.name
      attention_weights.append(pretrained_layers[lname].get_weights())
      print('Got attention weight for layer {}'.format(lname))

  # for lname, l in pretrained_layers.items():
  #   w = l.get_weights()
  #   if 'encoder' in lname or 'deep' in lname:
  #     try:
  #       # encode_model.get_layer(lname).set_weights(w)
  #       print('Got encoder weight for layer {}'.format(lname))
  #       encode_weights.append(w)
  #     except:
  #       pass
    
  #   if 'encoder' not in lname:
  #     try:
  #       # predict_model.get_layer(lname).set_weights(w)
  #       print('Got predict weight for layer {}'.format(lname))
  #       predict_weights.append(w)
  #     except:
  #       pass

  #   if (attention_model is not None) and ('encoder' not in lname):
  #     try:
  #       # attention_model.get_layer(lname).set_weights(w)
  #       print('Got attention weight for layer {}'.format(lname))
  #       attention_weights.append(w)
  #     except:
  #       pass

  encode_model.set_weights(encode_weights)
  predict_model.set_weights(predict_weights)

  if attention_model is not None:
    attention_model.set_weights(attention_weights)
    return encode_model, predict_model, attention_model
  else:
    return encode_model, predict_model