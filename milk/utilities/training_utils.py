from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import cv2
import time
import datetime

from .model_utils import loss_function, accuracy_function
from .drawing_utils import create_output_image

# https://github.com/keras-team/keras/issues/3556
import tensorflow.keras.backend as K
# from keras.legacy import interfaces
from tensorflow.keras.optimizers import Optimizer

class AdamAccumulate(Optimizer):

  def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
    if accum_iters < 1:
        raise ValueError('accum_iters must be >= 1')
    super(AdamAccumulate, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
        self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.beta_1 = K.variable(beta_1, name='beta_1')
        self.beta_2 = K.variable(beta_2, name='beta_2')
        self.decay = K.variable(decay, name='decay')
    if epsilon is None:
        epsilon = K.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay
    self.amsgrad = amsgrad
    self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
    self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

  # @interfaces.legacy_get_updates_support
  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]

    lr = self.lr
    # K.tf.floordiv --> tf.floordiv
    completed_updates = K.cast(tf.floordiv(self.iterations, self.accum_iters), K.floatx())

    if self.initial_decay > 0:
        lr = lr * (1. / (1. + self.decay * completed_updates))

    t = completed_updates + 1

    lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

    # self.iterations incremented after processing a batch
    # batch:              1 2 3 4 5 6 7 8 9
    # self.iterations:    0 1 2 3 4 5 6 7 8
    # update_switch = 1:        x       x    (if accum_iters=4)  
    update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
    update_switch = K.cast(update_switch, K.floatx())

    ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

    if self.amsgrad:
      vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    else:
      vhats = [K.zeros(1) for _ in params]

    self.weights = [self.iterations] + ms + vs + vhats

    for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

      sum_grad = tg + g
      avg_grad = sum_grad / self.accum_iters_float

      m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

      if self.amsgrad:
        vhat_t = K.maximum(vhat, v_t)
        p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
        self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
      else:
        p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

      self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
      self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
      self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
      new_p = p_t

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
    return self.updates

  def get_config(self):
    config = {'lr': float(K.get_value(self.lr)),
              'beta_1': float(K.get_value(self.beta_1)),
              'beta_2': float(K.get_value(self.beta_2)),
              'decay': float(K.get_value(self.decay)),
              'epsilon': self.epsilon,
              'amsgrad': self.amsgrad}
    base_config = super(AdamAccumulate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def setup_outputs(basepath='./', return_datestr=False):
    exptime = datetime.datetime.now()
    exptime_str = exptime.strftime('%Y_%m_%d_%H_%M_%S')
    logdir = os.path.join( basepath, 'log',  exptime_str)
    savedir = os.path.join(basepath, 'save', exptime_str)
    imgdir = os.path.join( basepath, 'img',  exptime_str)
    os.makedirs(logdir)
    os.makedirs(savedir)
    os.makedirs(imgdir)
    summary_writer = tf.contrib.summary.create_file_writer(logdir=logdir)

    save_prefix = os.path.join(savedir, 'snapshot')

    if return_datestr:
        return logdir, savedir, imgdir, save_prefix, exptime_str
    else:
        return logdir, savedir, imgdir, save_prefix

def logging(model, 
            val_dataset, 
            loss_function, 
            grads,
            last_N_train_losses,
            N = 10):
    """ Print to console and save to tensorboard summary some metrics

    average training loss, and val loss over N previous iterations
    (fixed network for val dataset, moving network for training losses)

    TODO make tensorboard logging stable. sometimes grads are None

    """
    last_N_val_losses = []
    for _ in range(N):
        val_loss = loss_function(model, val_dataset)
        last_N_val_losses.append(val_loss.numpy())

    print('TRAIN_LOSS = [{:3.5f}] VAL_LOSS = [{:3.5f}]'.format( 
        np.mean(last_N_train_losses), np.mean(last_N_val_losses)
        ))

    # tf.contrib.summary.scalar('train_loss', )
    # tf.contrib.summary.scalar('val_loss', val_loss)

    # grad_summary = grads(model, train_dataset)
    # for gr in grad_summary:
    #     tf.contrib.summary.histogram('grad_{}'.format(gr[1].name), gr[0])
    #     tf.contrib.summary.histogram('var_{}'.format(gr[1].name), gr[1])

def get_nested_variables(model):
    """ Recurse through model and get all the variables

    https://github.com/tensorflow/tensorflow/issues/19250
    """
    print('\nFetching a list of all model variables...')
    densenet_variables = []
    all_variables = []
    print('DenseNet variables:')
    for v in model.densenet.variables:
        print(' ', v.name, v.shape)
        all_variables.append(v)
        densenet_variables.append(v)

    print('DenseNet nested variables:')
    for dense_block in model.densenet.dense_blocks:
        print(' Dense Block:')
        for v in dense_block.variables:
            print('  ', v.name, v.shape)
            all_variables.append(v)
            densenet_variables.append(v)
        
        print('  Convs:')
        for convblock in dense_block.blocks:
            for v in convblock.variables:
                print('   ', v.name, v.shape)
                all_variables.append(v)
                densenet_variables.append(v)

    for transition_block in model.densenet.transition_blocks:
        print(' Transition Block:')
        for v in transition_block.variables:
            print('  ', v.name, v.shape)
            all_variables.append(v)
            densenet_variables.append(v)

    print('Model variables:')
    for v in model.variables:
        print(' ', v.name, v.shape)
        all_variables.append(v)

    return all_variables, densenet_variables

#LOG_EVERY_N = 25
## PRETRAIN_SNAPSHOT = 'pretraining/trained/eager_segmentation-13001'
## PRETRAIN_SNAPSHOT = 'pretraining/classifier/eager_classifier-14001'
#def mil_train_loop(EPOCHS=100, 
#                   EPOCH_ITERS=500, 
#                   global_step=None, 
#                   model=None,
#                   optimizer=None, 
#                   grads=None, 
#                   loss_function=loss_function, #saver=None,
#                   train_dataset=None, 
#                   val_dataset=None, 
#                   accuracy_function=None,
#                   save_prefix=None, 
#                   img_debug_dir=None,
#                   waiting_time=5,
#                   pretrain_snapshot=None):
#    ## Reset global step
#    if global_step.numpy() > 0:
#        tf.assign(global_step, 0)
#
#    best_val_acc = 0.
#    best_train_acc = 0.
#    accuracy_tol = 0.05
#    best_val_loss = np.log(2)
#
#    # all_vars, densenet_vars = get_nested_variables(model)
#    all_vars = model.variables
#    encoder_vars = model.encoder.variables 
#
#    print('Creating saver ({})'.format(save_prefix))
#    encoder_saver = tfe.Saver(encoder_vars)
#    saver = tfe.Saver(all_vars)
#    if pretrain_snapshot is not None:
#        print('Restoring from {}'.format(pretrain_snapshot))
#        encoder_saver.restore(pretrain_snapshot)
#
#    saver.save(save_prefix, global_step=0)
#    best_snapshot = 0
#    since_last_snapshot = 0
#    PATIENTCE = 10000
#
#    print('Performing initial log', end='...')
#    logging(model, val_dataset, loss_function, grads, [1.])
#    print('Done')
#
#    print('Entering training loop:')
#    print('Starting training at global step: {}'.format(global_step.numpy()))
#    global_index = 0
#    last_N_losses = []
#    for epoch in range(1, EPOCHS):
#        batch_times = []
#        for _ in range(EPOCH_ITERS):
#            tf.assign(global_step, global_step+1)
#            tstart = time.time()
#            with tf.device('/gpu:0'):
#                loss_val, grads_and_vars = grads(model, train_dataset)
#                optimizer.apply_gradients(grads_and_vars)
#
#            last_N_losses.append(loss_val.numpy())
#            batch_times.append(time.time()-tstart)
#
#            global_index += 1
#            if global_index % LOG_EVERY_N == 0:
#                print('EPOCH [{:04d}] STEP [{:05d}] '.format(epoch, global_index), end = ' ')
#                logging(model, val_dataset, loss_function, grads, last_N_losses)
#                last_N_losses = []
#
#        mean_batch_time = np.mean(batch_times)
#        train_loss, val_loss, train_acc, val_acc = mil_test_step(model, 
#            grads=grads, 
#            train_dataset=train_dataset, 
#            val_dataset=val_dataset, 
#            global_step=global_step, 
#            mean_batch=mean_batch_time, 
#            N=100,
#            loss_function=loss_function, 
#            accuracy_function=accuracy_function)
#
#        if epoch < waiting_time: 
#            print('Epoch < waiting time ({})'.format(waiting_time))
#            continue
#
#        ## Check whether to save
#        if val_acc > best_val_acc:
#            print('Val acc {} > previous best {}; Snapshotting'.format(val_acc, best_val_acc))
#            saver.save(save_prefix, global_step=global_step)
#            since_last_snapshot = 0
#            best_val_acc = val_acc
#            if val_loss < best_val_loss:
#                print('Val loss {} < previous best {}. Updating'.format(val_loss, best_val_loss))
#                best_val_loss = val_loss
#            best_snapshot = global_step.numpy()
#
#        elif val_acc > best_val_acc - accuracy_tol:
#            print('Matched best val accuracy within tolerance {}'.format(accuracy_tol))
#            # if val_loss < best_val_loss and train_acc >= best_train_acc:
#            if val_loss < best_val_loss:
#                print('Val loss {} < previous best snapshot {}; '.format(val_loss, best_val_loss))
#                print('train acc {} > previous best {}; Snapshotting'.format(train_acc, best_train_acc))
#                saver.save(save_prefix, global_step=global_step)
#                since_last_snapshot = 0
#                best_val_loss = val_loss
#                best_train_acc = train_acc
#                best_snapshot = global_step.numpy()
#
#        if since_last_snapshot > 15 and global_step > PATIENTCE:
#            print('25 Epochs have passed since the last snapshot. Exiting.')
#            break
#        else:
#            since_last_snapshot += 1
#
#    return saver, best_snapshot


def mil_test_step(model, 
              grads, 
              train_dataset,
              val_dataset, 
              global_step, 
              mean_batch, 
              N=75,
              loss_function=loss_function,
              accuracy_function=accuracy_function):
    train_losses = []
    val_losses = []
    with tf.device('/gpu:0'):
        ## accuracy_function has the bootstrap loop built-in
        print('Testing accuracy...')
        train_acc = accuracy_function(model, train_dataset, N=N, mc_dropout=False)
        val_acc = accuracy_function(model, val_dataset, N=N, mc_dropout=False)

        print('Gathering average losses')
        for k in range(N):
        #     if k % 10 == 0:
        #         print('TEST {}'.format(k))

            train_losses.append(loss_function(model, train_dataset, training=False))
            val_losses.append(loss_function(model, val_dataset, training=False))
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print('Iteration {:07d} '.format(global_step.numpy()), end='')
        print('train loss=[{:3.3f}] '.format(train_loss), end='')
        print('val loss=[{:3.3f}] '.format(val_loss), end='')
        print('train acc=[{:3.3}] '.format(train_acc), end='')
        print('val acc=[{:3.3}] '.format(val_acc), end='')
        print('time/batch: [{:3.3f}]s'.format(mean_batch))

    return train_loss, val_loss, train_acc, val_acc

def classifier_train_step(model, dataset, optimizer, grad_fn, global_step=None):
    # with tf.device('/gpu:0'):
    optimizer.apply_gradients(grad_fn(model, dataset))

    if global_step is not None:
        tf.assign(global_step, global_step+1)
