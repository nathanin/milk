from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL
def draw_to_grid(x_in, y, attention):
    n_x = x_in.shape[0]
    
    sq_side = x_in.shape[1]
    gridsize = np.ceil(np.sqrt(n_x))
    pred_class = np.argmax(np.squeeze(y))

    img_out = np.zeros((100+int(gridsize * sq_side), 
                        100+int(gridsize * sq_side),
                        3))
    img_out += pred_class

    xpts = np.linspace(10, 50+int(gridsize * sq_side)-sq_side-10, gridsize, dtype=np.int)
    ypts = np.linspace(10, 50+int(gridsize * sq_side)-sq_side-10, gridsize, dtype=np.int)

    idx = 0
    attention = np.squeeze(attention)
    img_scaled_attention = attention * (1/attention.max())
    scaled_attention = attention 
    for x in xpts:
        for y in ypts:
            img = np.squeeze(x_in[idx, ...])
            # img = cv2.resize(img, dsize=(0,0), fx=2., fy=2.)
            # print('draw to grid: img:', img.shape, img.min(), img.max())
            img *= img_scaled_attention[idx]
            img_out[x:x+sq_side, y:y+sq_side, :] = img

            idx+=1 
            if idx == n_x:
                break
        if idx == n_x:
            break

    idx = 0
    for x in xpts:
        for y in ypts:
            attn = scaled_attention[idx]
            ## burn in text
            cv2.putText(img_out, '{:1.3f}'.format(attn), (y+12, x+12), FONT, 0.8, (1,1,1), 1)
            idx += 1
            if idx == n_x:
                break
        if idx == n_x:
            break

    return img_out*255.

def create_output_image(model, dataset=None, x=None, y=None, T = 25, n_imgs = 36):
    """ Create a visualization for a test bag
    """
    if x is None:
        with tf.device('/cpu:0'):
            x, y = dataset.next()
            x = tf.squeeze(x, axis=0)
            # x = x.numpy()
            y = y.numpy()

    yhats = []
    atts = []
    for _ in range(T):
        yhat, att = model(x, batch_size=8, training=True, return_attention=True)
        yhats.append(np.expand_dims(yhat.numpy(), 0))
        atts.append(att.numpy())

    x = x.numpy()
    yhats = np.concatenate(yhats)
    atts = np.concatenate(atts)

    yhat_mean = np.mean(yhats, axis=0)
    # yhat_std = np.std(yhats, axis=0)
    att_mean = np.mean(atts, axis=0)
    att_std = np.std(atts, axis=0)

    max_att_idx = np.argsort(att_mean)
    idx_use = max_att_idx[-n_imgs:]
    x_use = x[idx_use]
    att_mean_use = att_mean[idx_use]
    att_std_use = att_std[idx_use]

    mean_grid = draw_to_grid(x_use, yhat_mean, att_mean_use)
    std_grid = draw_to_grid(x_use, y, att_std_use)
    img_out = np.concatenate([mean_grid, std_grid], axis=1)

    return img_out
