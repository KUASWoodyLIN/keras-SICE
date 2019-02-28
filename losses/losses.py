import tensorflow as tf


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_true, y_pred), axis=[1, 2, 3]))


def pass_loss(y_true, y_pred):
    return y_pred


def dssim_loss(y_true, y_pred):
    dssim = tf.reduce_sum((1 - tf.image.ssim(y_true, y_pred, max_val=1.0)) / 2)
    l1 = tf.reduce_mean(tf.reduce_sum(tf.abs(y_true - y_pred), axis=[1, 2, 3]))
    mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_true, y_pred), axis=[1, 2, 3]))
    total_loss = dssim * 1000 + l1 + mse
    return total_loss
