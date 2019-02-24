import tensorflow as tf


def pass_loss(y_true, y_pred):
    return y_pred


def dssim_loss(y_true, y_pred):
    return tf.reduce_sum((1 - tf.image.ssim(y_true, y_pred, max_val=1.0)) / 2)
