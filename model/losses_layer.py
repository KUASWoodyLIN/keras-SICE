import tensorflow as tf


class L1Losses(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L1Losses, self).__init__(**kwargs)

    def build(self, input_shape):
        super(L1Losses, self).build(input_shape)

    def call(self, inputs, **kwargs):
        y_pred = inputs[0]
        y_true = inputs[1]
        loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        # self.add_loss(loss, inputs=inputs)
        return loss
