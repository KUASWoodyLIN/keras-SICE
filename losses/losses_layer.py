import tensorflow as tf


class L1Losses(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L1Losses, self).__init__(**kwargs)

    def build(self, input_shape):
        super(L1Losses, self).build(input_shape)

    def call(self, inputs, **kwargs):
        y_pred = inputs[0]
        y_true = inputs[1]
        loss = tf.reduce_sum(tf.abs(y_true - y_pred), axis=[1, 2, 3])
        # self.add_loss(loss, inputs=inputs)
        return tf.reduce_mean(loss)


class MSELosses(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MSELosses, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MSELosses, self).build(input_shape)

    def call(self, inputs, **kwargs):
        y_pred = inputs[0]
        y_true = inputs[1]
        loss = tf.reduce_sum(tf.squared_difference(y_true, y_pred), axis=[1, 2, 3])
        # self.add_loss(loss, inputs=inputs)
        return tf.reduce_mean(loss)
