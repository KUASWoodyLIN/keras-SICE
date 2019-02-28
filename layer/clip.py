import tensorflow as tf


class ClipLayer(tf.keras.layers.Layer):
    def __init__(self, value_min=0, value_max=1, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)
        self.value_min = value_min
        self.value_max = value_max

    def build(self, input_shape):
        super(ClipLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.clip_by_value(inputs, self.value_min, self.value_max)
