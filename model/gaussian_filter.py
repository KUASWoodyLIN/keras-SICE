import tensorflow as tf


class GaussianFilterLayer(tf.keras.layers.Layer):
    def __init__(self, size, mean, std, **kwargs):
        super(GaussianFilterLayer, self).__init__(**kwargs)
        self.size = size
        self.mean = float(mean)
        self.std = float(std)

    def build(self, input_shape):
        # d = self.add_weight('kernel', )
        d = tf.distributions.Normal(self.mean, self.std)
        vals = d.prob(tf.range(start=-self.size, limit=self.size+1, dtype=tf.float32))
        gauss_kernel = tf.einsum('i,j->ij', vals, vals)
        gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
        self.gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis] * tf.ones((1, 1, 3, 1))
        super(GaussianFilterLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.nn.depthwise_conv2d(inputs, self.gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == "__main__":
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    tf.enable_eager_execution()
    image = cv2.imread('../test_images/1.JPG')
    image = image / 255.0
    org_h, org_w, _ = image.shape
    resize_h, resize_w = (129, 129)
    scale = max(resize_w / org_w, resize_h / org_h)
    new_w = int(org_w * scale)
    new_h = int(org_h * scale)
    image = tf.image.resize_images(image, (new_w, new_h))
    image = tf.expand_dims(tf.image.resize_image_with_crop_or_pad(image, 129, 129), axis=0)

    model = tf.keras.Sequential([GaussianFilterLayer(5, 0, 1)])
    output_image = model.predict(image)

    plt.figure()
    plt.imshow(image[0, :, :, ::-1])

    plt.figure()
    plt.imshow(output_image[0, :, :, ::-1])

    plt.figure()
    plt.imshow(image[0, :, :, ::-1] - output_image[0, :, :, ::-1])
    plt.show()
    print()
