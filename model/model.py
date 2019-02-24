import tensorflow as tf
from model.gaussian_filter import GaussianFilterLayer
from model.losses_layer import L1Losses
from model.losses import pass_loss, dssim_loss


def luminance_enhancement_network(inputs):
    x = tf.keras.layers.Conv2D(64, 9, 2)(inputs)
    skip = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 5, 2)(skip)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1)(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 1)(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 5, 2)(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = tf.keras.layers.Conv2DTranspose(3, 9, 2)(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Concatenate()([x, inputs])
    x = tf.keras.layers.Conv2D(3, 1, 1, name='luminance')(x)
    return x


def detail_enhancement_network(inputs):
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(inputs)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(3, 1, 1)(x)
    return x


def whole_image_enhancement_network(inputs):
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(3, 1, 1)(x)
    x = tf.keras.layers.Add(name='whole_image')([inputs, x])
    return x


def create_train_model():
    inputs = tf.keras.Input(shape=(129, 129, 3))
    inputs_y = tf.keras.Input(shape=(129, 129, 3))

    # Create Image detail
    gaussian_output = GaussianFilterLayer(5, 0, 1)(inputs)
    image_detail = tf.keras.layers.Subtract()([inputs, gaussian_output])
    detail_model = tf.keras.Model(inputs, image_detail)

    detail_outputs = detail_enhancement_network(detail_model.output)
    luminance_outputs = luminance_enhancement_network(inputs)
    x = tf.keras.layers.Add()([detail_outputs, luminance_outputs])
    outputs = whole_image_enhancement_network(x)

    # Calculation detail loss
    true_detail = detail_model(inputs_y)
    detail_outputs = L1Losses(name='detail')([detail_outputs, true_detail])

    # Trainable False
    detail_model.trainable = False
    for layer in detail_model.layers:
        layer.trainable = False

    model = tf.keras.Model(inputs=[inputs, inputs_y], outputs=[detail_outputs, luminance_outputs, outputs])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
                  loss=[pass_loss, 'mse', dssim_loss],
                  loss_weights=[10, 1, 1])

    return model


def create_pred_model():
    inputs = tf.keras.Input(shape=(129, 129, 3))
    detail_outputs = detail_enhancement_network(inputs)
    luminance_outputs = luminance_enhancement_network(inputs)
    x = tf.keras.layers.Add()([detail_outputs, luminance_outputs])
    outputs = whole_image_enhancement_network(x)
    outputs = tf.keras.layers.Add()([outputs, x])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model_ = create_train_model()
    model_.summary()
