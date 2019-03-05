import tensorflow as tf
from layer.clip import ClipLayer
from losses.losses_layer import L1Losses, MSELosses
from losses.losses import pass_loss, dssim_loss, mse_loss
from layer.filter import GaussianFilter
from layer.conv import Conv2DTranspose


def detail_enhancement_network(inputs):
    with tf.name_scope('detail_network'):
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(inputs)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(3, 1, 1, activation=tf.nn.tanh)(x)
        x = tf.keras.layers.Add(name='detail_output')([inputs, x])
        return x


def luminance_enhancement_network(inputs):
    with tf.name_scope('luminance_network'):
        x = tf.keras.layers.Conv2D(64, 9, 2)(inputs)
        skip = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 5, 2)(skip)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1)(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, 1)(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = Conv2DTranspose(64, 5, 2, align_shape=skip)(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = Conv2DTranspose(3, 9, 2, align_shape=inputs)(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Concatenate()([x, inputs])
        x = tf.keras.layers.Conv2D(3, 1, 1, activation=tf.nn.tanh, name='luminance_output')(x)
        return x


def whole_image_enhancement_network(inputs):
    with tf.name_scope('whole_image_network'):
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(3, 1, 1)(x)
        x = tf.keras.layers.Add(name='whole_image')([inputs, x])
        return x


def create_train_model():
    inputs = tf.keras.Input(shape=(None, None, 3))
    inputs_y_true = tf.keras.Input(shape=(None, None, 3))

    # Create Image detail
    low_freq_img = GaussianFilter(5, 0, 1)(inputs)
    high_freq_img = tf.keras.layers.Subtract()([inputs, low_freq_img])
    decomposition_model = tf.keras.Model(inputs, [low_freq_img, high_freq_img])

    detail_outputs = detail_enhancement_network(decomposition_model.output[1])
    luminance_outputs = luminance_enhancement_network(decomposition_model.output[0])
    # x = tf.keras.layers.Add()([detail_outputs, luminance_outputs])
    luminance_outputs_clip = ClipLayer()(luminance_outputs)
    x = tf.keras.layers.Add()([detail_outputs, luminance_outputs_clip])
    outputs = whole_image_enhancement_network(x)

    # Calculation luminance loss and detail loss
    true_low_freq, true_high_freq = decomposition_model(inputs_y_true)

    luminance_loss = MSELosses(name='luminance')([luminance_outputs, true_low_freq])
    detail_loss = L1Losses(name='detail')([detail_outputs, true_high_freq])

    # Trainable False
    decomposition_model.trainable = False
    for layer in decomposition_model.layers:
        layer.trainable = False

    model = tf.keras.Model(inputs=[inputs, inputs_y_true], outputs=[detail_loss, luminance_loss, outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),    # optimizer=tf.keras.optimizers.SGD(0.001, 0.9, 0.0001),
                  loss=[pass_loss, pass_loss, dssim_loss],
                  loss_weights=[1, 1, 1])

    outputs_final = ClipLayer()(outputs)
    model_pred = tf.keras.Model(inputs=inputs, outputs=outputs_final)

    return model, model_pred


def create_pred_model():
    inputs = tf.keras.Input(shape=(129, 129, 3))

    # Create Image detail
    gaussian_output = GaussianFilter(5, 0, 1)(inputs)
    image_detail = tf.keras.layers.Subtract()([inputs, gaussian_output])

    # Main
    detail_outputs = detail_enhancement_network(image_detail)
    luminance_outputs = luminance_enhancement_network(inputs)
    x = tf.keras.layers.Add()([detail_outputs, luminance_outputs])
    outputs = whole_image_enhancement_network(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model_, pred_model_ = create_train_model()

    model_1 = create_pred_model()
    model_1.load_weights('../sice_weights.h5')
    # for new_layer, org_layer in zip(filter(lambda l: l.weights, model_2.layers[1]), filter(lambda l: l.weights, model_1.layers[2])):
    #     new_layer.set_weights(org_layer.get_weights())
    #     print('"{}" & "{}" match'.format(new_layer.name, org_layer.name))
    #
    model_1.summary()

    # print()
