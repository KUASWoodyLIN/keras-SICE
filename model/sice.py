import tensorflow as tf
from model.losses_layer import L1Losses
from model.clip_layer import ClipLayer
from model.losses import pass_loss, dssim_loss, mse_loss
from model.filter import GaussianFilter
from model.conv import Conv2DTranspose


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
        x = tf.keras.layers.Conv2D(3, 1, 1)(x)
        x = tf.keras.layers.Add()([inputs, x])
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
        x = tf.keras.layers.Conv2D(3, 1, 1, name='luminance')(x)
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
    # inputs = tf.keras.Input(shape=(129, 129, 3))
    # inputs_y = tf.keras.Input(shape=(129, 129, 3))
    inputs = tf.keras.Input(shape=(None, None, 3))
    inputs_y = tf.keras.Input(shape=(None, None, 3))

    # Create Image detail
    gaussian_output = GaussianFilter(5, 0, 1)(inputs)
    image_detail = tf.keras.layers.Subtract()([inputs, gaussian_output])
    detail_model = tf.keras.Model(inputs, image_detail)

    detail_outputs = detail_enhancement_network(detail_model.output)
    luminance_outputs = luminance_enhancement_network(inputs)
    luminance_outputs = ClipLayer()(luminance_outputs)
    x = tf.keras.layers.Add()([detail_outputs, luminance_outputs])
    outputs = whole_image_enhancement_network(x)

    # Calculation detail loss
    true_detail = detail_model(inputs_y)
    detail_loss = L1Losses(name='detail')([detail_outputs, true_detail])

    # Trainable False
    detail_model.trainable = False
    for layer in detail_model.layers:
        layer.trainable = False

    model = tf.keras.Model(inputs=[inputs, inputs_y], outputs=[detail_loss, luminance_outputs, outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),    # optimizer=tf.keras.optimizers.SGD(0.001, 0.9, 0.0001),
                  loss=[pass_loss, mse_loss, dssim_loss],
                  loss_weights=[1, 1, 1])

    outputs_final = ClipLayer()(outputs)
    model_pred = tf.keras.Model(inputs=inputs, outputs=outputs_final)
    # model_pred = tf.keras.Model(inputs=inputs, outputs=[detail_outputs, luminance_outputs, outputs])

    # model_.load_weights('../logs/model/Total-best-ep082-val_loss3.35.h5')
    # model_pred_.save_weights('../sice_weights.h5')

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

    # model_2 = create_pred_model_2()
    # model_2.load_weights('../sice_weights.h5', True)
    #
    # model_1 = create_pred_model()
    # model_1.load_weights('../sice_weights.h5')
    # # for new_layer, org_layer in zip(filter(lambda l: l.weights, model_2.layers[1]), filter(lambda l: l.weights, model_1.layers[2])):
    # #     new_layer.set_weights(org_layer.get_weights())
    # #     print('"{}" & "{}" match'.format(new_layer.name, org_layer.name))
    # #
    # model_1.summary()
    # model_2.summary()
    #
    # print()
