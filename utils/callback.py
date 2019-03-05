import numpy as np
import tensorflow as tf
from utils.tensorboard_logging import log_images, log_histogram


class SavePredImageBest(tf.keras.callbacks.Callback):
    def __init__(self, callback, images, images_label, pred_model=None):
        super().__init__()
        self.callback = callback
        self.loss_minimum = 200000
        self.images = images
        self.images_label = images_label
        self.pred_model = pred_model
        # self.dummy = np.expand_dims(np.zeros_like(images[0]), axis=0)

    def on_train_begin(self, logs=None):
        for img_id, img in zip(self.images_label, self.images):
            log_images(self.callback, 'image/' + img_id, [img[..., ::-1]], 0)

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('val_loss')
        if loss < self.loss_minimum:
            for img_id, img in zip(self.images_label, self.images):
                img = np.expand_dims(img, axis=0)
                # _, _, pred_images = self.model.predict([img, self.dummy])
                pred_images = self.pred_model.predict(img)

                pred_images = pred_images[:, :, :, ::-1]
                log_images(self.callback, 'image/'+img_id, pred_images, epoch)
            self.loss_minimum = loss


class SavePredImageEpoch(tf.keras.callbacks.Callback):
    def __init__(self, callback, images, images_label, every_num_epoch=1, pred_model=None):
        super().__init__()
        self.callback = callback
        self.loss_minimum = 200000
        self.images = images
        self.images_label = images_label
        self.every_num_epoch = every_num_epoch
        self.pred_model = pred_model
        self.tab = 'image_every_{}_epoch/'.format(every_num_epoch)
        # self.dummy = np.expand_dims(np.zeros_like(images[0]), axis=0)

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('val_loss')
        if epoch % self.every_num_epoch == 0:
            for img_id, img in zip(self.images_label, self.images):
                img = np.expand_dims(img, axis=0)
                # _, _, pred_images = self.model.predict([img, self.dummy])
                pred_images = self.pred_model.predict(img)

                pred_images = pred_images[:, :, :, ::-1]
                log_images(self.callback, self.tab+img_id, pred_images, epoch)
            self.loss_minimum = loss


class SaveOutputHistogram(tf.keras.callbacks.Callback):
    def __init__(self, callback, test_img):
        super().__init__()
        self.callback = callback
        self.test_img = np.expand_dims(test_img, axis=0)

    def on_train_begin(self, logs=None):
        detail_layer = self.model.get_layer('detail_output').output
        luminance_layer = self.model.get_layer('luminance_output').output
        whole_image_layer = self.model.get_layer('whole_image').output

        self.pred_model = tf.keras.backend.function(inputs=[self.model.input[0]],
                                                    outputs=[detail_layer, luminance_layer, whole_image_layer])

        detail, luminance, whole_image = self.pred_model([self.test_img])
        log_histogram(self.callback, 'detail/', detail, -1)
        log_histogram(self.callback, 'luminance/', luminance, -1)
        log_histogram(self.callback, 'whole_image/', whole_image, -1)

    def on_epoch_end(self, epoch, logs=None):
        # detail, luminance, whole_image = self.model.predict([img, self.dummy])
        detail, luminance, whole_image = self.pred_model([self.test_img])
        log_histogram(self.callback, 'detail/', detail, epoch)
        log_histogram(self.callback, 'luminance/', luminance, epoch)
        log_histogram(self.callback, 'whole_image/', whole_image, epoch)


class SaveModelEpochs(tf.keras.callbacks.Callback):
    def __init__(self, filepath, every_num_epoch):
        super(SaveModelEpochs, self).__init__()
        self.filepath = filepath
        self.every_num_epoch = every_num_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 50 == 0 and epoch != 0:
            self.model.save_weights(self.filepath, overwrite=True)


class CustomReduceLR(tf.keras.callbacks.Callback):
    def __init__(self, factor=0.1, patience=10, **kwargs):
        super(CustomReduceLR, self).__init__()
        self.factor = factor
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

        old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        if epoch % 30 == 0 and epoch != 0:
            new_lr = old_lr * self.factor
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

    def in_cooldown(self):
        return self.cooldown_counter > 0