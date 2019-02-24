import numpy as np
import tensorflow as tf
from utils.tensorboard_logging import log_images


class SavePredImage(tf.keras.callbacks.Callback):
    def __init__(self, callback, images, images_label):
        super().__init__()
        self.callback = callback
        self.loss_minimum = 4.5
        self.images = images
        self.images_label = images_label
        self.dummy = np.expand_dims(np.zeros_like(images[0]), axis=0)

    def on_train_begin(self, logs=None):
        for img_id, img in zip(self.images_label, self.images):
            log_images(self.callback, 'image/' + img_id, [img[..., ::-1]], 0)

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('val_loss')
        if loss < self.loss_minimum:
            for img_id, img in zip(self.images_label, self.images):
                img = np.expand_dims(img, axis=0)
                _, _, pred_images = self.model.predict([img, self.dummy])
                pred_images[pred_images < 0] = 0
                pred_images[pred_images > 1] = 1
                pred_images = pred_images[:, :, :, ::-1]
                log_images(self.callback, 'image/'+img_id, pred_images, epoch)
            self.loss_minimum = loss

