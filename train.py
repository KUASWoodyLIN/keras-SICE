import os
import numpy as np
import tensorflow as tf

from utils.callback import SavePredImage
from model.model import create_train_model
from preprocessing.image_processing import load_data_path, data_generator_wapper, create_testing_data


def main():
    # Parameter
    dataset_path = '/home/share/dataset/SICE_data'
    batch_size = 16
    os.makedirs('./logs/model/')

    # Create data
    (x_train, y_train), (x_test, y_test) = load_data_path(dataset_path)
    train_num = len(x_train) * 7 // 8
    x_train, y_train, x_valid, y_valid = x_train[:train_num], y_train[:train_num], x_train[train_num:], y_train[train_num:]
    step = len(x_train) // batch_size
    train_generator = data_generator_wapper(x_train, y_train, batch_size)
    x_valid, y_valid = create_testing_data(x_valid, y_valid)
    x_test, x_name = create_testing_data(x_test, y_test, pred_model=True)

    # Create model
    model = create_train_model()

    # Checkpoint
    callback = tf.keras.callbacks.TensorBoard('./logs')
    callback.set_model(model)
    save_img = SavePredImage(callback, x_test, x_name)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './logs/model/Total-best-ep{epoch:03d}-val_loss{val_loss:.2f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        mode='min',
        period=1)

    # Training
    model.fit_generator(train_generator,
                        steps_per_epoch=step,
                        epochs=100,
                        validation_data=(x_valid, y_valid),
                        callbacks=[save_img, checkpoint])


if __name__ == '__main__':
    main()
