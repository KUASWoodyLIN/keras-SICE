import os
import tensorflow as tf

from utils.callback import SavePredImage, CustomReduceLR, SaveModelEpochs
from model.sice import create_train_model
from preprocessing.image_processing import load_data_path, data_generator_wapper, create_testing_data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # Parameter
    dataset_path = '/home/share/dataset/SICE_data'
    batch_size = 16
    logs_dir = 'logs-2'
    os.makedirs(logs_dir + '/model/')

    # Create data
    (x_train, y_train), (x_test, y_test) = load_data_path(dataset_path)
    train_num = len(x_train) * 7 // 8
    x_train, y_train, x_valid, y_valid = x_train[:train_num], y_train[:train_num], x_train[train_num:], y_train[train_num:]
    step = len(x_train) // batch_size
    train_generator = data_generator_wapper(x_train, y_train, batch_size)
    x_valid, y_valid = create_testing_data(x_valid, y_valid)
    x_test, x_name = create_testing_data(x_test, y_test, pred_model=True)

    # Create model
    model, pred_model = create_train_model()
    # model.load_weights('sice_weights_new.h5')
    model.load_weights('./logs-2-2/model/Total-best-ep047-val_loss13747.00.h5', True)

    # Checkpoint
    callback = tf.keras.callbacks.TensorBoard(logs_dir)
    callback.set_model(model)
    save_img = SavePredImage(callback, x_test, x_name, pred_model)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        logs_dir + '/model/Total-best-ep{epoch:03d}-val_loss{val_loss:.2f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        mode='min',
        period=1)
    checkpoint_2 = SaveModelEpochs(logs_dir + '/model/Epoch{epoch:03d}.h5', 50)
    # reduce_lr = CustomReduceLR(0.1, 30)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=30, min_lr=1e-6)

    # Training
    model.fit_generator(train_generator,
                        steps_per_epoch=step,
                        epochs=4000,
                        validation_data=(x_valid, y_valid),
                        callbacks=[reduce_lr, save_img, checkpoint, checkpoint_2, callback])


if __name__ == '__main__':
    main()
