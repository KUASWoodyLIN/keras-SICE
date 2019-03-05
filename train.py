import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from utils.callback import SavePredImageEpoch, SavePredImageBest, CustomReduceLR, SaveModelEpochs, SaveOutputHistogram
from model.sice import create_train_model
from preprocessing.image_processing import load_data_path_v2, load_data_path, data_generator_wapper, create_testing_data, image_process

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # Parameter
    dataset_path = '/home/share/dataset/SICE_data'
    batch_size = 32
    logs_dir = 'logs-1'
    os.makedirs(logs_dir + '/model/')

    # Create data
    # (x_train, y_train), (_, _) = load_data_path_v2(dataset_path)
    (x_train, y_train), (x_test, y_test) = load_data_path(dataset_path)

    train_num = len(x_train) * 7 // 8
    x_train, y_train, x_valid, y_valid = x_train[:train_num], y_train[:train_num], x_train[train_num:], y_train[train_num:]
    step = len(x_train) // batch_size
    train_generator = data_generator_wapper(x_train, y_train, batch_size)
    x_valid, y_valid = create_testing_data(x_valid, y_valid)
    x_test, x_name = create_testing_data(x_test, y_test, pred_model=True)

    # testing_path = os.path.join(dataset_path, 'TestingImage')
    # output_path = os.path.join(dataset_path, 'outputs_testingimage')
    # os.makedirs(output_path, exist_ok=True)
    # x_test_path = glob(testing_path + '/*')
    # x_name = [os.path.split(file_path)[-1] for file_path in x_test_path]
    # x_test = []
    # for image_path in x_test_path:
    #     img = image_process(image_path)
    #     org_h, org_w, _ = img.shape
    #     img = cv2.resize(img, (256, 256))
    #     x_test.append(img)
    # x_test = np.asarray(x_test)

    # Create model
    model, pred_model = create_train_model()
    # model.load_weights('sice_weights_new.h5')
    model.load_weights('./logs-2-2/model/Total-best-ep047-val_loss13747.00.h5', True)

    # Checkpoint
    callback = tf.keras.callbacks.TensorBoard(logs_dir, histogram_freq=1)
    callback.set_model(model)
    save_img_best = SavePredImageBest(callback, x_test, x_name, pred_model)
    save_img_every_epoch = SavePredImageEpoch(callback, x_test, x_name, 50, pred_model)
    save_histogram = SaveOutputHistogram(callback, x_test[0])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        logs_dir + '/model/Total-best-ep{epoch:03d}-val_loss{val_loss:.2f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        mode='min',
        period=1)
    checkpoint_2 = SaveModelEpochs(logs_dir + '/model/Epoch{epoch:03d}.h5', 50)
    # reduce_lr = CustomReduceLR(0.1, 30)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.8, patience=50, min_lr=1e-6)

    # Training    model.load_weights('./logs-2-best/model/Total-best-ep176-val_loss15324.76.h5')

    model.fit_generator(train_generator,
                        steps_per_epoch=step,
                        epochs=4000,
                        validation_data=(x_valid, y_valid),
                        callbacks=[save_img_best, save_histogram, save_img_every_epoch, checkpoint, checkpoint_2, callback])


if __name__ == '__main__':
    main()
