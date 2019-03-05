import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob


def load_data_path(dataset_path):
    # Path
    dataset_path_1 = os.path.join(dataset_path, 'Dataset_Part1')
    dataset_path_2 = os.path.join(dataset_path, 'Dataset_Part2')
    data_list_1 = os.listdir(dataset_path_1)

    # Test data index
    with open(os.path.join(dataset_path, 'testing_index.txt')) as f:
        line = f.readline()
        test_files = line.split()

    # Dir data
    data_list_1.remove('Label')
    for file in test_files:
        data_list_1.remove(file)
    data_list_2 = os.listdir(dataset_path_2)
    data_list_2.remove('Label')

    # Train data
    x_train_1 = [path for i in data_list_1 for path in glob(os.path.join(dataset_path_1, i + '/*'))]
    x_train_2 = [path for i in data_list_2 for path in glob(os.path.join(dataset_path_2, i + '/*'))]
    x_train = sorted(x_train_1 + x_train_2)
    y_train = []
    for x_data_path in x_train:
        dirs, filename = x_data_path.split('/')[-3:-1]
        if (dirs == 'Dataset_Part2') and int(filename) > 125:
            y_data_path = os.path.join(dataset_path, dirs, 'Label', filename + '.PNG')
        else:
            y_data_path = os.path.join(dataset_path, dirs, 'Label', filename + '.JPG')
        y_train.append(y_data_path)

    # y_train_1 = [os.path.join(dataset_path_1, 'Label/' + i + '.JPG') for i in data_list_1]
    # y_train_2 = [os.path.join(dataset_path_2, 'Label/' + i + '.JPG') if int(i) < 126 else
    #              os.path.join(dataset_path_2, 'Label/' + i + '.PNG') for i in data_list_2]
    # y_train = sorted(y_train_1 + y_train_2)

    # Test data
    x_test = sorted([path for i in test_files for path in glob(os.path.join(dataset_path_1, i + '/*'))])
    y_test = sorted([os.path.join(dataset_path_1, 'Label/' + i + '.JPG') for i in test_files])
    return (x_train, y_train), (x_test, y_test)


def train_image_process(x_file_path, y_file_path):
    x_image = cv2.imread(x_file_path)
    y_image = cv2.imread(y_file_path)
    # if x_image.shape == y_image.shape:
    #     return x_image, y_image

    image = np.concatenate([x_image, y_image], axis=2)
    org_h, org_w, _ = image.shape
    resize_h, resize_w = (162, 162)
    scale = max(resize_w / org_w, resize_h / org_h)
    new_w = int(org_w * scale)
    new_h = int(org_h * scale)
    image = tf.image.resize_images(image, (new_h, new_w))
    image = tf.random_crop(image, [129, 129, 6])    # 80%
    # image = tf.image.resize_image_with_crop_or_pad(image, new_w, new_h)
    # image = tf.image.resize_images(image, (129, 129))
    image = image.numpy()
    x_image = image[:, :, :3]
    y_image = image[:, :, 3:]
    return x_image, y_image


def test_image_process(x_file_path, y_file_path):
    x_image = cv2.imread(x_file_path)
    y_image = cv2.imread(y_file_path)

    image = np.concatenate([x_image, y_image], axis=2)
    org_h, org_w, _ = image.shape
    resize_h, resize_w = (256, 256)
    scale = max(resize_w / org_w, resize_h / org_h)
    new_w = int(org_w * scale)
    new_h = int(org_h * scale)
    image = tf.image.resize_images(image, (new_h, new_w))
    image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
    image = image.numpy()
    x_image = image[:, :, :3]
    y_image = image[:, :, 3:]
    return x_image, y_image


def create_resize_data(save_path, x_train_path, y_train_path, x_test_path, y_test_path):
    # Create training data
    error_images = []
    x_dir_name = os.path.join(save_path, 'x_train-2')
    os.makedirs(x_dir_name, exist_ok=True)

    y_dir_name = os.path.join(save_path, 'y_train-2')
    os.makedirs(y_dir_name, exist_ok=True)
    for x_path, y_path in zip(x_train_path, y_train_path):
        x_first_dir, x_second_dir, _x_filename = x_path.split('/')[-3::]
        _x_filename = _x_filename.split('.JPG')[0]
        for j in range(3):
            x_filename = "{}_{}_{}_{}.jpg".format(x_first_dir, x_second_dir, _x_filename, j)
            y_filename = "{}_{}_{}_{}.jpg".format(y_path.split('/')[-3], y_path.split('/')[-1].split('.')[0], _x_filename, j)

            print(x_filename)
            print(y_filename)
            try:
                x_image, y_image = train_image_process(x_path, y_path)
            except:
                error_images.append(x_filename)
                break
            data_save_path = os.path.join(x_dir_name, x_filename)
            cv2.imwrite(data_save_path, x_image)
            data_save_path = os.path.join(y_dir_name, y_filename)
            cv2.imwrite(data_save_path, y_image)
    print(error_images)

    # Create testing data
    error_images = []
    x_dir_name = os.path.join(save_path, 'x_test')
    os.makedirs(x_dir_name, exist_ok=True)

    y_dir_name = os.path.join(save_path, 'y_test')
    os.makedirs(y_dir_name, exist_ok=True)
    for x_path, y_path in zip(x_test_path, y_test_path):
        _x_filename = os.path.split(x_path)[0]
        y_filepath, _y_filename = os.path.split(y_path)
        x_filename = "{}_{}.jpg".format(os.path.split(os.path.split(_x_filename)[0])[1], os.path.split(_x_filename)[1])
        y_filename = "{}_{}.jpg".format(os.path.split(y_filepath.split('/Label')[0])[1], _y_filename.split('.')[0])

        print(x_filename)
        print(y_filename)
        try:
            x_image, y_image = test_image_process(x_path, y_path)
        except:
            error_images.append(x_filename)
            break
        data_save_path = os.path.join(x_dir_name, x_filename)
        cv2.imwrite(data_save_path, x_image)
        # print(data_save_path)
        data_save_path = os.path.join(y_dir_name, y_filename)
        cv2.imwrite(data_save_path, y_image)
        # print(data_save_path)
    print(error_images)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset_path_ = '/home/share/dataset/SICE_data'

    # Create org test image
    # create_test_org_data(dataset_path_)

    # Create resize image
    tf.enable_eager_execution()
    (x_train_path, y_train_path), (x_test_path, y_test_path) = load_data_path(dataset_path_)
    create_resize_data(dataset_path_, x_train_path, y_train_path, x_test_path, y_test_path)
