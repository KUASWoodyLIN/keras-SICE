import os
import cv2
import tensorflow as tf


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
    x_train_1 = [os.path.join(dataset_path_1, i + '/1.JPG') for i in data_list_1]
    x_train_2 = [os.path.join(dataset_path_2, i + '/1.JPG') for i in data_list_2]
    x_train = x_train_1 + x_train_2
    y_train_1 = [os.path.join(dataset_path_1, 'Label/' + i + '.JPG') for i in data_list_1]
    y_train_2 = [os.path.join(dataset_path_2, 'Label/' + i + '.JPG') if int(i) < 126 else 
                 os.path.join(dataset_path_2, 'Label/' + i + '.PNG') for i in data_list_2 ]
    y_train = y_train_1 + y_train_2

    # Test data
    x_test = [os.path.join(dataset_path_1, i + '/1.JPG') for i in test_files]
    y_test = [os.path.join(dataset_path_1, 'Label/' + i + '.JPG') for i in test_files]
    return (x_train, y_train), (x_test, y_test)


def image_process(file_path, input_shape):
    image = cv2.imread(file_path)
    org_h, org_w, _ = image.shape
    resize_h, resize_w = input_shape
    scale = max(resize_w / org_w, resize_h / org_h)
    new_w = int(org_w * scale)
    new_h = int(org_h * scale)
    image = tf.image.resize_images(image, (new_w, new_h))
    image = tf.image.resize_image_with_crop_or_pad(image, 129, 129)

    return image.numpy()


def create_resize_data(save_path, x_train_path, y_train_path, x_test_path, y_test_path):
    for i, (dir_name, data_org_path) in enumerate(zip(['x_train', 'x_test'], [x_train_path, x_test_path])):
        dir_name = os.path.join(save_path, dir_name)
        os.makedirs(dir_name, exist_ok=True)
        for path in data_org_path:
            filename = os.path.split(path)[0]
            filename = "{}_{}".format(os.path.split(os.path.split(filename)[0])[1], os.path.split(filename)[1] + '.jpg')
            image = image_process(path, (129, 129))
            data_save_path = os.path.join(dir_name, filename)
            cv2.imwrite(data_save_path, image)
            print(data_save_path)
    count = 0
    print('train data {}'.format(len(x_train_path)))
    print('test data {}'.format(len(x_test_path)))

    for i, (dir_name, data_org_path) in enumerate(zip(['y_train', 'y_test'], [y_train_path, y_test_path])):
        dir_name = os.path.join(save_path, dir_name)
        os.makedirs(dir_name, exist_ok=True)
        for path in data_org_path:
            filepath, filename = os.path.split(path)
            filename = "{}_{}".format(os.path.split(filepath.split('/Label')[0])[1], filename.split('.')[0] + '.jpg')
            print(count, path)
            image = image_process(path, (129, 129))
            data_save_path = os.path.join(dir_name, filename)
            cv2.imwrite(data_save_path, image)
            count+=1


if __name__ == '__main__':
    tf.enable_eager_execution()
    dataset_path_ = '/home/share/dataset/SICE_data'
    (x_train_path, y_train_path), (x_test_path, y_test_path) = load_data_path(dataset_path_)
    create_resize_data(dataset_path_, x_train_path, y_train_path, x_test_path, y_test_path)
