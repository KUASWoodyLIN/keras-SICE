import os
import cv2
import numpy as np
from glob import glob


def load_data_path(dataset_path):
    x_train = glob(dataset_path+'/x_train/*')
    y_train = glob(dataset_path+'/y_train/*')
    x_test = glob(dataset_path+'/x_test/*')
    y_test = glob(dataset_path+'/y_test/*')

    return (x_train, y_train), (x_test, y_test)


def load_org_data_path(dataset_path):
    # Test data index
    with open(os.path.join(dataset_path, 'testing_index.txt')) as f:
        line = f.readline()
        test_files = line.split()
    dataset_path = os.path.join(dataset_path, 'Dataset_Part1')
    test_data_list = [os.path.join(dataset_path, file + '/1.JPG') for file in test_files]
    test_files_name = [file + '.jpg' for file in test_files]
    return test_data_list, test_files_name


def data_generator_wapper(x_path, y_path, batch_size):
    n = len(x_path)
    if n == 0 or batch_size <= 0:
        return None
    return _data_generator(x_path, y_path, batch_size)


def _data_generator(x_path, y_path, batch_size):
    n = len(x_path)
    i = 0
    index = 0
    while True:
        x_data = []
        y_data = []
        if i == 0:
            index = np.random.permutation(n)
        for b in range(batch_size):
            x_data.append(image_process(x_path[index[i]], True))
            y_data.append(image_process(y_path[index[i]], True))
            i = (i + 1) % n
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        dummy = np.zeros([batch_size, 1])
        yield [x_data, y_data], [dummy, y_data, y_data]


def create_testing_data(x_path, y_path, pred_model=False):
    x_data = []
    y_data = []
    files_name = []
    for i in range(len(x_path)):
        x_data.append(image_process(x_path[i]))
        y_data.append(image_process(y_path[i]))
        files_name.append(os.path.split(x_path[i])[-1])
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    dummy = np.zeros([len(x_path), 1])
    if pred_model:
        return x_data, files_name
    return [x_data, y_data], [dummy, y_data, y_data]


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def image_process(file_path, random=False):
    image = cv2.imread(file_path)
    image = image / 255.
    if not random:
        return image
    flip = rand() < .5
    if flip:
        image = cv2.flip(image, 1)
    return image


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset_path_ = '/home/share/dataset/SICE_data'

    (x_train_, y_train_), (x_test_, y_test_) = load_data_path(dataset_path_)
    generator = data_generator_wapper(x_train_, y_train_, 8)
    x, y = next(generator)

    plt.figure(figsize=(10, 10))
    for i in range(8):
        plt.subplot(4, 4, i + 1)
        # plt.xticks([])
        # plt.yticks([])
        # plt.grid(False)
        plt.imshow(x[0][i][:, :, ::-1])
    for i in range(8):
        plt.subplot(4, 4, i + 9)
        # plt.xticks([])
        # plt.yticks([])
        # plt.grid(False)
        plt.imshow(y[2][i][:, :, ::-1])

    plt.show()
    print()
