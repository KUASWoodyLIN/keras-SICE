import os
import cv2
import numpy as np
from glob import glob
from preprocessing.create_dataset import load_data_path


def resize(img, reshape=(1024, 1024)):
    resize_h, resize_w = reshape
    img = cv2.resize(img, (resize_w, resize_h))
    return img


dataset_path = '/home/share/dataset/SICE_data'
dataset = 'sice_test'    # sice_train, sice_test or test

if dataset == 'sice_train':
    (x_data, y_data), (_, _) = load_data_path(dataset_path)
    pred_path = os.path.join(dataset_path, 'outputs_sice_train_no_tanh/')       # outputs_sice_train_no_tanh or outputs_sice_train
    pred_data = [pred_path + path.split('SICE_data/')[-1].split('/1.')[0].replace('/', '_') + '.jpg' for path in x_data]
    combine_data_path = os.path.join(dataset_path, 'combine_sice_train_no_tanh')
    os.makedirs(combine_data_path, exist_ok=True)

    for img_1, img_2, img_3 in zip(x_data, pred_data, y_data):
        file_name = os.path.split(img_2)[-1]
        img2 = cv2.imread(img_2)
        h, w, _ = img2.shape

        img1 = cv2.imread(img_1)
        img1 = resize(img1, (h, w))

        img3 = cv2.imread(img_3)
        img3 = resize(img3, (h, w))

        combine_img = np.hstack([img1, img2, img3])

        cv2.imwrite(os.path.join(combine_data_path, file_name), combine_img)
        print(file_name)
elif dataset == 'sice_test':
    # x_data_path = os.path.join(dataset_path, 'x_test_org')
    # pred_data_path = os.path.join(dataset_path, 'outputs_sice_test_no_tanh')
    # y_data_path = os.path.join(dataset_path, 'y_test_org')
    # combine_data_path = os.path.join(dataset_path, 'combine_sice_test_no_tanh')
    # os.makedirs(combine_data_path, exist_ok=True)
    #
    # x_data = sorted(glob(x_data_path + '/*'))
    # pred_data = sorted(glob(pred_data_path + '/*'))
    # y_data = sorted(glob(y_data_path + '/*'))
    #
    # for img_1, img_2, img_3 in zip(x_data, pred_data, y_data):
    #     file_name = os.path.split(img_3)[-1]
    #     img2 = cv2.imread(img_2)
    #     h, w, _ = img2.shape
    #
    #     img1 = cv2.imread(img_1)
    #     img1 = resize(img1, (h, w))
    #
    #     img3 = cv2.imread(img_3)
    #     img3 = resize(img3, (h, w))
    #
    #     combine_img = np.hstack([img1, img2, img3])
    #
    #     cv2.imwrite(os.path.join(combine_data_path, file_name), combine_img)
    #     print(file_name)
    x_data_path = os.path.join(dataset_path, 'x_test_org')
    pred_data_path1 = os.path.join(dataset_path, 'outputs_sice_test_epochs_15')
    pred_data_path2 = os.path.join(dataset_path, 'outputs_sice_test_epochs_1487')

    y_data_path = os.path.join(dataset_path, 'y_test_org')
    combine_data_path = os.path.join(dataset_path, 'combine_sice_test_ep_15and1487')
    os.makedirs(combine_data_path, exist_ok=True)

    x_data = sorted(glob(x_data_path + '/*'))
    pred_data1 = sorted(glob(pred_data_path1 + '/*'))
    pred_data2 = sorted(glob(pred_data_path2 + '/*'))

    y_data = sorted(glob(y_data_path + '/*'))

    for img_1, img_2, img_3, img_4 in zip(x_data, pred_data1, pred_data2, y_data):
        file_name = os.path.split(img_4)[-1]
        img2 = cv2.imread(img_2)
        h, w, _ = img2.shape
        img3 = cv2.imread(img_3)

        img1 = cv2.imread(img_1)
        img1 = resize(img1, (h, w))

        img4 = cv2.imread(img_4)
        img4 = resize(img4, (h, w))

        combine_img = np.hstack([img1, img2, img3, img4])

        cv2.imwrite(os.path.join(combine_data_path, file_name), combine_img)
        print(file_name)
else:
    # x_data_path = os.path.join(dataset_path, 'TestingImage')
    # pred_data_path = os.path.join(dataset_path, 'outputs_testingimage')
    # combine_data_path = os.path.join(dataset_path, 'combine_test')
    # os.makedirs(combine_data_path, exist_ok=True)
    #
    # x_data = sorted(glob(x_data_path + '/*'))
    # pred_data = sorted(glob(pred_data_path + '/*'))
    #
    # for img_1, img_2 in zip(x_data, pred_data):
    #     file_name = os.path.split(img_1)[-1]
    #     img1 = cv2.imread(img_1)
    #     img2 = cv2.imread(img_2)
    #
    #     combine_img = np.hstack([img1, img2])
    #
    #     cv2.imwrite(os.path.join(combine_data_path, file_name), combine_img)
    #     print(file_name)

    x_data_path = os.path.join(dataset_path, 'TestingImage')
    pred_data_path1 = os.path.join(dataset_path, 'outputs_testingimage_epochs_15')
    pred_data_path2 = os.path.join(dataset_path, 'outputs_testingimage_epochs_1487')
    combine_data_path = os.path.join(dataset_path, 'combine_test_ep_15and1487')
    os.makedirs(combine_data_path, exist_ok=True)

    x_data = sorted(glob(x_data_path + '/*'))
    pred_data1 = sorted(glob(pred_data_path1 + '/*'))
    pred_data2 = sorted(glob(pred_data_path2 + '/*'))

    for img_1, img_2, img_3 in zip(x_data, pred_data1, pred_data2):
        file_name = os.path.split(img_1)[-1]
        img1 = cv2.imread(img_1)
        img2 = cv2.imread(img_2)
        img3 = cv2.imread(img_3)

        combine_img = np.hstack([img1, img2, img3])

        cv2.imwrite(os.path.join(combine_data_path, file_name), combine_img)
        print(file_name)
