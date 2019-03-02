import os
import cv2
import numpy as np

from glob import glob
from model.sice import create_pred_model, create_train_model
from preprocessing.image_processing import load_org_data_path, image_process
from preprocessing.create_dataset import load_data_path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # Parameter
    dataset_path = '/home/share/dataset/SICE_data'
    dataset = 'sice_test'    # sice_train, sice_test or test

    if dataset == 'sice_train':
        output_path = os.path.join(dataset_path, 'outputs_sice_train')
        os.makedirs(output_path, exist_ok=True)
        (x_test_path, _), (_, _) = load_data_path(dataset_path)
        x_name = [path.split('SICE_data/')[-1].split('/1.')[0].replace('/', '_') + '.jpg' for path in x_test_path]

    elif dataset == 'sice_test':
        output_path = os.path.join(dataset_path, 'outputs_sice_test')
        os.makedirs(output_path, exist_ok=True)
        x_test_path, x_name = load_org_data_path(dataset_path)

    else:
        testing_path = os.path.join(dataset_path, 'TestingImage')
        output_path = os.path.join(dataset_path, 'outputs_testingimage')
        os.makedirs(output_path, exist_ok=True)
        x_test_path = glob(testing_path + '/*')
        x_name = [os.path.split(file_path)[-1] for file_path in x_test_path]

    # Create and load model
    # model = create_pred_model()
    # model.load_weights('sice_weights.h5')
    # model = create_pred_model_2()
    # model.load_weights('sice_weights.h5', True)
    model, pred_model = create_train_model()
    # model.load_weights('./logs-1/model/Epoch{epoch:03d}.h5')
    model.load_weights('Total-best-ep176-val_loss15324.76.h5')

    # model.load_weights('./sice_weights.h5', True)

    # Predict data
    for img_id, img_path in zip(x_name, x_test_path):
        print('Processing {}'.format(img_id))
        img = image_process(img_path)
        if dataset == 'sice_train' or dataset == 'sice_test':
            org_h, org_w, _ = img.shape
            resize_h, resize_w = 1024, 1024
            scale = max(resize_w / org_w, resize_h / org_h)
            new_h = int(org_h * scale)
            new_w = int(org_w * scale)
            img = cv2.resize(img, (new_w, new_h))

        img = np.expand_dims(img, axis=0)
        img_out = pred_model.predict(img)
        # img_1, img_2, img_3 = pred_model.predict(img)

        img_out = img_out * 255
        cv2.imwrite(os.path.join(output_path, img_id), img_out[0])


if __name__ == '__main__':
    main()
