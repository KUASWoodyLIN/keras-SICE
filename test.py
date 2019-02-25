import os
import cv2
import numpy as np

from model.model import create_pred_model
from preprocessing.image_processing import load_org_data_path, image_process


def main():
    # Parameter
    dataset_path = '/home/share/dataset/SICE_data'
    # testing_path = os.path.join(dataset_path, 'TestingImage')
    output_path = os.path.join(dataset_path, 'outputs')

    # Create and load model
    model = create_pred_model()
    model.load_weights('logs_1/model/Total-best-ep040-val_loss3.84.h5')

    # Create data
    x_test_path, x_name = load_org_data_path(dataset_path)

    for img_id, img_path in zip(x_name, x_test_path):
        img = image_process(img_path)
        img = np.expand_dims(img, axis=0)
        img_out = model.predict(img)
        cv2.imwrite(os.path.join(output_path, img_id), img_out)
        print('Save {}'.format(output_path))


if __name__ == '__main__':
    main()
