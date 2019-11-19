import os
import pandas as pd
import numpy as np
import prepare_data
from skimage import transform
from numba import jit
from multiprocessing import Pool


if __name__ == '__main__':
    pd.options.display.width = 200
    pd.options.display.max_columns = 20

    test_mode = True
    kernel_mode = False
    if kernel_mode:
        base_dir = os.path.join('/kaggle', 'input', 'understanding_cloud_organization')
        output_dir_base = os.path.join('/kaggle', 'working')
    else:
        base_dir = 'data'
        output_dir_base = 'data'

    num_calibration_samples = 500
    num_testing_samples = 100
    num_training_samples = len(os.listdir(os.path.join(base_dir, 'train_images'))) - num_calibration_samples - num_testing_samples

    def get_im_arr(folder, folder_list, save_name):
        args = [(folder, f) for f in folder_list]
        with Pool() as p:
            res = p.starmap_async(prepare_data.process_one_image, args)
            im_arr = res.get()
        data = np.concatenate([x[0][np.newaxis, :, :, np.newaxis] for x in im_arr], axis=0)
        np.save(os.path.join(output_dir_base, save_name), data)
        names = [x[1] for x in im_arr]
        pd.DataFrame({'names': names}).to_csv(os.path.join(output_dir_base, '{}.csv'.format(save_name)), index=False,
                                              header=True)


    train_folder = os.path.join(base_dir, 'train_images')
    train_folder_list = os.listdir(train_folder)[:num_training_samples]
    get_im_arr(train_folder, train_folder_list, 'train_data')
    calib_folder_list = os.listdir(train_folder)[-(num_calibration_samples + num_testing_samples):-num_testing_samples]
    get_im_arr(train_folder, calib_folder_list, 'calib_data')
    test_folder_list = os.listdir(train_folder)[-num_testing_samples:]
    get_im_arr(train_folder, test_folder_list, 'test_data')
    submit_folder = os.path.join(base_dir, 'test_images')
    submit_folder_list = os.listdir(submit_folder)
    get_im_arr(submit_folder, submit_folder_list, 'submit_data')


    @jit
    def add_label(pixel_arr, label_arr_shape, label):
        label_arr = np.zeros(label_arr_shape)
        height, width = label_arr.shape
        for j in range(len(pixel_arr) // 2):
            start, length = pixel_arr[2 * j] - 1, pixel_arr[2 * j + 1]
            for k in range(length):
                p = start + k
                row = p % height
                col = p // height
                label_arr[row, col] = label
        return label_arr


    def get_masks(file_list, labels):
        mask = np.zeros((len(file_list), 350, 525, 4), np.int8)
        for i, f in enumerate(file_list):
            for j, cat in enumerate(['Fish', 'Flower', 'Gravel', 'Sugar']):
                pixels = labels.loc[labels['Image_Label'] == f + '_' + cat, 'EncodedPixels'].values[0]
                if type(pixels) != str:
                    mask[i, :, :, j] = np.zeros((350, 525), np.int8)
                else:
                    pixel_arr = np.array([int(p) for p in pixels.split(' ')])
                    class_mask = add_label(pixel_arr, (1400, 2100), 1)
                    class_mask = transform.downscale_local_mean(class_mask, (4, 4))
                    mask[i, :, :, j] = (class_mask > 0.5).astype(np.int8)
        return mask

    labels = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    train_names = pd.read_csv(os.path.join(base_dir, 'train_data.csv'))['names'].values
    mask = get_masks(train_names, labels)
    np.save(os.path.join(output_dir_base, 'train_masks'), mask)
    calib_names = pd.read_csv(os.path.join(base_dir, 'calib_data.csv'))['names'].values
    mask = get_masks(calib_names, labels)
    np.save(os.path.join(output_dir_base, 'calib_masks'), mask)
    test_names = pd.read_csv(os.path.join(base_dir, 'test_data.csv'))['names'].values
    mask = get_masks(test_names, labels)
    np.save(os.path.join(base_dir, 'test_masks'), mask)
