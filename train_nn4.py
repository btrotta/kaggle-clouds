import numpy as np
import os
from numba import jit
import keras as ks
import segmentation_models as sm

np.random.seed(0)

kernel_mode = False
if kernel_mode:
    base_dir = os.path.join('/kaggle', 'input', 'understanding_cloud_organization')
    base_dir_data = os.path.join('/kaggle', 'input', 'get-image-arrays-grey-full')
    output_dir_base = os.path.join('/kaggle', 'working')
else:
    base_dir = 'data'
    base_dir_data = 'data'
    output_dir_base = 'data'

def train(use_all_data):

    if use_all_data:
        train_data = np.load(os.path.join(base_dir, 'train_data.npy'))
        calib_data = np.load(os.path.join(base_dir, 'calib_data.npy'))
        test_data = np.load(os.path.join(base_dir, 'test_data.npy'))
        train_data = np.concatenate([train_data, calib_data, test_data], axis=0)
        train_masks = np.load(os.path.join(base_dir, 'train_masks.npy'))
        calib_masks = np.load(os.path.join(base_dir, 'calib_masks.npy'))
        test_masks = np.load(os.path.join(base_dir, 'test_masks.npy'))
        train_masks = np.concatenate([train_masks, calib_masks, test_masks], axis=0)
    else:
        train_data = np.load(os.path.join(base_dir, 'train_data.npy'))
        train_masks = np.load(os.path.join(base_dir, 'train_masks.npy'))

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


    backbone = sm.Unet(backbone_name='efficientnetb5', encoder_weights='imagenet', encoder_freeze=True, classes=4)
    inputs = ks.layers.Input((350, 525, 1))
    inputs_padded = ks.layers.ZeroPadding2D(((1, 1), (10, 9)))(inputs)
    inputs_3_channel = ks.layers.Lambda(lambda x: ks.backend.repeat_elements(x, 3, 3))(inputs_padded)
    outputs = backbone(inputs_3_channel)
    outputs_cropped = ks.layers.Cropping2D(((1, 1), (10, 9)))(outputs)

    # data generators
    imgen = ks.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    maskgen = ks.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    batch_size = 4
    if use_all_data:
        train_im_gen = imgen.flow(train_data, batch_size=batch_size, seed=0, shuffle=True)
        train_mask_gen = maskgen.flow(train_masks, batch_size=batch_size, seed=0, shuffle=True)
        train_gen = zip(train_im_gen, train_mask_gen)
    else:
        valid_frac = 0.1
        num_valid_samples = int(train_data.shape[0] * valid_frac)
        train_im_gen = imgen.flow(train_data[:-num_valid_samples, :, :], batch_size=batch_size, seed=0, shuffle=True)
        train_mask_gen = maskgen.flow(train_masks[:-num_valid_samples:, :, :, :], batch_size=batch_size, seed=0,
                                      shuffle=True)
        train_gen = zip(train_im_gen, train_mask_gen)

    # fit model
    est = ks.models.Model(inputs, outputs_cropped)
    est.compile(optimizer=ks.optimizers.Adam(0.0005), loss='binary_crossentropy')
    # train decoder
    if use_all_data:
        est.fit_generator(train_gen, epochs=10, steps_per_epoch=np.ceil(train_data.shape[0] / batch_size), verbose=2)
    else:
        est.fit_generator(train_gen, epochs=10,
                          steps_per_epoch=np.ceil(train_data.shape[0] * (1 - valid_frac) / batch_size),
                          validation_data=(
                          train_data[-num_valid_samples:, :, :], train_masks[-num_valid_samples:, :, :, :]),
                          verbose=2)
    # unfreeze encoder and fine tune
    for layer in est.layers:
        layer.trainable = True
    est.compile(optimizer=ks.optimizers.Adam(0.0002), loss='binary_crossentropy')
    if use_all_data:
        est.fit_generator(train_gen, epochs=10, steps_per_epoch=np.ceil(train_data.shape[0] / batch_size), verbose=2)
    else:
        est.fit_generator(train_gen, epochs=10,
                          steps_per_epoch=np.ceil(train_data.shape[0] * (1 - valid_frac) / batch_size),
                          validation_data=(
                          train_data[-num_valid_samples:, :, :], train_masks[-num_valid_samples:, :, :, :]),
                          verbose=2)

    if use_all_data:
        est.save(os.path.join(output_dir_base, 'nn_est_full.h5'))
    else:
        est.save(os.path.join(output_dir_base, 'nn_est.h5'))

train(False)
train(True)
