import numpy as np
import os
from skimage import io, transform, filters
from numba import jit


@jit(nopython=True)
def get_background(im_arr, t):
    h, w = im_arr.shape[:2]
    background_mask = np.zeros(im_arr.shape, np.int8)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = im_arr[i:i+8, j:j+8]
            if (np.std(block) < t) & (np.mean(block) < 200) & (np.mean(block) > 2):
                background_mask[i:i+8, j:j+8] = 1
    return background_mask


@jit(nopython=True)
def get_background_colour(background_image):
    background_colour = np.zeros((background_image.shape[0]//50, background_image.shape[1]//50), np.float32)
    h, w = background_image.shape
    for i in range(0, h//50):
        for j in range(0, w//50):
            curr_slice = background_image[i*50:(i+1)*50, j*50:(j+1)*50]
            nan_sum = np.sum(np.isnan(curr_slice))
            if nan_sum < 50 * 50:
                background_colour[i, j] = np.nanmean(curr_slice)
    return background_colour


def correct(im_arr, t):
    # get mask for missing areas, we will later colour them grey
    grad1, grad2 = np.gradient(im_arr)
    missing = grad1 + grad2 + im_arr == 0
    background_mask = get_background(im_arr, t).astype(bool)
    background_image = np.where(background_mask, im_arr, np.nan)
    background_colour = get_background_colour(background_image)
    background_colour = np.repeat(np.repeat(background_colour, 50, 0), 50, 1)
    background_colour = filters.gaussian(background_colour, sigma=50, preserve_range=True)
    # rescale so [background_colour_full, 255] maps onto [0, 255]
    im_arr_out = 255 * (im_arr - background_colour) / (255 - background_colour)
    im_arr_out = np.maximum(0, im_arr_out)
    im_arr_out[missing] = 125
    return im_arr_out


def process_one_image(source_folder, f):
    im = io.imread(os.path.join(source_folder, f))
    im = np.mean(im, 2)
    im = transform.downscale_local_mean(correct(im, 5), (4, 4)) / 255 - 0.5
    return im.astype(np.float32), f

