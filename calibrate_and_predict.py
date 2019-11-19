import numpy as np
import pandas as pd
import os
from numba import jit
import keras as ks
from sklearn import metrics
from sklearn import linear_model
import segmentation_models as sm


test_mode = False
kernel_mode = False
if kernel_mode:
    base_dir = os.path.join('/kaggle', 'input', 'understanding_cloud_organization')
    output_dir_base = os.path.join('/kaggle', 'working')
else:
    base_dir = 'data'
    output_dir_base = 'data'


train_folder = os.path.join(base_dir, 'train_images')
train_folder_list = os.listdir(train_folder)
calib_data = np.load(os.path.join(base_dir, 'calib_data.npy'))
test_data = np.load(os.path.join(base_dir, 'test_data.npy'))
calib_names = pd.read_csv(os.path.join(base_dir, 'calib_data.csv'))['names'].values
test_names = pd.read_csv(os.path.join(base_dir, 'test_data.csv'))['names'].values
calib_masks = np.load(os.path.join(base_dir, 'calib_masks.npy'))
test_masks = np.load(os.path.join(base_dir, 'test_masks.npy'))


@jit
def dice_per_image_per_class(Y_true, Y_predicted):
    int_size = ((Y_predicted == 1) & (Y_true == Y_predicted)).sum()
    union_size = Y_true.sum() + Y_predicted.sum()
    if union_size == 0:
        return 1
    else:
        return 2 * int_size / union_size

# load models
backbone = sm.Unet(backbone_name='efficientnetb4', encoder_weights='imagenet', encoder_freeze=True, classes=4)
inputs = ks.layers.Input((350, 525, 1))
inputs_padded = ks.layers.ZeroPadding2D(((1, 1), (10, 9)))(inputs)
inputs_3_channel = ks.layers.Lambda(lambda x: ks.backend.repeat_elements(x, 3, 3))(inputs_padded)
outputs = backbone(inputs_3_channel)
outputs_cropped = ks.layers.Cropping2D(((1, 1), (10, 9)))(outputs)
est1 = ks.models.Model(inputs, outputs_cropped)
est1.load_weights(os.path.join('nn14', 'nn_est.h5'))
backbone = sm.Unet(backbone_name='efficientnetb5', encoder_weights='imagenet', encoder_freeze=True, classes=4)
inputs = ks.layers.Input((350, 525, 1))
inputs_padded = ks.layers.ZeroPadding2D(((1, 1), (10, 9)))(inputs)
inputs_3_channel = ks.layers.Lambda(lambda x: ks.backend.repeat_elements(x, 3, 3))(inputs_padded)
outputs = backbone(inputs_3_channel)
outputs_cropped = ks.layers.Cropping2D(((1, 1), (10, 9)))(outputs)
est2 = ks.models.Model(inputs, outputs_cropped)
est2.load_weights(os.path.join('nn15', 'nn_est.h5'))

# check model auc score
preds = (est1.predict(calib_data) + est2.predict(calib_data)) / 2
print("model auc scores, calib set: ")
for i in range(4):
    print(metrics.roc_auc_score(calib_masks[:, :, :, i].flatten(), preds[:, :, :, i].flatten()))


def local_log_reg(var_arr, target_arr, window, step_size):
    est = linear_model.LogisticRegression(random_state=0, solver='lbfgs')
    ans = np.zeros(len(range(0, len(var_arr), step_size)))
    ans[:] = np.nan
    for i, center in enumerate(range(0, len(var_arr), step_size)):
        min_ind = max(0, center - window // 2)
        max_ind = min(len(var_arr) - 1, center + window // 2)
        x_curr = var_arr[min_ind:max_ind][:, np.newaxis]
        y_curr = target_arr[min_ind:max_ind]
        if len(np.unique(y_curr)) > 1:
            est.fit(x_curr, y_curr)
            ans[i] = est.predict_proba(np.array([var_arr[center]])[:, np.newaxis])[:, 1]
        else:
            ans[i] = np.unique(y_curr)[0]
    return np.interp(var_arr, var_arr[np.arange(0, len(var_arr), step_size)], ans)


# "local logistic regression" model to predict probability that class exists in image, given 95th percentile value
# of prediction for that class
true_arr = np.split(calib_masks, calib_masks.shape[0], 0)
pred_arr = np.split(preds, preds.shape[0], 0)
prob_nonzero = []
pc = 95
for i in range(4):
    pred_df = pd.DataFrame({'true': [np.mean(x[:, :, :, i]) for x in true_arr],
                            'pred': [np.percentile(x[:, :, :, i], pc) for x in pred_arr]})
    pred_df.sort_values('pred', inplace=True)
    pred_df['smoothed_nonzero_prob'] \
        = local_log_reg(pred_df['pred'].values, (pred_df['true'] > 0).astype(int).values, 200, 1)
    pred_df['prob_any_nonzero'] = (pred_df['true'] > 0).astype(int)
    prob_nonzero.append(pred_df)

# calculate expected dice at various thresholds, smooth values to get a simple model
exp_dice = []
for i in range(4):
    threshold_list = list(np.arange(0.05, 1, 0.05))
    high_pred_arr = []
    dice_arr_at_t = {t_num: [] for t_num in range(len(threshold_list))}
    for j in range(preds.shape[0]):
        if np.sum(calib_masks[j, :, :, i]) > 0:
            high_pred_arr.append(np.percentile(preds[j, :, :, i], pc))
            for t_num, t in enumerate(threshold_list):
                dice_arr_at_t[t_num]\
                    .append(dice_per_image_per_class(calib_masks[j, :, :, i], (preds[j, :, :, i] > t).astype(int)))
    exp_dice_df = pd.DataFrame({'max_pred': high_pred_arr})
    for t_num, t in enumerate(threshold_list):
        exp_dice_df['dice_at_{}'.format(t_num)] = dice_arr_at_t[t_num]
    exp_dice_df.sort_values('max_pred', inplace=True)
    for t_num, t in enumerate(threshold_list):
        exp_dice_df['smoothed_dice_at_{}'.format(t_num)] \
            = exp_dice_df['dice_at_{}'.format(t_num)].rolling(100, min_periods=1, center=True).mean()
    exp_dice.append(exp_dice_df)


def choose_prediction_for_class(class_preds, exp_dice_with_zero_pred, exp_dice_with_nonzero_pred, best_threshold):
    preds_processed = np.zeros(class_preds.shape[:3], np.int8)
    for i in range(class_preds.shape[0]):
        if exp_dice_with_zero_pred[i] < exp_dice_with_nonzero_pred[i]:
            preds_processed[i, :, :] = (class_preds[i, :, :] > best_threshold[i]).astype(np.int8)
    return preds_processed


def get_prediction_for_images(preds, prob_nonzero, exp_dice):
    preds_processed = np.zeros(preds.shape, np.int8)
    for i in range(4):
        class_preds = preds[:, :, :, i]
        class_exp_dice = exp_dice[i]
        max_pred = np.percentile(class_preds, pc, axis=(1, 2))
        dice_at_threshold = np.zeros((class_preds.shape[0], len(threshold_list)))
        for t_num in range(len(threshold_list)):
            dice_at_threshold[:, t_num] = np.interp(max_pred, class_exp_dice['max_pred'].values,
                                                    class_exp_dice['smoothed_dice_at_{}'.format(t_num)])
        exp_dice_at_best_threshold = np.max(dice_at_threshold, axis=1)
        best_threshold = np.array(threshold_list)[np.argmax(dice_at_threshold, axis=1)]
        class_prob_nonzero = prob_nonzero[i]
        predicted_prob_nonzero = np.interp(max_pred, class_prob_nonzero['pred'].values,
                                           class_prob_nonzero['smoothed_nonzero_prob'].values)
        exp_dice_with_zero_pred = 1 - predicted_prob_nonzero
        exp_dice_with_nonzero_pred = predicted_prob_nonzero * exp_dice_at_best_threshold
        preds_processed[:, :, :, i] = choose_prediction_for_class(class_preds, exp_dice_with_zero_pred,
                                                                  exp_dice_with_nonzero_pred, best_threshold)
    return preds_processed


@jit
def dice_for_image_array(true, pred):
    dice_arr = np.zeros(true.shape[0])
    for i in range(true.shape[0]):
        curr_image_dice = 0
        for j in range(4):
            curr_image_dice += dice_per_image_per_class(true[i, :, :, j], pred[i, :, :, j])
        dice_arr[i] = curr_image_dice / 4
    return np.mean(dice_arr)


# evaluate (test mode) or predict (submit mode)
if test_mode:
    preds_test = (est1.predict(test_data) + est2.predict(test_data)) / 2
    preds_processed = get_prediction_for_images(preds_test, prob_nonzero, exp_dice)
    print("validation dice: ", dice_for_image_array(test_masks, preds_processed))
    preds_processed = get_prediction_for_images(preds, prob_nonzero, exp_dice)
    print("calibration dice: ", dice_for_image_array(calib_masks, preds_processed))
else:
    @jit
    def rle_arr(class_arr):
        rle = []
        curr_start = -1
        j = 0
        for j in range(0, len(class_arr)):
            if class_arr[j] == 0:
                if curr_start > 0:
                    rle.append(curr_start + 1)
                    rle.append(j - curr_start)
                    curr_start = -1
            else:
                if curr_start == -1:
                    curr_start = j
        if (j > 0) and (class_arr[j] == 1):
            rle.append(curr_start + 1)
            rle.append(j - curr_start + 1)
        return rle


    def encode(x):
        ans = []
        for i in range(4):
            class_arr = x[:, :, i].flatten('F')
            ans.append(' '.join([str(x) for x in rle_arr(class_arr)]))
        return ans


    submit_names = pd.read_csv(os.path.join(output_dir_base, 'submit_data.csv'))['names']
    submit_data = np.load(os.path.join(output_dir_base, 'submit_data.npy'))
    est1.load_weights(os.path.join('nn14', 'nn_est_full.h5'))
    est2.load_weights(os.path.join('nn15', 'nn_est_full.h5'))
    preds = (est1.predict(submit_data) + est2.predict(submit_data)) / 2
    preds_processed = get_prediction_for_images(preds, prob_nonzero, exp_dice)
    df_arr = []
    for j in range(preds_processed.shape[0]):
        df = pd.DataFrame(columns=['Image_Label', 'EncodedPixels'])
        enc = encode(np.squeeze(preds_processed[j, :, :, :]))
        for i, name in enumerate(['Fish', 'Flower', 'Gravel', 'Sugar']):
            df.loc[len(df)] = [submit_names[j] + '_' + name, enc[i]]
        df_arr.append(df)
    df = pd.concat(df_arr, axis=0, sort=True)
    output_filename = 'Submission_{}.csv'.format(pd.datetime.now().strftime('%y%m%d_%H%M'))
    df[['Image_Label', 'EncodedPixels']].to_csv(output_filename, index=False, header=True)
