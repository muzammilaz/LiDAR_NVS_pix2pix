import os
from PIL import Image
import numpy as np
# import shutil
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import lpips
import torch
import json
from glob import glob


# change directory as required
# lidar4d_data / lidar4d_data_res_300 / lidar4d_data_res_600
data_dir = "/home/woody/i9vl/i9vl106h/data/lidar4d_data"
files = glob(f"{data_dir}/ground_truth/*.png")

lpips_vgg = lpips.LPIPS(net='alex')

normalized_by_mask = False

# epoch_list = [100, 200, 300, 400, 500, 600]
# for epoch in epoch_list:

# using fixed epoch
epoch = 400
comp_dirs = sorted(glob(f"{data_dir}/preds_spade*{epoch}")) 
comp_dirs = [d for d in comp_dirs if "masked" in d or "vanilla" in d]
comp_dir_kitti_baseline = sorted(glob(f"{data_dir}/preds_spade*single*700"))
comp_dirs += comp_dir_kitti_baseline

metric_dict = {}
# keys are the names of the directories in comp_dirs
# get average mse and ssim for each directory

for comp_dir in comp_dirs:
    imgs_batch = []
    preds_batch = []
    mse = 0
    ssim = 0
    l1 = 0
    psnr = 0
    lpips = 0

    # files = glob(os.path.join(comp_dir, "*.png"))
    for file in files:
        # read with PIL and convert to grayscale 
        img = np.array(Image.open(file).convert("L"))
        # skip if the image is all black
        if np.sum(img) == 0:
            continue

        pred = np.array(Image.open(os.path.join(comp_dir, os.path.basename(file))).convert("L"))
        # scale to [0, 1] 
        # img = np.array(img) / 255
        # pred = np.array(pred) / 255

        mask = img > 0
        black = np.zeros_like(img)

        img = np.where(mask, img, black)
        pred = np.where(mask, pred, black)

        imgs_batch.append(img)
        preds_batch.append(pred)


        img_flattenned_masked = img.flatten()[mask.flatten()].astype(np.float64)
        pred_flattenned_masked = pred.flatten()[mask.flatten()].astype(np.float64)
        
        # mse += mean_squared_error(img_flattenned_masked, pred_flattenned_masked)
        mse += np.mean((pred_flattenned_masked - img_flattenned_masked) ** 2)
        ssim += structural_similarity(img_flattenned_masked, pred_flattenned_masked)
        l1 += np.mean(np.abs(img_flattenned_masked - pred_flattenned_masked))
        # mae += np.median(np.abs(img_flattenned_masked - pred_flattenned_masked))

        psnr += peak_signal_noise_ratio(img_flattenned_masked, pred_flattenned_masked, data_range=255)

        if normalized_by_mask:
            mse /= np.sum(mask)
            ssim /= np.sum(mask)
            l1 /= np.sum(mask)
            psnr /= np.sum(mask)
            lpips /= np.sum(mask)
    
    imgs_batch = np.array(imgs_batch)
    preds_batch = np.array(preds_batch)

    # repeat 3 times for each channel
    imgs_batch = torch.tensor(imgs_batch).unsqueeze(1).repeat(1, 3, 1, 1).float()
    preds_batch = torch.tensor(preds_batch).unsqueeze(1).repeat(1, 3, 1, 1).float()
    # normalize to [-1, 1]
    imgs_batch = imgs_batch / 127.5 - 1
    preds_batch = preds_batch / 127.5 - 1
    
    lpips += lpips_vgg(imgs_batch, preds_batch).sum().item()

    mse /= len(files)
    ssim /= len(files)
    l1 /= len(files)
    psnr /= len(files)
    lpips /= len(files)

    # only keep the last 3 digits
    metric_dict[os.path.basename(comp_dir)] = {"mse": round(mse, 3), "ssim": round(ssim, 3), "l1": round(l1, 3), "psnr": round(psnr, 3), "lpips": round(lpips, 3)}

is_masked = "_mask_normalized" if normalized_by_mask else ""

with open(f"{data_dir}/qualitative/{epoch}{is_masked}_metrics_fixed.json", "w") as f:
    json.dump(metric_dict, f)