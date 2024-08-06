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

metric_dict = {}
# keys are the names of the directories in comp_dirs
# get average mse and ssim for each directory

for comp_dir in comp_dirs:
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
        # img = img / 255
        # pred = pred / 255

        mask = img > 0
        black = np.zeros_like(img)

        img = np.where(mask, img, black)
        pred = np.where(mask, pred, black)

        mse += mean_squared_error(img, pred)
        ssim += structural_similarity(img, pred)
        l1 += np.sum(np.abs(img - pred))
        # to fix the division by zero error
        # to fix the division by zero error
        psnr += peak_signal_noise_ratio(img, pred)

        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
        pred = torch.tensor(pred).unsqueeze(0).unsqueeze(0).float()
        
        # repeat the image 3 times to make it RGB
        img = torch.cat((img, img, img), dim=1)
        pred = torch.cat((pred, pred, pred), dim=1)

        # normalize to [-1, 1]
        img = img / 255 * 2 - 1
        pred = pred / 255 * 2 - 1

        lpips += lpips_vgg(img, pred).item()
        if normalized_by_mask:
            mse /= np.sum(mask)
            ssim /= np.sum(mask)
            l1 /= np.sum(mask)
            psnr /= np.sum(mask)
            lpips /= np.sum(mask)
        
    mse /= len(files)
    ssim /= len(files)
    l1 /= len(files)
    psnr /= len(files)
    lpips /= len(files)
    metric_dict[os.path.basename(comp_dir)] = {"mse": mse, "ssim": ssim, "l1": l1, "psnr": psnr, "lpips": lpips}
    
is_masked = "_mask_normalized" if normalized_by_mask else ""
with open(f"{data_dir}/qualitative/{epoch}{is_masked}_metrics.json", "w") as f:
    json.dump(metric_dict, f)