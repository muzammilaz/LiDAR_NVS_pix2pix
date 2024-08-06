import sys
import numpy as np
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

from models.pix2pix_model import Pix2PixModel
from util import util
import os
from glob import glob

# change dataroot as from lidar4d_data / lidar4d_data_res_300 / lidar4d_data_res_600 and select checkpoints dir appropriately
args = "--name spade_res_300_vanilla --dataset_mode pix2pix --dataroot /home/woody/i9vl/i9vl106h/data/lidar4d_data --no_instance --label_nc 0 --preprocess_mode fixed_wh --load_size 1000 --crop 896 --aspect_ratio 4 --display_freq 5000 --niter 1000 --batchSize 4 --use_masked_training --checkpoints_dir /home/woody/i9vl/i9vl106h/logs --output_nc 1 --continue_train"
sys.argv = [sys.argv[0], *args.split()]

opt = TrainOptions().parse()
opt.isTrain = False

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# change pattern to get the concerned training runs
names = [os.path.basename(f) for f in glob(f"{opt.checkpoints_dir}/spade_test_6_single_*") if os.path.exists(os.path.join(f, "iter.txt")) and int(open(os.path.join(f, "iter.txt"), "r").readline().strip()) > 150]
# names = [name for name in names if "6_lidar_data" not in name]
names
for epoch in [700]:
    opt.which_epoch = epoch
    for name in names:
        opt.name = name
        # if "lidar" not in opt.name:
        #     opt.which_epoch = int(np.ceil((opt.which_epoch // 2)//10)*10)
        #     # min of opt.which_epoch or iter from file
        #     iter_file = int(open(f"{opt.checkpoints_dir}/{opt.name}/iter.txt", "r").readline().strip())
            # opt.which_epoch = np.where(iter_file < opt.which_epoch, "latest", opt.which_epoch)

        pix2pixmodel = Pix2PixModel(opt)
        pix2pixmodel.eval()

        
        # create trainer for our model
        trainer = Pix2PixTrainer(opt)

        for i, data_i in enumerate(dataloader):
            out = pix2pixmodel(data_i, mode='inference')
            paths_out = [path.replace('test_B', f'preds_{opt.name}_{opt.which_epoch}') for path in data_i['path']]

            for path, img in zip(paths_out, out):
                image_numpy = util.tensor2im(img)
                util.save_image(image_numpy, path, create_dir=True)
