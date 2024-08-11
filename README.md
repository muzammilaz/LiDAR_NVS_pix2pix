# Synthesizing LiDAR from RGB Image using generated dense LiDAR data from LiDAR Novel View Synthesis  

In this project, we trained Image translation model [SPADE](https://github.com/NVlabs/SPADE/) to generate LiDAR intensities in camera perspective conditioned on the RGB image from the camera. We utilize KITTI and KITTI-360 datasets for training the models.

We utilize (LiDAR4D)[https://github.com/ispc-lab/LiDAR4D], a recent approach to LiDAR Novel View Synthesis, to generate dense LiDAR point cloud and use the projected intensities to train Image-to-Image translation models.

## Reults
| Experiment Setting                     | 66 Vertical Resolution | 300 Vertical Resolution                                                                   | 600 Vertical Resolution                                                                   |
|----------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| RGB Image                              | ![RGB Image 66](SPADE/qualitative_results/qualitative_66/rgb_0000_0000010769.png)        | ![RGB Image 300](SPADE/qualitative_results/qualitative_300/rgb_0000_0000010769.png)       | ![RGB Image 600](SPADE/qualitative_results/qualitative_600/rgb_0000_0000010769.png)       |
| Ground Truth                           | ![Ground Truth 66](SPADE/qualitative_results/qualitative_66/ground_truth_0000_0000010769.png) | ![Ground Truth 300](SPADE/qualitative_results/qualitative_300/ground_truth_0000_0000010769.png) | ![Ground Truth 600](SPADE/qualitative_results/qualitative_600/ground_truth_0000_0000010769.png) |
| KITTI Pix2Pix (-1 masked)              | ![KITTI Pix2Pix 66](SPADE/qualitative_results/qualitative_66/preds_spade_test_6_single_channel_0000_0000010769.png) | ![KITTI Pix2Pix 300](SPADE/qualitative_results/qualitative_66/preds_spade_test_6_single_channel_0000_0000010769.png) | ![KITTI Pix2Pix 600](SPADE/qualitative_results/qualitative_66/preds_spade_test_6_single_channel_0000_0000010769.png) |
| LiDAR4D (Projected points masked)      | ![LiDAR4D (Projected points masked) 66](SPADE/qualitative_results/qualitative_66/preds_spade_test_7_lidar_data_masked_0000_0000010769.png) | ![LiDAR4D (Projected points masked) 300](SPADE/qualitative_results/qualitative_300/preds_spade_res_300_masked_0000_0000010769.png) | ![LiDAR4D (Projected points masked) 600](SPADE/qualitative_results/qualitative_600/preds_spade_res_600_masked_0000_0000010769.png) |
| LiDAR4D (-1 masked)                    | ![LiDAR4D (-1 masked) 66](SPADE/qualitative_results/qualitative_66/preds_spade_test_8_lidar_data_masked_no_data_0000_0000010769.png) | ![LiDAR4D (-1 masked) 300](SPADE/qualitative_results/qualitative_300/preds_spade_res_300_masked_no_data_0000_0000010769.png) | ![LiDAR4D (-1 masked) 600](SPADE/qualitative_results/qualitative_600/preds_spade_res_600_masked_no_data_0000_0000010769.png) |
| LiDAR4D (Unmasked)                     | ![LiDAR4D (Unmasked) 66](SPADE/qualitative_results/qualitative_66/preds_spade_test_8_lidar_data_vanilla_0000_0000010769.png) | ![LiDAR4D (Unmasked) 300](SPADE/qualitative_results/qualitative_300/preds_spade_res_300_vanilla_0000_0000010769.png) | ![LiDAR4D (Unmasked) 600](SPADE/qualitative_results/qualitative_600/preds_spade_res_600_vanilla_0000_0000010769.png) |



#### Data files and checkpoints:

https://drive.google.com/drive/folders/1rG8l0ZZa0umXQzeiCLBDX6x1AVLgRu10?usp=drive_link
