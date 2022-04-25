# Training and Testing Pose Auto Encoders

## NERF
### Train
```
models/nerfmm/train_nerf.py
--scene_name /mnt/data/Cambridge/
--img_names_file ../../datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv
--results_name shop_facade

```
###Test
```
models/nerfmm/test_nerf.py 
--ckpt_dir <checkpoint_dir>
--resize_ratio 5 --num_rows_eval_img 1 --N_img_per_circle 10 --N_circle_traj 1```
```
or to see it can reconstruct train images 
```
test_nerf.py --ckpt_dir <checkpoint_dir> --resize_ratio 30 --num_rows_eval_img 1 --test_on_train
```
## RPR with single image
### Train (Cambridge: Shop Facade) with RPR from APR latent
```
train
ems-transposenet
models/backbones/efficient-net-b0.pth
models/pretrained_aprs/ems_transposenet_cambridge_pretrained_finetuned.pth
models/pretrained_nerfs/
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv
CambridgeLandmarks_config.json
```

### Train (Cambridge) with RPR from image (APR guess)
In config, set 'apr_for_pose_guess' to true and 'rpr_backbone_type' to resnet50
```
train
ems-transposenet
models/backbones/efficient-net-b0.pth
models/pretrained_aprs/ems_transposenet_cambridge_pretrained_finetuned.pth
models/pretrained_nerfs/
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv
CambridgeLandmarks_config.json
--rpr_backbone_path
models/backbones/resnet50_pretrained.pth
```



### Test - Example on Cambridge: Shop Facade
```
main.py
test
ems-transposenet
models/backbones/efficient-net-b0.pth
models/pretrained_aprs/ems_transposenet_cambridge_pretrained_finetuned.pth
models/pretrained_nerfs/
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_test.csv
CambridgeLandmarks_config.json
--rpr_checkpoint_path out/run_22_04_22_01_25_rpr_checkpoint-590.pth
--ref_poses_file datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv
```

