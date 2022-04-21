# Training and Testing Pose Auto Encoders

## NERF
### Train
```
```
###Test
```
models/nerfmm/test_nerf.py
--ckpt_dir
nerf_results/shop_facade/
--resize_ratio
30
--num_rows_eval_img
1
--N_img_per_circle
10
--N_circle_traj
1
```

## RPR with single image
### Train (Cambridge)
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
--rpr_checkpoint_path out/run_21_04_22_15_41_rpr_checkpoint-10.pth
--ref_poses_file datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv
```

