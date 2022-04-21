## Relative Pose Regression with a Single Image
Official PyTorch implementation of relative pose regrssion with a single image
Our method is illustarted below:

![method](.imgs/approach.jpg)

### Pre-training absolute pose regressors and NERF models 
We implement our approach with [MS-Transformer](https://github.com/yolish/multi-scene-pose-transformer) as our pre-trained APR and [nerf-mm](https://github.com/ActiveVisionLab/nerfmm) as our pre-trained NERF model (which can be trained without extrnsitics and intrinsics)
1. Training and testing of MS-Transformer (as well as other APRs) is described in the MS-Transformer repository 
2. Training and testing of nerf-mm is described in the [nerf-mm repository](https://github.com/ActiveVisionLab/nerfmm). 

We have also included dedicated files for easy training and testing of NERF models on the CambridgeLandmarks / 7Scenes datasets, which are copied and adapted from the original repository.

Pre-trained models are available below. 

### Datasets and Prerequisites
We use the following datasets:
 1. [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) 
 2. [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) 
 
Software dependencies:
1. Python3 (tested with Python 3.7.7)
1. PyTorch deep learning framework (tested with version 1.0.0)
1. Use torch==1.4.0, torchvision==0.5.0

Note: All experiments reported in our paper were performed with an 8GB 1080 NVIDIA GeForce GTX GPU
For a quick set up you can run: pip install -r requirments.txt 


### Usage
#### Training and testing our approach
The entry point for training and testing APRs is the ```main.py``` script in the root directory

TBA 

See ```example_cmd.txt```

### Pre-trained models
You can download pretrained models in order to easily reproduce our results 
| Model (Linked) | Description | 
--- | ---
| APR models ||
[MS-Transformer](https://drive.google.com/file/d/1ZEIKQSbZmkSnJwETjACvMbs5OeCn7f3q/view?usp=sharing) | Multi-scene APR, CambridgeLandmarks dataset|
[MS-Transformer](https://drive.google.com/file/d/1Ryn5oQ0zRV_3KVORzMAk99cP0fY2ff85/view?usp=sharing) | Multi-scene APR, 7Scenes dataset|
| NERF models||
TBA | TBA 
| Conv and RPR models| |
TBA | TBA




 
  
  
  
  
