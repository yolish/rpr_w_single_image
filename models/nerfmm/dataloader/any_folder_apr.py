# Code copied with small modifications from: https://github.com/ActiveVisionLab/nerfmm
import os

import torch
import numpy as np
from tqdm import tqdm
import imageio
import pandas as pd
import torch.nn.functional as F

def resize_imgs(imgs, new_h, new_w):
    """
    :param imgs:    (N, H, W, 3)            torch.float32 RGB
    :param new_h:   int/torch int
    :param new_w:   int/torch int
    :return:        (N, new_H, new_W, 3)    torch.float32 RGB
    """
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear')  # (N, 3, new_H, new_W)
    imgs = imgs.permute(0, 2, 3, 1)  # (N, new_H, new_W, 3)

    return imgs  # (N, new_H, new_W, 3) torch.float32 RGB


def load_imgs(data_dir, img_names_file, num_img_to_load, start, end, skip, load_sorted, load_img):
    # read file
    df = pd.read_csv(img_names_file)
    img_names = [os.path.join(data_dir, path) for path in df['img_path'].values]
    scene = df['scene'].values[0]

    # down sample frames in temporal domain
    if end == -1:
        img_names = img_names[start::skip]
    else:
        img_names = img_names[start:end:skip]

    if not load_sorted:
        np.random.shuffle(img_names)

    # load images after down sampled
    if num_img_to_load > len(img_names):
        print('Asked for {0:6d} images but only {1:6d} available. Exit.'.format(num_img_to_load, len(img_names)))
        exit()
    elif num_img_to_load == -1:
        print('Loading all available {0:6d} images'.format(len(img_names)))
    else:
        print('Loading {0:6d} images out of {1:6d} images.'.format(num_img_to_load, len(img_names)))
        img_names = img_names[:num_img_to_load]

    img_paths = img_names
    N_imgs = len(img_paths)

    img_list = []
    if load_img:
        for p in tqdm(img_paths):
            img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
            img_list.append(img)
        img_list = np.stack(img_list)  # (N, H, W, 3)
        img_list = torch.from_numpy(img_list).float() / 255  # (N, H, W, 3) torch.float32
        H, W = img_list.shape[1], img_list.shape[2]
    else:
        tmp_img = imageio.imread(img_paths[0])  # load one image to get H, W
        H, W = tmp_img.shape[0], tmp_img.shape[1]

    results = {
        'imgs': img_list,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'H': H,
        'W': W,
        'scene': scene
    }

    return results


class DataLoaderAnyFolder:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self, base_dir, img_names_file, res_ratio, num_img_to_load, start, end, skip, load_sorted, load_img=True):
        """
        :param base_dir:
        :param res_ratio:       int [1, 2, 4] etc to resize images to a lower resolution.
        :param start/end/skip:  control frame loading in temporal domain.
        :param load_sorted:     True/False.
        :param load_img:        True/False. If set to false: only count number of images, get H and W,
                                but do not load imgs. Useful when vis poses or debug etc.
        """
        self.base_dir = base_dir
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.start = start
        self.end = end
        self.skip = skip
        self.load_sorted = load_sorted
        self.load_img = load_img


        image_data = load_imgs(base_dir, img_names_file, self.num_img_to_load, self.start, self.end, self.skip,
                                self.load_sorted, self.load_img)
        self.imgs = image_data['imgs']  # (N, H, W, 3) torch.float32
        self.img_names = image_data['img_names']  # (N, )
        self.N_imgs = image_data['N_imgs']
        self.ori_H = image_data['H']
        self.ori_W = image_data['W']
        self.scene = image_data['scene']

        # always use ndc
        self.near = 0.0
        self.far = 1.0

        if self.res_ratio > 1:
            self.H = self.ori_H // self.res_ratio
            self.W = self.ori_W // self.res_ratio
        else:
            self.H = self.ori_H
            self.W = self.ori_W

        if self.load_img:
            self.imgs = resize_imgs(self.imgs, self.H, self.W)  # (N, H, W, 3) torch.float32

