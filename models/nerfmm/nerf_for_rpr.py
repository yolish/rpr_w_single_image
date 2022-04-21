import argparse
import torch
from models.nerfmm.utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from models.nerfmm.train_nerf import model_render_image
import transforms3d as t3d
import numpy as np

class NerfArgs():
    def __init__(self):
        super(NerfArgs, self).__init__()
        self.hidden_dims = 128
        self.num_sample = 128
        self.pos_enc_levels = 10
        self.pos_enc_inc_in = True
        self.use_dir_enc = True
        self.dir_enc_levels = 4
        self.dir_enc_inc_in = True
        self.learn_focal = False
        self.focal_order = 2
        self.fx_only = False
        self.h = 32
        self.w = 32


def get_nerf_args():
    return NerfArgs()


def run_nerf(nerf_model, focal_net, p, h, w, device, args, near=0.0, far=1.0):

    nerf_model.eval()
    focal_net.eval()

    fxfy = focal_net(0)
    ray_dir_cam = comp_ray_dir_cam_fxfy(h, w, fxfy[0], fxfy[1]).to(device)
    t_vals = torch.linspace(near, far, args.num_sample, device=device)  # (N_sample,) sample position

    # convert pose to matrix representation and then to c2w - TODO verify
    c2w = np.zeros((4, 4)).astype(np.float)
    q = p[3:]
    c2w[:3, :3] = t3d.quaternions.quat2mat(q/np.linalg.norm(q))
    c2w[3, :3] = p[:3]
    c2w[3,3] = 1
    c2w = torch.Tensor(c2w).to(device)

    # Render image and depth
    render_result = model_render_image(c2w, ray_dir_cam, t_vals, near, far, h, w, fxfy,
                                       nerf_model, False, 0.0, args, rgb_act_fn=torch.sigmoid)
    rgb = render_result['rgb'] # (h, w, 3)
    depth = render_result['depth_map'] # (h, W)
    return rgb, depth

