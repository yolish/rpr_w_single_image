import torch
from models.nerfmm.utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from models.nerfmm.train_nerf import model_render_image
import numpy as np
import imageio

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


def get_nerf_args():
    return NerfArgs()


def run_nerf(nerf_model, focal_net, c2w, h, w, device, args, near=0.0, far=1.0, plot_temp_img=True):
    nerf_model.eval()
    focal_net.eval()

    fxfy = focal_net(0)
    ray_dir_cam = comp_ray_dir_cam_fxfy(h, w, fxfy[0], fxfy[1]).to(device)
    t_vals = torch.linspace(near, far, args.num_sample, device=device)  # (N_sample,) sample position

    # Render image and depth
    render_result = model_render_image(c2w, ray_dir_cam, t_vals, near, far, h, w, fxfy,
                                       nerf_model, False, 0.0, args, rgb_act_fn=torch.sigmoid)
    rgb = render_result['rgb'] # (h, w, 3)
    depth = render_result['depth_map'] # (h, W)

    if plot_temp_img:
        img = (rgb.cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite('tmp.png', img)

    return rgb, depth

