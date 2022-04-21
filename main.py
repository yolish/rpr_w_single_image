"""
Entry point training and testing our approach
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors import get_model
from os.path import join
from models.nerfmm.utils.training_utils import load_ckpt_to_net
from models.nerfmm.models.nerf_models import OfficialNerf
from models.nerfmm.models.intrinsics import LearnFocal
from models.nerfmm.nerf_for_rpr import run_nerf, get_nerf_args
from models.rpr.RelativePoseRegressor import RelativePoseRegressor
import os


def get_closest_pose(query_poses, db_poses, sample_radius=0):
    query_poses = query_poses.cpu().numpy()
    ref_poses = np.zeros((query_poses.shape[0], 7))
    for i, p in enumerate(query_poses):
        dist_x = np.linalg.norm(p[:3] - db_poses[:, :3], axis=1)
        dist_x = dist_x / np.max(dist_x)
        dist_q = np.linalg.norm(p[3:] - db_poses[:, 3:], axis=1)
        dist_q = dist_q / np.max(dist_q)
        if sample_radius == 0: # take closest pose (test mode)
            ref_poses[i, :] = db_poses[np.argmin(dist_x + dist_q)]
        else: # sample within a radius from second place to radius + 1 (training mode)
            sorted = np.argsort(dist_x + dist_q)
            np.random.randint(1, sample_radius+1)
            ref_poses[i, :] = db_poses[sorted[np.random.randint(1, sample_radius+1)]]
    return ref_poses


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("apr_model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("apr_backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("apr_checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("nerf_model_dir", help="path to directory with nerf model(s) per relevant scene")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file")
    arg_parser.add_argument("--rpr_checkpoint_path", help="path to a trained pose encoder")
    arg_parser.add_argument("--ref_poses_file", help="path to dataset with ref poses")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {}".format(args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.apr_model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Load the apr model
    apr = get_model(args.apr_model_name, args.apr_backbone_path, config).to(device)
    apr.load_state_dict(torch.load(args.apr_checkpoint_path, map_location=device_id))
    logging.info("Initializing from checkpoint: {}".format(args.apr_checkpoint_path))
    apr.eval()

    # Instantiate and load pretrained NERF models
    nerfmm = {}
    nerf_args = get_nerf_args()

    h = nerf_args.h
    w = nerf_args.w

    scene_nerfs = [dir for dir in os.listdir(args.nerf_model_dir)]
    for scene in scene_nerfs:
        ckpt_dir = join(args.nerf_model_dir, scene)
        pos_enc_in_dims = (2 * nerf_args.pos_enc_levels + int(nerf_args.pos_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
        if nerf_args.use_dir_enc:
            dir_enc_in_dims = (2 * nerf_args.dir_enc_levels + int(nerf_args.dir_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
        else:
            dir_enc_in_dims = 0

        model = OfficialNerf(pos_enc_in_dims, dir_enc_in_dims, nerf_args.hidden_dims)
        model = model.to(device=device)
        model = load_ckpt_to_net(join(ckpt_dir, 'latest_nerf.pth'), model, map_location=device)

        focal_net = LearnFocal(h, w, nerf_args.learn_focal, nerf_args.fx_only, order=nerf_args.focal_order)
        focal_net = focal_net.to(device=device)
        focal_net = load_ckpt_to_net(join(ckpt_dir, 'latest_focal.pth'), focal_net, map_location=device)
        nerfmm[scene] = [model, focal_net]

    # Instatiate the RPR (CONV2D + RPR)
    rpr = RelativePoseRegressor(config).to(device)
    if args.rpr_checkpoint_path:
        rpr.load_state_dict(torch.load(args.rpr_checkpoint_path, map_location=device_id))
        logging.info("Initializing encoder from checkpoint: {}".format(args.rpr_checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        rpr.train()

        # Set the losses
        pose_loss = CameraPoseLoss(config).to(device)

        # Set the optimizer and scheduler
        params = list(rpr.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        transform = utils.train_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, False)

        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")
        sample_radius = config.get("pose_sample_radius")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_scene = minibatch.get('scene')
                gt_scene_str = dataloader.dataset.scene_unique_names[gt_scene.cpu().numpy()]
                minibatch['scene'] = None
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                # Zero the gradients
                optim.zero_grad()

                # Get latents and initial pose from APR
                with torch.no_grad():
                    res = apr(minibatch)
                latent_x_init = res.get("latent_x")
                latent_q_init = res.get("latent_q")
                p_init = res.get('pose')
                latent_p_init = torch.cat((latent_x_init, latent_q_init), dim=1)

                # Sample a pose in near radius
                ref_p = get_closest_pose(p_init, dataset.poses, sample_radius=sample_radius)

                # Pass pose to NERF
                ref_rgb = []
                ref_depth = []

                with torch.no_grad():
                    for j in range(ref_p.shape[0]):
                        nerf_model = nerfmm[gt_scene_str[j]][0].eval().to(device)
                        focal_net = nerfmm[gt_scene_str[j]][1].eval().to(device)
                        rgb, depth = run_nerf(nerf_model, focal_net, ref_p[j], h, w, device, nerf_args, near=0.0, far=1.0)
                        ref_rgb.append(rgb)
                        ref_depth.append(depth)

                #Convert to Tensor
                ref_rgb = torch.stack(ref_rgb, dim=0).to(device).to(dtype=torch.float32).permute(0, 3, 1, 2)
                ref_depth = torch.stack(ref_depth, dim=0).to(device).to(dtype=torch.float32).unsqueeze(1)

                # Compute relative pose and absolute pose
                ref_p = torch.Tensor(ref_p).to(device).to(dtype=latent_p_init.dtype)
                est_p = rpr(ref_rgb, ref_depth, ref_p, latent_p_init)['pose']

                # Compute loss
                criterion = pose_loss(est_p, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_p.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(rpr.state_dict(), checkpoint_prefix + '_rpr_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(rpr.state_dict(), checkpoint_prefix + '_pose_encoder_final.pth'.format(epoch))

    else: # Test

        # Set to eval mode
        rpr.eval()

        # Get the train poses
        ref_poses = CameraPoseDataset(args.dataset_path, args.ref_poses_file, None).poses

        # Set the test dataset and data loader
        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):

                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_scene = minibatch.get('scene')
                minibatch['scene'] = None

                tic = time.time()
                # Get latents and initial pose from APR
                res = apr(minibatch)
                latent_x_init = res.get("latent_x")
                latent_q_init = res.get("latent_q")
                p_init = res.get('pose')
                scene_str = dataloader.dataset.scene_unique_names[0] # single scene at a time

                latent_p_init = torch.cat((latent_x_init, latent_q_init), dim=1)

                # Get closest pose
                ref_p = get_closest_pose(p_init, ref_poses, sample_radius=0)

                # Pass pose to NERF
                nerf_model = nerfmm[scene_str][0].eval().to(device)
                focal_net = nerfmm[scene_str][1].eval().to(device)

                ref_rgb, ref_depth = run_nerf(nerf_model, focal_net, ref_p[0], h, w, device, nerf_args, near=0.0,
                                      far=1.0)

                # Convert to Tensor
                ref_rgb = ref_rgb.to(device).to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                ref_depth = ref_depth.to(device).to(dtype=torch.float32).unsqueeze(0).unsqueeze(1)

                # Compute relative pose and absolute pose
                ref_p = torch.Tensor(ref_p).to(device).to(dtype=latent_p_init.dtype)
                est_p = rpr(ref_rgb, ref_depth, ref_p, latent_p_init)['pose']

                toc = time.time()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_p, gt_pose)

                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.rpr_checkpoint_path, args.labels_file))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))





