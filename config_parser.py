import configargparse


def config_parser():
    """
    Configuration parser for training.

    """
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default="configs/config.txt", help='config file path')
    parser.add_argument("--experiment_name", type=str, default='default', help='experiment name')
    parser.add_argument('--model_type', default="nerf", type=str,
                        help='choose model type for model [smpl_nerf, nerf, append_to_nerf, smpl]')
    parser.add_argument("--dataset_dir", type=str, default='data', help='directory with specific dataset structure')
    parser.add_argument("--number_validation_images", type=int, default=1,
                        help='number of images to take from the validation images directory and use to render validation images')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
    parser.add_argument("--skips", type=int, default=[4], help='layers with concateneted positional input',
                        action="append")
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, help='channels per layer in fine network')
    parser.add_argument("--skips_fine", type=int, default=[4],
                        help='layers with concateneted positional input of fine net', action="append")
    parser.add_argument("--run_fine", type=int, default=1, help='If 1 use fine network else only coarse')

    parser.add_argument("--netdepth_warp", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_warp", type=int, default=256, help='channels per layer in fine network')
    parser.add_argument("--gmm_std", type=float, default=0.07,
                        help='std of gaussian mixture model that is used for the loss of the densities')
    parser.add_argument("--restrict_gmm_loss", default=0, type=int,
                        help='restricts the loss of the densities with the gmm to only be optimized in the warp field')
    parser.add_argument("--use_gmm_loss", default=0, type=int,
                        help='additional gaussian mixture loss')

    parser.add_argument("--batchsize", type=int, default=2048,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--batchsize_val", type=int, default=512,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--weight_decay", type=int, default=0, help='adam weight decay')
    parser.add_argument("--log_iterations", type=int, default=10,
                        help='number of iterations to pass to run extra validation and logging')
    parser.add_argument("--mesh_epochs", type=int, default=[0, 0.5, 1],
                        help='layers with concateneted positional input',
                        action="append")
    parser.add_argument("--early_validation", type=int, default=0,
                        help='run extra validation loop every log_iterations')
    parser.add_argument("--num_epochs", type=int, default=100, help='number of epochs to run')
    parser.add_argument("--near", type=int, default=1, help='near ray bound for coarse sampling')
    parser.add_argument("--far", type=int, default=4, help='far ray bound for coarse sampling')
    parser.add_argument("--number_coarse_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--number_fine_samples", type=int, default=128, help='number of fine samples per ray')
    parser.add_argument("--human_pose_encoding", type=int, default=0,
                        help='whether or not to encode the human pose')
    parser.add_argument("--use_identity_positional", type=int, default=0,
                        help='add identity function to positional encoding functions')
    parser.add_argument("--use_identity_directional", type=int, default=0,
                        help='add identity function to directional encoding functions')
    parser.add_argument("--use_identity_pose", type=int, default=0,
                        help='add identity function to directional encoding functions')
    parser.add_argument("--number_frequencies_pose", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--number_frequencies_postitional", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--number_frequencies_directional", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--sigma_noise_std", type=float, default=1,
                        help='std dev of noise added to regularize sigma_a output, 1e0 r ecommended')
    parser.add_argument("--white_background", default=0, type=int,
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    return parser
