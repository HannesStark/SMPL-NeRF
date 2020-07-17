import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import argparse
import os
import numpy as np
import torch
import tqdm
import imageio
from torch.autograd import Variable

from kaolin.graphics import NeuralMeshRenderer as Renderer
from kaolin.graphics.nmr.util import get_points_from_angles
from kaolin.rep import TriangleMesh

from models.dummy_smpl_estimator_model import DummySmplEstimatorModel
from util_nmr import normalize_vertices, pre_normalize_vertices

import smplx
from PIL import Image
from io import BytesIO
import torch
import smplx

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def parse_arguments():
    parser = argparse.ArgumentParser(description='NMR Example 1: Render mesh')

    parser.add_argument('--mesh', type=str, default=os.path.join(ROOT_DIR, 'rocket.obj'),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--output_path', type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help='Path to the output directory')
    parser.add_argument('--camera_distance', type=float, default=2.4,
                        help='Distance from camera to object center')
    parser.add_argument('--elevation', type=float, default=0,
                        help='Camera elevation')
    parser.add_argument('--texture_size', type=int, default=2,
                        help='Dimension of texture')

    return parser.parse_args()


def main():
    args = parse_arguments()

    experiment_name = 'change_arms_2_betas'
    torch.autograd.set_detect_anomaly(True)
    smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
    uv_map_file_name = "textures/smpl_uv_map.npy"
    uv = np.load(uv_map_file_name)
    texture_file_name = "textures/texture.jpg"
    with open(texture_file_name, 'rb') as file:
        texture = Image.open(BytesIO(file.read()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = smplx.create(smpl_file_name, model_type='smpl')
    model = model.to(device)
    specific_angles_only = True
    perturb_betas = True
    betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                           -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]]).to(device)
    if perturb_betas:
        perturbed_betas = Variable(torch.tensor([[0.3596, -1.0232, 1.7584, -2.0465, -0.3387,
                                                  0.8562, 0.8869, -0.5013, 0.5338, 0.0210]]).to(device),
                                   requires_grad=True)
    else:
        perturb_betas = betas
    expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                                0.5643, -1.2158, 1.4149, 0.4050, 0.6516]]).to(device)
    perturbed_pose = torch.zeros(69).view(1, -1).to(device)
    perturbed_pose[0, 38] = -np.deg2rad(60)
    perturbed_pose[0, 41] = np.deg2rad(60)
    perturbed_pose = Variable(perturbed_pose, requires_grad=True)
    canonical_pose0 = torch.zeros(2).view(1, -1).to(device)
    canonical_pose1 = torch.zeros(35).view(1, -1).to(device)
    canonical_pose2 = torch.zeros(2).view(1, -1).to(device)
    canonical_pose3 = torch.zeros(27).view(1, -1).to(device)
    arm_angle_l = Variable(torch.tensor([-np.deg2rad(60)]).float().view(1, -1).to(device), requires_grad=True)
    arm_angle_r = Variable(torch.tensor([np.deg2rad(60)]).float().view(1, -1).to(device), requires_grad=True)
    leg_angle_l = Variable(torch.tensor([np.deg2rad(60)]).float().view(1, -1).to(device), requires_grad=True)

    canonical_output = model(betas=betas, expression=expression,
                             return_verts=True, body_pose=None)

    # Normalize vertices
    # output = model(betas=betas, expression=expression,
    #               return_verts=True, body_pose=perturbed_pose)

    # vertices_goal = output.vertices[0]
    # vertices_abs_max = torch.abs(vertices_goal).max().detach()
    # vertices_min = vertices_goal.min(0)[0][None, :].detach()
    # vertices_max = vertices_goal.max(0)[0][None, :].detach()

    faces = torch.tensor(model.faces * 1.0).to(device)

    mesh = TriangleMesh.from_tensors(canonical_output.vertices[0], faces)
    vertices = mesh.vertices.unsqueeze(0)
    # vertices = pre_normalize_vertices(mesh.vertices, vertices_min, vertices_max,
    #                                  vertices_abs_max).unsqueeze(0)

    faces = mesh.faces.unsqueeze(0)

    textures = torch.ones(
        1, faces.shape[1], args.texture_size, args.texture_size, args.texture_size,
        3, dtype=torch.float32,
        device='cuda'
    )
    renderer = Renderer(camera_mode='look_at')
    azimuth = 180
    renderer.eye = get_points_from_angles(
        args.camera_distance, args.elevation, azimuth)
    images, _, _ = renderer(vertices, faces, textures)
    true_image = images[0].permute(1, 2, 0)
    true_image = true_image.detach()

    if specific_angles_only:
        optim = torch.optim.Adam([arm_angle_l, arm_angle_r, leg_angle_l], lr=1e-2)
    else:
        optim = torch.optim.Adam([perturbed_pose], lr=1e-2)
    results = []
    arm_parameters_l = []
    arm_parameters_r = []
    losses = []
    for i in range(200):
        optim.zero_grad()
        if specific_angles_only:
            perturbed_pose = torch.cat(
                [canonical_pose0, leg_angle_l, canonical_pose1, arm_angle_l, canonical_pose2, arm_angle_r,
                 canonical_pose3],
                dim=-1)
        output = model(betas=perturbed_betas, expression=expression,
                       return_verts=True, body_pose=perturbed_pose)

        vertices_goal = output.vertices[0]

        mesh = TriangleMesh.from_tensors(vertices_goal, faces)

        vertices = vertices_goal.unsqueeze(0)
        # vertices = pre_normalize_vertices(mesh.vertices, vertices_min, vertices_max,
        #                              vertices_abs_max).unsqueeze(0)

        images, _, _ = renderer(vertices, faces, textures)
        image = images[0]
        loss = (image.permute(1, 2, 0) - true_image).abs().mean()
        loss.backward()
        optim.step()

        results.append((255 * image.permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8))
        if specific_angles_only:
            arm_parameters_l.append(arm_angle_l.item())
            arm_parameters_r.append(arm_angle_r.item())
        losses.append(loss.item())
        print("Loss: ", loss.item())
    imageio.mimsave("results/" + experiment_name + "_gif.gif", results, fps=30)

    plt.plot(losses)
    plt.title("applied loss")
    plt.savefig("results/" + experiment_name + "_loss.png")
    plt.clf()
    plt.plot(arm_parameters_r)
    plt.title("right arm angle")
    plt.savefig("results/" + experiment_name + "_right.png")
    plt.clf()
    plt.plot(arm_parameters_l)
    plt.title("left arm angle")
    plt.savefig("results/" + experiment_name + "_left.png")


if __name__ == '__main__':
    main()
