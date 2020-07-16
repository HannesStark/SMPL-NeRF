# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Neural Mesh Renderer

# MIT License

# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# A PyTorch implementation of Neural 3D Mesh Renderer (https://github.com/hiroharu-kato/neural_renderer)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import numpy as np
import torch
import tqdm
import imageio

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

    ###########################
    # Load mesh
    ###########################
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
    
    betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                           -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]]).to(device)
    expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                                0.5643, -1.2158, 1.4149, 0.4050, 0.6516]]).to(device)
    goal_pose = torch.zeros(69).view(1, -1).to(device)
    goal_pose[0, 38] = np.deg2rad(30)
    goal_pose[0, 41] = np.deg2rad(30)

    smpl_model = DummySmplEstimatorModel(goal_pose, expression, betas)
    smpl_model = smpl_model.to(device)

    canonical_output = model(betas=betas, expression=expression,
                               return_verts=True, body_pose=None)
    #Normalize vertices
    output = model(betas=smpl_model.betas, expression=smpl_model.expression,
                       return_verts=True, body_pose=smpl_model.goal_poses)

    vertices_goal = output.vertices[0]
    print("vertices goal: ", vertices_goal.abs().max())

    vertices_abs_max = torch.abs(vertices_goal).max().detach()
    vertices_min = vertices_goal.min(0)[0][None, :].detach()
    vertices_max = vertices_goal.max(0)[0][None, :].detach()
    
    faces = torch.tensor(model.faces * 1.0).to(device)
    print(faces.shape)
    print(canonical_output.vertices[0].shape)
    mesh = TriangleMesh.from_tensors(canonical_output.vertices[0], faces)
    # mesh = TriangleMesh.from_obj(args.mesh)
    # Normalize into unit cube, and expand such that batch size = 1
    print("vertices can: ", mesh.vertices.abs().max())
    vertices = mesh.vertices.unsqueeze(0)
    #vertices = pre_normalize_vertices(mesh.vertices, vertices_min, vertices_max, 
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
    print(true_image.requires_grad)
    print(list(smpl_model.parameters()))
    optim=torch.optim.Adam(list(smpl_model.parameters()), lr=1e-2)
    
    imageio.imwrite("results/img_true.png", (255 * true_image.cpu().numpy()).astype(np.uint8))
    for i in range(1000):
        optim.zero_grad()
        output = model(betas=smpl_model.betas, expression=smpl_model.expression,
                       return_verts=True, body_pose=smpl_model.goal_poses)

        vertices_goal = output.vertices[0]

        mesh = TriangleMesh.from_tensors(vertices_goal, faces)
        #mesh = TriangleMesh.from_obj(args.mesh)

        # Normalize into unit cube, and expand such that batch size = 1
        vertices = vertices_goal.unsqueeze(0)
        #vertices = pre_normalize_vertices(mesh.vertices, vertices_min, vertices_max, 
        #                              vertices_abs_max).unsqueeze(0)
        #print("vertices goal: ", vertices.max())
        images, _, _ = renderer(vertices, faces, textures)
        image = images[0]
        loss = (image.permute(1, 2, 0)-true_image).abs().mean()
        imageio.imwrite("results/img_out{:03d}.png".format(i), (255 * image.permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8))
        
        loss.backward()
        
        print("Loss: ", loss.item())
        optim.step()
        with torch.no_grad():
            grad = smpl_model.goal_poses.grad
            #print('grads of poses ', grad)
            #print('armangle grad ', grad[0, 38],' armagnle2 grad ', grad[0, 41])
            #print('armangle', smpl_model.goal_poses[0, 38], ' armagnle2 ', smpl_model.goal_poses[0, 41])
            #smpl_model.goal_poses -= 1e-4 * grad
            #smpl_model.goal_poses.grad.zero_()
            smpl_model.goal_poses.grad.detach_()
            loss.detach_()


if __name__ == '__main__':
    main()
