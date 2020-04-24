import smplx
import torch

import torch
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector

model = smplx.create('SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl', model_type='smpl')
# set betas and expression to fixed values
betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                       -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]])
expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                            0.5643, -1.2158, 1.4149, 0.4050, 0.6516]])
body_pose = torch.zeros(69).view(1, -1)

output = model(betas=betas, expression=expression,
               return_verts=True, body_pose=body_pose)
vertices = output.vertices.detach().cpu().numpy().squeeze()

smpl_mesh = trimesh.Trimesh(vertices, model.faces)

intersector = RayMeshIntersector(smpl_mesh)

print(smpl_mesh)
