import smplx
import torch
import numpy as np
from vedo import Mesh, show

betas = torch.randn([2, 10], dtype=torch.float32)
expression = torch.randn([2, 10], dtype=torch.float32)

goal_pose = torch.zeros(69).view(1, -1)
goal_pose[0, 38] = np.deg2rad(45)
goal_pose[0, 41] = np.deg2rad(30)

goal_pose = goal_pose.expand((2, -1))

print(betas.shape)
print(expression.shape)
print(goal_pose.shape)
smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
model = smplx.create(smpl_file_name, model_type='smpl')
print(model.batchsize)
print(model.global_orient.shape)
output = model(betas=betas, expression=expression, return_verts=True)
vertices = output.vertices
faces = model.faces
print(faces.shape)
print(vertices.shape)
smpl_mesh = Mesh([vertices.detach().cpu().numpy().squeeze(), faces], alpha=0.3)
show(smpl_mesh, at=0)
