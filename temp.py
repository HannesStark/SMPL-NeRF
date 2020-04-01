import os.path as osp
import argparse
import pickle
from io import BytesIO

import numpy as np
import torch
import cv2
from PIL import Image
import smplx
import pyrender
import trimesh

model = smplx.create("models/model.pkl", model_type='smpl')
betas = torch.randn([1, 10], dtype=torch.float32)
expression = torch.randn([1, 10], dtype=torch.float32)
output = model(betas=betas, expression=expression,
               return_verts=True)
vertices = output.vertices.detach().cpu().numpy().squeeze()


with open('textures/texture.jpg', 'rb') as file:
    texture = Image.open(BytesIO(file.read()))

uv = np.load('textures/smpl_uv_map.npy')
print(uv)
smpl_mesh = trimesh.Trimesh(vertices, model.faces, visual=trimesh.visual.TextureVisuals(uv=uv, image=texture), process=False)

pymesh = pyrender.Mesh.from_trimesh(smpl_mesh)

scene = pyrender.Scene()
scene.add(pymesh)

light =  pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=30.0)
scene.add(light)

pyrender.Viewer(scene)
