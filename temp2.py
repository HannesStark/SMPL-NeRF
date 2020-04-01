import pyrender
import trimesh

smpl_mesh = trimesh.load_mesh('model.obj')
print(smpl_mesh.visual)