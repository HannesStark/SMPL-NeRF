import pickle

import smplx
import torch
import numpy as np
from vedo import Mesh, show



b = np.load('textures/smpl_data.npz')

favorite_color = pickle.load( open( "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl", "rb" ) )
print(favorite_color)