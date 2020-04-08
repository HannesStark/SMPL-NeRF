import pickle
import h5py
import numpy as np

from camera import get_pose_matrix

poses = [ ]
poses.append(get_pose_matrix())
poses.append(get_pose_matrix(1))
image_transform_map = {'r_0.png': np.array([[
                    -0.9999021887779236,
                    0.004192245192825794,
                    -0.013345719315111637,
                    -0.05379832163453102
                ],
                [
                    -0.013988681137561798,
                    -0.2996590733528137,
                    0.95394366979599,
                    3.845470428466797
                ],
                [
                    -4.656612873077393e-10,
                    0.9540371894836426,
                    0.29968830943107605,
                    1.2080823183059692
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]])}
dict = {'camera_angle_x': 0.6911112070083618, 'image_transform_map': image_transform_map}

with open('testposes.pkl', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
