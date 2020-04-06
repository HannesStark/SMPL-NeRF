import pickle
import h5py

from camera import get_pose_matrix

poses = [ ]
poses.append(get_pose_matrix())
poses.append(get_pose_matrix(1))
image_transform_map = {'01.png': get_pose_matrix(), '02.png': get_pose_matrix(1,2)}
dict = {'camera_angle_x': 0.44626346, 'image_transform_map': image_transform_map}

with open('testposes.pkl', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
