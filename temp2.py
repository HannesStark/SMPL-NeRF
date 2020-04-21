import pickle
import numpy as np


from camera import get_pose_matrix, get_circle_pose
from inference import inference

poses = []
for i in range(-20, 20):
    poses.append(get_circle_pose(i, 2.4))

inference('runs/Apr17_11-16-54_hannesgpu/small_angles.pkl', poses, batch_size=50)
