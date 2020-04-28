import pickle
import numpy as np


from camera import get_pose_matrix, get_circle_pose
from inference import inference

poses = []
for i in range(-20, 20):
    poses.append(get_circle_pose(i, 2.4))

inference('runs/Apr18_07-59-51_hannesgpu/36_512Images.pkl', poses, batch_size=1000)
