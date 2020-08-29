# -*- coding: utf-8 -*-
import numpy as np
from camera import get_pose_matrix, get_circle_pose, get_sphere_pose
import json
import os
import numpy as np
import operator
import json

def nearest_neighbor(val_transforms):

    # take transforms from RaysFromCameraDataset and calculate the nearest transform out of the transform pickle for training images
    # we have a dict with image names and corresponding transformations
    # calculate nearest neighbor from input transform vs list of transformation matrices: frobenius norm
    # copy images with nearest transforms in new directory

    with open('data_nn/train/transforms.json', 'r') as transforms_file:
        transforms_dict = json.load(transforms_file)
    image_transform_map = transforms_dict.get('image_transform_map')
    train_transforms = list(image_transform_map.items())

    nneighbors = []
    print ("test")
    for i, val_pair in enumerate (val_transforms):
        distances = []
        print("round: ", i)
        for train_pair in train_transforms:
            distance = np.linalg.norm(train_pair[1] - val_pair[1], "fro")
            print ("   ", distance)
            distances.append((train_pair[0], distance))
        distances.sort(key=operator.itemgetter(1))
        closest = distances[0]
        print("smallest distance: ", closest[1])
        nneighbors.append((val_pair[0], ) + closest)
        
    print(os.path.exists("data/nn"))

    if not os.path.exists("data/nn"):
        os.makedirs("data/nn")

    for nn in nneighbors:
        #print(nn)
        image_name = os.path.join("../data/train/", nn[1])
        nn_image = os.path.join("data/nn/", "nn_" + nn[0])


angles = np.linspace(0, np.pi*2, 25)
with open('data_nn/train/transforms.pkl', 'r') as transforms_file:
    transforms_dict = json.load(transforms_file)
    image_transform_map = transforms_dict.get('image_transform_map')
    train_transforms = list(image_transform_map.items())
radius = 20
transforms_list = []
height, width, yfov = 512, 512, np.pi / 3
camera_radius = 2.4
for angle in angles:
    phi = radius*np.cos(angle)
    theta = radius*np.sin(angle)
    camera_pose = get_sphere_pose(phi, theta, camera_radius)
    transforms_list.append(camera_pose)
print("Len Train Transforms: ", len(train_transforms))
print("Len Inference Transforms: ", len(transforms_list))
