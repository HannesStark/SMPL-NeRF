
import os
import numpy as np
import operator
import shutil
import json
from torch import norm
from tqdm import tqdm

def nearest_neighbor(inference_dir='data/val', use_pose=True):

    # take transforms from RaysFromCameraDataset and calculate the nearest transform out of the transform json for training images
    # we have a dict with image names and corresponding transformations
    # calculate nearest neighbor from input transform vs list of transformation matrices: frobenius norm
    # copy images with nearest transforms in new directory

    with open('../data/train/transforms.json', 'r') as transforms_file:
        transforms_dict = json.load(transforms_file)
    image_transform_map = transforms_dict.get('image_transform_map')
    train_transforms = list(image_transform_map.items())
    if use_pose:
        train_poses = transforms_dict.get('image_pose_map')
        #train_poses = list(train_poses.items())

    with open(os.path.join(inference_dir, 'transforms.json'), 'r') as transforms_file:
        transforms_dict = json.load(transforms_file)
    image_transform_map = transforms_dict.get('image_transform_map')
    inference_transforms = list(image_transform_map.items())
    if use_pose:
        inference_poses = transforms_dict.get('image_pose_map')
        #inference_poses = list(image_pose_map.items())


    nneighbors = []
    for i, inference_pair in tqdm(enumerate (inference_transforms)):  ## pair: image name & transform
        distances = []
        print("round: ", i)

        for train_pair in train_transforms:
            distance = np.linalg.norm(np.array(train_pair[1]) - np.array(inference_pair[1]), "fro")
            #print ("   ", distance)
            distances.append((train_pair[0], distance))
        distances.sort(key=operator.itemgetter(1))

        closest = distances[0]

        print("smallest camera distance: ", closest[1])
        # only take the pairs corresponding to the smallest distance into account
        if use_pose:

            best_camera = [(i[0], np.linalg.norm(np.array(train_poses[i[0]]) - np.array(inference_poses[inference_pair[0]]))) for i in distances if i[1] == closest[1]]
            best_camera.sort(key=operator.itemgetter(1))
            print("Length of equal camera positions: ", len(best_camera))
            closest = best_camera[0]


        print("smallest human distance: ", closest[1])
        nneighbors.append((inference_pair[0], ) + closest)

    if not os.path.exists("data/nn"):
        os.makedirs("data/nn")

    for nn in nneighbors:
        #print(nn)
        image_name = os.path.join("../data/train/", nn[1])
        nn_image = os.path.join("data/nn/", "nn_" + nn[0])
        shutil.copy(image_name, nn_image)


if __name__ == '__main__':
    nearest_neighbor(use_pose=True)