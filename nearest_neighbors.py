
import os
import numpy as np
import operator
import shutil
import json

def nearest_neighbor():

    # take transforms from RaysFromCameraDataset and calculate the nearest transform out of the transform json for training images
    # we have a dict with image names and corresponding transformations
    # calculate nearest neighbor from input transform vs list of transformation matrices: frobenius norm
    # copy images with nearest transforms in new directory

    with open('data/train/transforms.json', 'r') as transforms_file:
        transforms_dict = json.load(transforms_file)
    image_transform_map = transforms_dict.get('image_transform_map')
    train_transforms = list(image_transform_map.items())

    with open('data/val/transforms.json', 'r') as transforms_file:
        transforms_dict = json.load(transforms_file)
    image_transform_map = transforms_dict.get('image_transform_map')
    val_transforms = list(image_transform_map.items())


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
        image_name = os.path.join("data/train/", nn[1])
        nn_image = os.path.join("data/nn/", "nn_" + nn[0])
        shutil.copy(image_name, nn_image)


if __name__ == '__main__':
    nearest_neighbor()