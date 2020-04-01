# -*- coding: utf-8 -*-
import math
from typing import Tuple

import pyrender
import numpy as np
import os.path as osp
import argparse

import torch

import smplx
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pyrender
import trimesh

def get_pose_matrix(x: float=0, y: float=0, z: float=0, 
                    phi: float=0,  theta: float=0, psi: float=0) -> np.array:
    """Returns pose matrix (3, 4) for given translation/rotation
        parameters.
    
        Args:
            x (float): x coordinate
            y (float): y coordinate
            z (float): z coordinate
            phi (float): rotation around x axis in degrees
            theta (float): rotation around y axis in degrees
            psi (float): rotation around x axis in degree
    """
    rot = R.from_euler('xyz',[phi, theta, psi],degrees=True).as_matrix()
    trans = np.array([[x, y, z]])
    pose = np.concatenate((np.concatenate((rot, trans.T), axis=1),
                           [[0, 0, 0, 1]]), axis=0)
    return pose

def get_circle_pose(alpha: float, r: float) -> np.array:
    """ Returns pose matrix for angle alpha in xz-circle with radius r around 
        y-axis (alpha = 0 corresponds position (0, 0, r))
    
        Args:
            alpha (float): rotation around y axis in degrees
            r (float): radius of circle 
    """
    z = r*np.cos(np.radians(alpha))
    x = r*np.sin(np.radians(alpha))
    pose = get_pose_matrix(x=x, z=z, theta=alpha)
    return pose

def get_sphere_pose(alpha: float, beta: float, r: float) -> np.array:
    """Returns pose matrix for angle alpha in xz-circle with radius r around
        y-axis and angle beta in yz-circle around x-axis (spherical coordinates)
        
        Args:
            alpha (float): rotation around y axis in degrees
            beta (float): rotation around x axis in degrees
            r (float): radius of circle 
    """
    z = r*np.cos(np.radians(beta))*np.cos(np.radians(alpha))
    x = r*np.cos(np.radians(beta))*np.sin(np.radians(alpha))
    y = r*np.sin(np.radians(beta))
    pose = get_pose_matrix(x=x, y=y, z=z, theta=alpha, phi=-beta)
    return pose

def camera_origin_direction(x: float, y: float, z: float) -> Tuple[float, float]:
    """Calculates phi and theta in degrees for a camera too face the origin of the coordinate system

        Args:
            x (float): x coordinate of camera
            y (float): y coordinate of camera
            z (float): z coordinate of camera
        """
    phi = np.degrees(np.arctan2(y, z))
    theta = np.degrees(np.arctan2(x, z))
    return phi.item(), theta.item()

