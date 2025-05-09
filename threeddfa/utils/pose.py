# coding: utf-8

"""
Reference: https://github.com/YadiraF/PRNet/blob/master/utils/estimate_pose.py

Calculating pose from the output 3DMM parameters, you can also try to use solvePnP to perform estimation
"""

__author__ = 'cleardusk'

import cv2
import numpy as np
from math import cos, sin, atan2, asin, sqrt, degrees

from .functions import calc_hypotenuse, plot_image


def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z

def calculate_2d_inplane_rotation_from_3d_orientation(R):
    """
    Calculates the 2D in-plane rotation angle needed to make the head's
    natural "up" direction align with the image's vertical axis.
    Assumes image Y-axis points downwards. Positive angle for CCW rotation.

    Args:
        R (np.ndarray): The 3x3 rotation matrix representing the head's orientation
                        in the camera coordinate system.

    Returns:
        float: The rotation angle in radians to align head's up with image's vertical.
    """
    # Define the canonical "up" vector in the head's local coordinate system
    # (Y-axis points up from neck to top of head)
    up_model = np.array([0., 1., 0.])

    # Transform this "up" vector to the camera's coordinate system
    up_camera = R @ up_model

    # up_camera[0] is the x-component in the image plane (horizontal)
    # up_camera[1] is the y-component in the image plane (vertical, positive down)
    # We want the head's "up" to align with the image's negative Y-axis ([0, -1]).
    # The angle of the projected vector (up_camera[0], up_camera[1])
    # with respect to the positive Y-axis (downwards) is atan2(up_camera[0], up_camera[1]).
    # A positive angle means up_camera is tilted towards positive X (right).
    # We need to rotate by the negative of this angle to align it vertically.
    angle_rad = atan2(up_camera[0], up_camera[1])

    # Return the negative of the calculated angle in radians.
    # A positive rotation for cv2.getRotationMatrix2D is CCW.
    # If head's "up" (when projected) is tilted towards positive X (right),
    # angle_rad will be positive. We need a CCW rotation (-angle_rad) to make it vertical.
    return -angle_rad

def calc_pose(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)
    # pose = [p * 180 / np.pi for p in pose]

    return P, pose, R


def build_camera_box(rear_size=90):
    point_3d = []
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = int(4 / 3 * rear_size)
    front_depth = int(4 / 3 * rear_size)
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    return point_3d


def plot_pose_box(img, P, ver, color=(40, 255, 0), line_width=2):
    """ Draw a 3D box as annotation of pose.
    Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    Args:
        img: the input image
        P: (3, 4). Affine Camera Matrix.
        kpt: (2, 68) or (3, 68)
    """
    llength = calc_hypotenuse(ver)
    point_3d = build_camera_box(llength)
    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    point_2d = point_3d_homo.dot(P.T)[:, :2]

    point_2d[:, 1] = - point_2d[:, 1]
    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(ver[:2, :27], 1)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

    return img


def viz_pose(img, param_lst, ver_lst, show_flag=False, wfp=None):
    for param, ver in zip(param_lst, ver_lst):
        P, pose = calc_pose(param)
        img = plot_pose_box(img, P, ver)
        # print(P[:, :3])
        print(f'yaw: {pose[0]:.1f}, pitch: {pose[1]:.1f}, roll: {pose[2]:.1f}')

    if wfp is not None:
        cv2.imwrite(wfp, img)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(img)

    return img
