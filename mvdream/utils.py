# Directyle copied from bytedance/MVDream/mvdream/camera_utils.py

import numpy as np
import torch
import random


def create_camera_to_world_matrix(elevation, azimuth, cam_dist=1.0):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
    x = np.cos(elevation) * np.cos(azimuth) * cam_dist
    y = np.cos(elevation) * np.sin(azimuth) * cam_dist
    z = np.sin(elevation) * cam_dist
    
    # Calculate camera position, target, and up vectors
    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    
    # Construct view matrix
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.array([right, new_up, -forward]).T
    cam2world[:3, 3] = camera_pos
    return cam2world


def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_matrix_blender = np.dot(flip_yz, camera_matrix)
    else:
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        if camera_matrix.ndim == 3:
            flip_yz = flip_yz.unsqueeze(0)
        camera_matrix_blender = torch.matmul(flip_yz.to(camera_matrix), camera_matrix)
    return camera_matrix_blender

def convert_blender_to_opencv(camera_matrix):
    trans = np.diag([1., -1., -1., 1.])
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix_opencv = camera_matrix @ trans
    else:
        trans = torch.from_numpy(trans).to(camera_matrix)
        camera_matrix_opencv = camera_matrix @ trans
    return camera_matrix_opencv


def normalize_camera(camera_matrix):
    ''' normalize the camera location onto a unit-sphere'''
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    return camera_matrix


def get_camera(num_frames, elevation=15, azimuth_start=0, azimuth_span=360, opencv_coord=False, cam_dist=1.5):
    angle_gap = azimuth_span / num_frames
    cameras = []
    for azimuth in np.arange(azimuth_start, azimuth_span+azimuth_start, angle_gap):
        # azimuth = random.uniform(0, 360)
        # elevation = random.uniform(-30, 30)
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth, cam_dist)
        if opencv_coord:
            camera_matrix = convert_blender_to_opencv(camera_matrix)
        cameras.append(camera_matrix)
    return torch.tensor(np.stack(cameras, 0)).float()

def get_camera2(num_frames, elevation=15, azimuth_list=None, opencv_coord=False, cam_dist=1.5):
    cameras = []
    for i in range(4):
        # azimuth = random.uniform(0, 360)
        # elevation = random.uniform(-30, 30)
        azimuth = azimuth_list[i]
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth, cam_dist)
        if opencv_coord:
            camera_matrix = convert_blender_to_opencv(camera_matrix)
        cameras.append(camera_matrix)
    return torch.tensor(np.stack(cameras, 0)).float()


from kiui.cam import orbit_camera
# def get_camera(
#     num_frames, elevation=15, azimuth_start=0, azimuth_span=360, blender_coord=False, extra_view=False,
# ):
#     angle_gap = azimuth_span / num_frames
#     cameras = []
#     for azimuth in np.arange(azimuth_start, azimuth_span + azimuth_start, angle_gap):
        
#         pose = orbit_camera(-elevation, azimuth, radius=1) # kiui's elevation is negated, [4, 4]

#         # opengl to blender
#         if blender_coord:
#             pose[2] *= -1
#             pose[[1, 2]] = pose[[2, 1]]

#         cameras.append(pose.flatten())

#     if extra_view:
#         cameras.append(np.zeros_like(cameras[0]))

#     return torch.from_numpy(np.stack(cameras, axis=0)).float() # [num_frames, 16]

def get_camera_GS(
    num_frames, elevations, azimuths, blender_coord=False, extra_view=False,
): 
    cameras = []
    for i in range(len(azimuths)):
        azimuth = azimuths[i][0]
        pose = create_camera_to_world_matrix(elevations[0][0], azimuth, cam_dist=1) 

        # opengl to blender
        if blender_coord:
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        cameras.append(pose)
        # cameras.append(pose.flatten())

    if extra_view:
        cameras.append(np.zeros_like(cameras[0]))

    return torch.from_numpy(np.stack(cameras, axis=0)).float() # [num_frames, 4, 4]