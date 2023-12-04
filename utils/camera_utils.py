#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

# ref: regnerf/internal/datasets.py
def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world

# ref: regnerf/internal/datasets.py
def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

# ref: regnerf/internal/datasets.py
def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)

# ref: regnerf/internal/datasets.py
def focus_pt_fn(poses):
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def cameraList_from_camInfos(cam_infos, resolution_scale, args, unobserved, isTrain=True):
    if isTrain and unobserved:
        camera_list = []
        poses = []
        width, height = 0, 0

        for id, c in enumerate(cam_infos):
            tmp_cam = loadCam(args, id, c, resolution_scale)
            height = tmp_cam.image_height
            width = tmp_cam.image_width
            poses.append(tmp_cam.world_view_transform)
            camera_list.append(tmp_cam)
        
        # ref: regnerf/internal/datasets.py
        # here we are precalculating the values needed when generating random poses
        positions = poses[:, :3, 3]
        radii = np.percentile(np.abs(positions), 100, 0)
        radii = np.concatenate([radii, [1.]])
        cam2world = poses_avg(poses)
        up = poses[:, :3, 1].mean(0)
        z_axis = focus_pt_fn(poses)
        unobserved_list = generate_cams(args.n_random_cams, radii, cam2world, up, z_axis, height, width, args.data_device)

        return camera_list, unobserved_list
    else:
        camera_list = []
        
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCam(args, id, c, resolution_scale))

        return camera_list, []

def generate_cams(n_random_cams, radii, cam2world, up, z_axis, height, width, device):
    cams = []

    for _ in range(n_random_cams):
        # generate T (ref: regnerf/internal/datasets.py)
        t = radii * np.concatenate([2 * (np.random.rand(3) - 0.5), [1,]])
        position = cam2world @ t
        z_axis_i = z_axis + np.random.randn(*z_axis.shape) * 0.125
        # generate R (here homogeneous representation not sure)
        r = viewmatrix(z_axis_i, up, position, True)[:3, :3].transpose() / position[3]
        t = position[:3] / position[3]

        # get focus
        f = np.linalg.norm(position - z_axis_i)
        # generate FovX
        fovX = focal2fov(f, width)
        # generate FovY
        fovY = focal2fov(f, height)
        # two ids set to None and image_name set to empty for generated unobserved camera views
        cams.append(Camera(colmap_id=None, R=r, T=t, 
                            FoVx=fovX, FoVy=fovY, 
                            image=None, gt_alpha_mask=None,
                            image_name='', uid=None, 
                            width=width, height=height, 
                            data_device=device, isUnobserved=True))
    return cams

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
