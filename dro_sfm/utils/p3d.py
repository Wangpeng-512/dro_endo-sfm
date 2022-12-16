from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def load_yaml(file: str):
    import yaml
    with open(file, "r") as stream:
        intr = yaml.load(stream, Loader=yaml.FullLoader)
    return intr
class CameraModel(object):

    def __init__(self, path: str):
        """
        Args:
            path (str): Intrinsics file of yaml format.
        """
        param = load_yaml(path)
        self.K = torch.FloatTensor(param['intrinsics'])
        self.IK = torch.inverse(self.K)
        self.rd = torch.FloatTensor(param['radial_distortion'])
        self.td = torch.FloatTensor(param['tangent_distortion'])

    def to(self, device: str):
        self.K = self.K.to(device)
        self.IK = self.IK.to(device)
        self.rd = self.rd.to(device)
        self.td = self.td.to(device)
        return self

    def cuda(self):
        self.to('cuda')
        return self

    def cpu(self):
        self.to('cpu')
        return self

    def __call__(self, inverse: bool = False):
        return self.K if not inverse else self.IK


class DepthBackprojD(nn.Module):
    """Layer to transform a depth image into a point cloud.
    """

    def __init__(self, height: int, width: int, IK: torch.Tensor,
                 rdistor: List[float] = None, tdistor: List[float] = None,):
        super(DepthBackprojD, self).__init__()

        H, W = height, width
        v, u = torch.meshgrid(
            torch.linspace(0, 1, H).float(),
            torch.linspace(0, 1, W).float(), indexing='ij')
        i = torch.ones_like(u)

        self.height = height
        self.width = width
        self.uv = torch.stack([u, v, i], -1).view(H, W, 3, 1).to(IK.device)
        self.uv = nn.Parameter(self.uv, requires_grad=False)
        self.f = torch.FloatTensor([1 - 1 / W, 1 - 1 / H, 1.0]).view(1, 1, 3).to(IK.device)
        self.f = nn.Parameter(self.f, requires_grad=False)

        # A. back-projection
        self.IK = torch.eye(4)
        self.IK[:3,:3] = IK[0]
        self.IK = self.IK.view(4, 4).float().to(IK.device)
        self.pix = (self.IK[:3, :3] @ self.uv).view(H, W, 3)
        self.pix *= self.f  # u -> u(W-1)/W, v -> v(H-1)/H

        # B. de-distortion
        x, y = self.pix[:, :, 0:1], self.pix[:, :, 1:2]
        r2 = x * x + y * y
        # a). radial distortion
        kr = 1.0
        if rdistor is not None:
            k1, k2, k3 = rdistor
            r4 = r2 * r2
            r6 = r4 * r2
            kr += k1 * r2 + k2 * r4 + k3 * r6
        # b). tangent distortion
        kdx, kdy = 0.0, 0.0
        if tdistor is not None:
            p1, p2 = tdistor
            kdx, kdy = 2 * p1 * x * y + p2 * (r2 + 2 * x * x), p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        # c). total distortion
        self.pix[:, :, 0:1] = x * kr + kdx
        self.pix[:, :, 1:2] = y * kr + kdy
        self.pix = nn.Parameter(self.pix)

    def forward(self, depth: torch.Tensor, ret_scaled_depth=False,
                use_mono: bool = True, dim: int = -3):
        """ 
        Args:
            depth: [B,N,H,W]
            K (torch.Tensor, optional): The *normalized* camera intrinsics matrix [4, 4].
        """
        with torch.no_grad():
            H, W = self.height, self.width
            if depth.shape[-2:] != (H, W):
                depth = F.interpolate(depth, (H, W), mode='bilinear', align_corners=False)

            pc = depth.unsqueeze(-1) * self.pix

        if use_mono:
            I = torch.ones_like(depth).unsqueeze(-1)
            pc = torch.cat((pc, I), dim=-1)

        if dim in (-3, 2):
            pc = pc.permute(0, 1, 4, 2, 3).contiguous()

        if ret_scaled_depth:
            return pc, depth
        else:
            return pc




def npxyz_o3d(xyz: np.ndarray):
    """
    Convert Nx3 numpy array to open3d point cloud. 
    Args:
        xyz (np.ndarray): Nx3 numpy array points
    Returns:
        [type]: [description]
    """
    assert(len(xyz.shape) == 2)
    assert(xyz.shape[1] == 3)
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def o3d_npxyz(pcd: o3d.geometry.PointCloud):
    return np.asarray(pcd.points)


def npxyzw_o3d(xyzw: np.ndarray, cmap='viridis'):
    """
    Convert Nx4 numpy array to open3d point cloud. 
    Args:
        xyzw (np.ndarray):Nx4 numpy array points with value (eg. TSDF)
        cmap (str, optional): matplotlib color maps. Defaults to ''.
    Returns:
        [type]: [description]
    """
    assert(len(xyzw.shape) == 2)
    assert(xyzw.shape[1] == 4)
    cmap = plt.cm.get_cmap(cmap)
    rgb = cmap(xyzw[:, 3])[:, :3]
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzw[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def np6d_o3d_normal(pose: np.ndarray):
    """
    Convert Nx6 numpy array to open3d point cloud with normals. 
    Args:
        pose (np.ndarray):Nx6 numpy array points with normals.
    Returns:
        [type]: [description]
    """
    assert(len(pose.shape) == 2)
    assert(pose.shape[1] == 6)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pose[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(pose[:, 3:])
    return pcd


def np6d_o3d_color(point: np.ndarray):
    """
    Convert Nx6 numpy array to open3d point cloud with colors. 
    Args:
        pose (np.ndarray):Nx6 numpy array points with colors.
    Returns:
        [type]: [description]
    """
    assert(len(point.shape) == 2)
    assert(point.shape[1] == 6)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point[:, 3:])
    return pcd


def o3d_show(items, *args, **kwargs):
    o3d.visualization.draw_geometries(items, *args, **kwargs)

