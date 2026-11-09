#!/usr/bin/env python

"""
File Name: pose_cfg.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/10/16
Last Modified: 2024/10/16
Version: 1.0.

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/6/29] wux024: Initial file creation
"""

import numpy as np

OKS_SIGMA_USER = np.array([0.246, 0.182, 0.184, 0.189, 0.190, 0.252])  # fish 6-kpt, derived from training data distribution
SKELETON_USER =   [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
  ]


def GetOKSSigma():
    global OKS_SIGMA_USER
    return OKS_SIGMA_USER

def SetOKSSigma(sigma):
    global OKS_SIGMA_USER
    OKS_SIGMA_USER = np.array(sigma)

def SetDefaultOKSSigma(kpt_nums):
    global OKS_SIGMA_USER
    OKS_SIGMA_USER = np.array([1.0 / kpt_nums] * kpt_nums)

def GetSkeleton():
    global SKELETON_USER
    return SKELETON_USER

def SetSkeleton(skeleton):
    global SKELETON_USER
    SKELETON_USER = skeleton


# 鱼类6关键点骨骼连接（用户可在训练前调用 SetSkeleton 覆盖）
SKELETON_FISH = [
    [0, 1],  # 嘴 - 左眼
    [0, 2],  # 嘴 - 右眼
    [1, 2],  # 左眼 - 右眼
    [0, 3],  # 嘴 - 背鳍中部
    [3, 4],  # 背鳍中部 - 尾柄前端
    [4, 5],  # 尾柄前端 - 尾柄后端
]


def GetFishSkeleton():
    return SKELETON_FISH


def compute_bone_length_stats(label_keypoints, skeleton):
    """
    从 GT 关键点数组计算每根骨骼的长度均值和标准差。

    Args:
        label_keypoints: (N, nk_pts, ndim) numpy array，坐标单位需一致
        skeleton: [[i, j], ...] 骨骼连接列表

    Returns:
        means: (n_bones,) ndarray
        stds:  (n_bones,) ndarray，最小值 1e-3
    """
    means, stds = [], []
    for i, j in skeleton:
        kp_i = label_keypoints[:, i, :2]
        kp_j = label_keypoints[:, j, :2]
        if label_keypoints.shape[2] == 3:
            vis_i = label_keypoints[:, i, 2]
            vis_j = label_keypoints[:, j, 2]
            valid = ((vis_i > 0) | ((kp_i[:, 0] != 0) | (kp_i[:, 1] != 0)))
            valid &= ((vis_j > 0) | ((kp_j[:, 0] != 0) | (kp_j[:, 1] != 0)))
        else:
            valid = (
                ((kp_i[:, 0] != 0) | (kp_i[:, 1] != 0)) &
                ((kp_j[:, 0] != 0) | (kp_j[:, 1] != 0))
            )
        if valid.sum() < 2:
            means.append(1.0)
            stds.append(1.0)
            continue
        lengths = np.linalg.norm(kp_j[valid] - kp_i[valid], axis=1)
        means.append(float(lengths.mean()))
        stds.append(float(max(lengths.std(), 1e-3)))
    return np.array(means), np.array(stds)
