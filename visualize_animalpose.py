#!/usr/bin/env python3
"""
AnimalPose 数据集可视化脚本
将关键点和骨架连接绘制在图片上
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# AnimalPose 数据集的骨架连接定义
# 20 个关键点的标准定义
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),      # 鼻子到眼睛
    (1, 3), (2, 4),      # 眼睛到耳朵
    (5, 6),              # 肩膀之间
    (5, 7), (6, 8),      # 肩膀到肘部
    (7, 9), (8, 10),     # 肘部到腕部
    (5, 11), (6, 12),    # 肩膀到髋部
    (11, 12),            # 髋部之间
    (11, 13), (12, 14),  # 髋部到膝盖
    (13, 15), (14, 16),  # 膝盖到踝部
    (12, 17),            # 髋部右到尾部根
    (17, 18), (18, 19)   # 尾部连接
]

# 关键点名称
KPT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
    'tail_root',      # 17
    'tail_mid',       # 18
    'tail_tip'        # 19
]

# 颜色定义
COLORS = {
    'skeleton': (0, 255, 0),      # 绿色用于骨架
    'kpt_visible': (0, 255, 0),   # 绿色用于可见关键点
    'kpt_hidden': (0, 0, 255)     # 红色用于不可见关键点
}

def parse_yolo_labels(label_path, img_height, img_width):
    """
    解析 YOLO 格式的关键点标签
    格式: [class_id, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, kpt1_x, kpt1_y, kpt1_v, ...]
    返回: [(x, y, visibility), ...] 列表，共 20 个关键点
    """
    keypoints = []

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # 格式: class_id(1) + bbox(4) + keypoints(60) = 65 values
        if len(parts) < 65:
            continue

        # 跳过 class_id 和 bbox，从第 6 个值开始是关键点数据
        # 每个关键点有 3 个值：x, y, visibility
        start_idx = 5  # 跳过 class_id(0) 和 bbox(1-4)

        for i in range(start_idx, len(parts), 3):
            if i + 2 < len(parts):
                x = float(parts[i]) * img_width
                y = float(parts[i + 1]) * img_height
                vis = float(parts[i + 2])
                keypoints.append((x, y, vis))

    return keypoints

def draw_keypoints_and_skeleton(img, keypoints, visibility_threshold=0.5):
    """
    在图片上绘制关键点和骨架连接
    """
    img_with_kpts = img.copy()

    # 绘制骨架连接
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_kpt = keypoints[start_idx]
            end_kpt = keypoints[end_idx]

            # 只有两个关键点都可见时才绘制连接
            if start_kpt[2] >= visibility_threshold and end_kpt[2] >= visibility_threshold:
                pt1 = (int(start_kpt[0]), int(start_kpt[1]))
                pt2 = (int(end_kpt[0]), int(end_kpt[1]))
                cv2.line(img_with_kpts, pt1, pt2, COLORS['skeleton'], 2)

    # 绘制关键点
    for idx, (x, y, vis) in enumerate(keypoints):
        if vis >= visibility_threshold:
            pt = (int(x), int(y))
            color = COLORS['kpt_visible']
            cv2.circle(img_with_kpts, pt, 5, color, -1)
            cv2.circle(img_with_kpts, pt, 5, (255, 255, 255), 1)
            # 添加关键点编号
            cv2.putText(img_with_kpts, str(idx), (pt[0] + 5, pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        else:
            # 不可见的关键点用红色标记（但仍然显示，便于调试）
            pt = (int(x), int(y))
            if pt[0] > 0 and pt[0] < img.shape[1] and pt[1] > 0 and pt[1] < img.shape[0]:
                color = COLORS['kpt_hidden']
                cv2.circle(img_with_kpts, pt, 3, color, -1)

    return img_with_kpts

def main():
    # 配置路径
    dataset_dir = Path('/home/ng204/GH/ultralytics+rtmo/datasets/animalpose_yolo')
    images_dir = dataset_dir / 'images' / 'test'
    labels_dir = dataset_dir / 'labels' / 'test'

    output_dir = Path('/home/ng204/GH/ultralytics+rtmo/runs/keshihua/animalpose')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_files = sorted(images_dir.glob('*.jpeg')) + sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png'))

    print(f"Found {len(image_files)} images to visualize")
    print(f"Output directory: {output_dir}")

    # 处理每张图片
    successful = 0
    failed = 0

    for img_file in tqdm(image_files):
        # 读取图片
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Failed to read image: {img_file}")
            failed += 1
            continue

        img_height, img_width = img.shape[:2]

        # 对应的标签文件
        label_file = labels_dir / (img_file.stem + '.txt')
        if not label_file.exists():
            print(f"Label file not found: {label_file}")
            failed += 1
            continue

        # 解析标签
        keypoints = parse_yolo_labels(str(label_file), img_height, img_width)

        if len(keypoints) == 0:
            failed += 1
            continue

        if len(keypoints) != 20:
            print(f"Warning: Expected 20 keypoints, got {len(keypoints)} for {img_file.name}")
            failed += 1
            continue

        # 绘制关键点和骨架
        img_vis = draw_keypoints_and_skeleton(img, keypoints)

        # 保存结果
        output_file = output_dir / (img_file.stem + '_vis.jpg')
        cv2.imwrite(str(output_file), img_vis)
        successful += 1

    print(f"\nVisualization complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main()
