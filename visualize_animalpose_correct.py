#!/usr/bin/env python3
"""
AnimalPose 数据集可视化脚本 - 使用正确的骨架连接
从官方 JSON 注解文件中提取的骨架定义
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# AnimalPose 官方骨架连接（从 JSON 转换为 0-indexed）
# 原始 JSON 中: [[1,2], [1,3], [2,4], [1,5], [2,5], [5,6], [6,8], [7,8], ...]
# 转换为 0-indexed:
SKELETON_CONNECTIONS = [
    (0, 1),      # L_Eye - R_Eye
    (0, 2),      # L_Eye - L_EarBase
    (1, 3),      # R_Eye - R_EarBase
    (0, 4),      # L_Eye - Nose
    (1, 4),      # R_Eye - Nose
    (4, 5),      # Nose - Throat
    (5, 7),      # Throat - Withers
    (6, 7),      # TailBase - Withers
    (5, 8),      # Throat - L_F_Elbow
    (8, 12),     # L_F_Elbow - L_F_Knee
    (12, 16),    # L_F_Knee - L_F_Paw
    (5, 9),      # Throat - R_F_Elbow
    (9, 13),     # R_F_Elbow - R_F_Knee
    (13, 17),    # R_F_Knee - R_F_Paw
    (6, 10),     # TailBase - L_B_Elbow
    (10, 14),    # L_B_Elbow - L_B_Knee
    (14, 18),    # L_B_Knee - L_B_Paw
    (6, 11),     # TailBase - R_B_Elbow
    (11, 15),    # R_B_Elbow - R_B_Knee
    (15, 19),    # R_B_Knee - R_B_Paw
]

KPT_NAMES = [
    'L_Eye', 'R_Eye', 'L_EarBase', 'R_EarBase', 'Nose',
    'Throat', 'TailBase', 'Withers', 'L_F_Elbow', 'R_F_Elbow',
    'L_B_Elbow', 'R_B_Elbow', 'L_F_Knee', 'R_F_Knee',
    'L_B_Knee', 'R_B_Knee', 'L_F_Paw', 'R_F_Paw',
    'L_B_Paw', 'R_B_Paw'
]

def parse_animalpose_label(label_path, img_height, img_width):
    """
    解析 YOLO 格式标签
    格式: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, ...]
    """
    with open(label_path, 'r') as f:
        parts = f.read().strip().split()

    if len(parts) < 65:
        return None, None

    class_id = int(parts[0])
    bbox_x = float(parts[1]) * img_width
    bbox_y = float(parts[2]) * img_height
    bbox_w = float(parts[3]) * img_width
    bbox_h = float(parts[4]) * img_height
    bbox = (bbox_x, bbox_y, bbox_w, bbox_h)

    keypoints = []
    for i in range(20):
        idx = 5 + i * 3
        if idx + 2 < len(parts):
            x = float(parts[idx]) * img_width
            y = float(parts[idx + 1]) * img_height
            v = float(parts[idx + 2])
            keypoints.append((x, y, v))
        else:
            keypoints.append((0, 0, 0))

    return keypoints, bbox

def draw_keypoints_and_skeleton(img, keypoints, bbox=None):
    """在图上绘制关键点和骨架"""
    img_drawn = img.copy()

    if keypoints is None or len(keypoints) < 20:
        return img_drawn

    # 绘制骨架连接 (绿色线条)
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_pt = keypoints[start_idx]
            end_pt = keypoints[end_idx]

            # 两个点都可见才绘制
            if start_pt[2] > 0 and end_pt[2] > 0:
                pt1 = (int(start_pt[0]), int(start_pt[1]))
                pt2 = (int(end_pt[0]), int(end_pt[1]))
                cv2.line(img_drawn, pt1, pt2, (0, 255, 0), 2)

    # 绘制关键点
    for idx, (x, y, v) in enumerate(keypoints):
        pt = (int(x), int(y))
        if v > 0:  # 可见的关键点
            cv2.circle(img_drawn, pt, 5, (0, 255, 0), -1)
            cv2.circle(img_drawn, pt, 5, (255, 255, 255), 1)
            cv2.putText(img_drawn, str(idx), (pt[0] + 8, pt[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        else:  # 不可见的关键点用红色
            cv2.circle(img_drawn, pt, 3, (0, 0, 255), -1)

    return img_drawn

def main():
    dataset_dir = Path('/home/ng204/GH/ultralytics+rtmo/datasets/animalpose_yolo')
    images_dir = dataset_dir / 'images' / 'test'
    labels_dir = dataset_dir / 'labels' / 'test'
    output_dir = Path('/home/ng204/GH/ultralytics+rtmo/runs/keshihua/animalpose')

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob('*.jpeg')) + sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png'))

    print(f"Found {len(image_files)} images")
    print(f"Using correct AnimalPose skeleton with {len(SKELETON_CONNECTIONS)} connections")

    success_count = 0
    fail_count = 0

    for img_file in tqdm(image_files, desc="Visualizing"):
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                fail_count += 1
                continue

            h, w = img.shape[:2]

            label_file = labels_dir / (img_file.stem + '.txt')
            if not label_file.exists():
                fail_count += 1
                continue

            kpts, bbox = parse_animalpose_label(str(label_file), h, w)
            if kpts is None:
                fail_count += 1
                continue

            img_vis = draw_keypoints_and_skeleton(img, kpts, bbox)

            output_path = output_dir / (img_file.stem + '_vis.jpg')
            cv2.imwrite(str(output_path), img_vis)
            success_count += 1

        except Exception as e:
            fail_count += 1
            continue

    print(f"\n✓ Visualization complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Saved to: {output_dir}")

if __name__ == '__main__':
    main()
