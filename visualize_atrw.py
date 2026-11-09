#!/usr/bin/env python3
"""
ATRW (Amur Tiger Re-identification in the Wild) 数据集可视化脚本
15 个关键点，官方骨架连接定义
标签格式: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, ...].
"""

from pathlib import Path

import cv2
from tqdm import tqdm

# ATRW 官方骨架连接（0-indexed）
# 关键点：[left_ear, right_ear, nose, right_shoulder, right_front_paw,
#          left_shoulder, left_front_paw, right_hip, right_knee, right_back_paw,
#          left_hip, left_knee, left_back_paw, tail, center]
SKELETON_CONNECTIONS = [
    (0, 2),  # left_ear - nose
    (1, 2),  # right_ear - nose
    (2, 14),  # nose - center
    (5, 6),  # left_shoulder - left_front_paw
    (5, 14),  # left_shoulder - center
    (3, 4),  # right_shoulder - right_front_paw
    (3, 14),  # right_shoulder - center
    (13, 14),  # tail - center
    (9, 8),  # right_back_paw - right_knee
    (8, 7),  # right_knee - right_hip
    (7, 13),  # right_hip - tail
    (12, 11),  # left_back_paw - left_knee
    (11, 10),  # left_knee - left_hip
    (10, 13),  # left_hip - tail
]

KPT_NAMES = [
    "left_ear",
    "right_ear",
    "nose",
    "right_shoulder",
    "right_front_paw",
    "left_shoulder",
    "left_front_paw",
    "right_hip",
    "right_knee",
    "right_back_paw",
    "left_hip",
    "left_knee",
    "left_back_paw",
    "tail",
    "center",
]


def parse_atrw_label(label_path, img_height, img_width):
    """解析 ATRW YOLO 格式标签 格式: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, ...].
    """
    with open(label_path) as f:
        parts = f.read().strip().split()

    if len(parts) < 50:  # 1 + 4 + 45
        return None, None

    int(parts[0])
    bbox_x = float(parts[1]) * img_width
    bbox_y = float(parts[2]) * img_height
    bbox_w = float(parts[3]) * img_width
    bbox_h = float(parts[4]) * img_height
    bbox = (bbox_x, bbox_y, bbox_w, bbox_h)

    keypoints = []
    for i in range(15):
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
    """在图上绘制关键点和骨架."""
    img_drawn = img.copy()

    if keypoints is None or len(keypoints) < 15:
        return img_drawn

    # 绘制骨架连接 (绿色线条)
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_pt = keypoints[start_idx]
            end_pt = keypoints[end_idx]

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
            cv2.putText(img_drawn, str(idx), (pt[0] + 8, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        else:  # 不可见的关键点用红色
            cv2.circle(img_drawn, pt, 3, (0, 0, 255), -1)

    return img_drawn


def main():
    dataset_dir = Path("/home/ng204/GH/ultralytics+rtmo/datasets/atrw")
    images_dir = dataset_dir / "images" / "val"
    labels_dir = dataset_dir / "labels" / "val"
    output_dir = Path("/home/ng204/GH/ultralytics+rtmo/runs/keshihua/atrw")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = (
        sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg")) + sorted(images_dir.glob("*.png"))
    )

    print(f"Found {len(image_files)} images in val set")
    print(f"Using ATRW skeleton with {len(SKELETON_CONNECTIONS)} connections")

    success_count = 0
    fail_count = 0

    for img_file in tqdm(image_files, desc="Visualizing ATRW"):
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                fail_count += 1
                continue

            h, w = img.shape[:2]

            label_file = labels_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                fail_count += 1
                continue

            kpts, bbox = parse_atrw_label(str(label_file), h, w)
            if kpts is None:
                fail_count += 1
                continue

            img_vis = draw_keypoints_and_skeleton(img, kpts, bbox)

            output_path = output_dir / (img_file.stem + "_vis.jpg")
            cv2.imwrite(str(output_path), img_vis)
            success_count += 1

        except Exception:
            fail_count += 1
            continue

    print("\n✓ ATRW Visualization complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    main()
