#!/usr/bin/env python3
"""
AnimalPose 数据集可视化脚本 - 最终完整版本
正确的标签格式: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, kpt1_x, kpt1_y, kpt1_v, ...].
"""

from pathlib import Path

import cv2
from tqdm import tqdm

# AnimalPose 数据集的骨架连接定义
SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),  # 鼻子到眼睛
    (1, 3),
    (2, 4),  # 眼睛到耳朵
    (5, 6),  # 肩膀之间
    (5, 7),
    (6, 8),  # 肩膀到肘部
    (7, 9),
    (8, 10),  # 肘部到腕部
    (5, 11),
    (6, 12),  # 肩膀到髋部
    (11, 12),  # 髋部之间
    (11, 13),
    (12, 14),  # 髋部到膝盖
    (13, 15),
    (14, 16),  # 膝盖到踝部
    (12, 17),  # 髋部右到尾部根
    (17, 18),
    (18, 19),  # 尾部连接
]

KPT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "tail_root",
    "tail_mid",
    "tail_tip",
]


def parse_animalpose_label(label_path, img_height, img_width):
    """解析 AnimalPose YOLO 格式标签 格式: [class_id, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, ...] 返回: keypoints
    列表 (x, y, visibility) 和 bbox.
    """
    with open(label_path) as f:
        parts = f.read().strip().split()

    if len(parts) < 65:
        return None, None

    # 解析类别和 bbox
    int(parts[0])
    bbox_x = float(parts[1]) * img_width
    bbox_y = float(parts[2]) * img_height
    bbox_w = float(parts[3]) * img_width
    bbox_h = float(parts[4]) * img_height
    bbox = (bbox_x, bbox_y, bbox_w, bbox_h)

    # 解析 20 个关键点 (每个 3 个值: x, y, visibility)
    keypoints = []
    for i in range(20):
        idx = 5 + i * 3  # 从第 5 个值开始
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

    if keypoints is None or len(keypoints) < 20:
        return img_drawn

    # 绘制 bbox (可选)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(img_drawn, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 255, 0), 2)

    # 绘制骨架连接 (绿色)
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_pt = keypoints[start_idx]
            end_pt = keypoints[end_idx]

            # 只有两个点都有效且可见才绘制连接
            if start_pt[2] > 0 and end_pt[2] > 0:
                pt1 = (int(start_pt[0]), int(start_pt[1]))
                pt2 = (int(end_pt[0]), int(end_pt[1]))
                cv2.line(img_drawn, pt1, pt2, (0, 255, 0), 2)

    # 绘制关键点
    for idx, (x, y, v) in enumerate(keypoints):
        pt = (int(x), int(y))
        if v > 0:  # 可见的关键点用绿色
            cv2.circle(img_drawn, pt, 5, (0, 255, 0), -1)
            cv2.circle(img_drawn, pt, 5, (255, 255, 255), 1)
            # 添加关键点编号
            cv2.putText(img_drawn, str(idx), (pt[0] + 8, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:  # 不可见的关键点用红色标记
            cv2.circle(img_drawn, pt, 4, (0, 0, 255), -1)

    return img_drawn


def main():
    dataset_dir = Path("/home/ng204/GH/ultralytics+rtmo/datasets/animalpose_yolo")
    images_dir = dataset_dir / "images" / "test"
    labels_dir = dataset_dir / "labels" / "test"
    output_dir = Path("/home/ng204/GH/ultralytics+rtmo/runs/keshihua/animalpose")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片
    image_files = (
        sorted(images_dir.glob("*.jpeg")) + sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    )

    print(f"Found {len(image_files)} images")

    success_count = 0
    fail_count = 0

    for img_file in tqdm(image_files, desc="Visualizing"):
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

            kpts, bbox = parse_animalpose_label(str(label_file), h, w)
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

    print("\nVisualization complete!")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
