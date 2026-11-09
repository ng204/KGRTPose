#!/usr/bin/env python3
"""
AnimalPose 数据集可视化脚本 - 正确的格式解析
格式: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, kpt1_x, kpt1_y, kpt1_v, ...].
"""

from pathlib import Path

import cv2
from tqdm import tqdm

# AnimalPose 骨架连接定义
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

COLORS = {
    "skeleton": (0, 255, 0),  # 绿色
    "kpt_visible": (0, 255, 0),  # 绿色
    "kpt_hidden": (0, 0, 255),  # 红色
}


def parse_yolo_labels(label_path, img_height, img_width):
    """解析 YOLO 格式标签 格式: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, ...].
    """
    keypoints = []
    with open(label_path) as f:
        line = f.readline().strip()

    parts = line.split()

    # 跳过前 5 个值 (class + bbox)，从第 6 个值开始解析关键点
    idx = 5
    while idx + 2 < len(parts):
        x = float(parts[idx]) * img_width
        y = float(parts[idx + 1]) * img_height
        vis = float(parts[idx + 2])
        keypoints.append((x, y, vis))
        idx += 3

    return keypoints


def draw_keypoints_and_skeleton(img, keypoints, vis_threshold=0.5):
    """绘制关键点和骨架连接."""
    img_vis = img.copy()

    # 绘制骨架
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_kpt = keypoints[start_idx]
            end_kpt = keypoints[end_idx]

            if start_kpt[2] >= vis_threshold and end_kpt[2] >= vis_threshold:
                pt1 = (int(start_kpt[0]), int(start_kpt[1]))
                pt2 = (int(end_kpt[0]), int(end_kpt[1]))
                cv2.line(img_vis, pt1, pt2, COLORS["skeleton"], 2)

    # 绘制关键点
    for idx, (x, y, vis) in enumerate(keypoints):
        pt = (int(x), int(y))
        if vis >= vis_threshold:
            cv2.circle(img_vis, pt, 5, COLORS["kpt_visible"], -1)
            cv2.circle(img_vis, pt, 5, (255, 255, 255), 1)
            cv2.putText(img_vis, str(idx), (pt[0] + 5, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.circle(img_vis, pt, 3, COLORS["kpt_hidden"], -1)

    return img_vis


def main():
    dataset_dir = Path("/home/ng204/GH/ultralytics+rtmo/datasets/animalpose_yolo")
    images_dir = dataset_dir / "images" / "test"
    labels_dir = dataset_dir / "labels" / "test"

    output_dir = Path("/home/ng204/GH/ultralytics+rtmo/runs/keshihua/animalpose")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取图片文件
    image_files = sorted(
        list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    )

    print(f"Found {len(image_files)} images to visualize")
    print(f"Output directory: {output_dir}")

    success_count = 0
    fail_count = 0

    for img_file in tqdm(image_files):
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                fail_count += 1
                continue

            img_height, img_width = img.shape[:2]

            label_file = labels_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                fail_count += 1
                continue

            keypoints = parse_yolo_labels(str(label_file), img_height, img_width)

            if len(keypoints) < 20:
                fail_count += 1
                continue

            # 使用前 20 个关键点
            keypoints = keypoints[:20]

            img_vis = draw_keypoints_and_skeleton(img, keypoints)

            output_file = output_dir / (img_file.stem + "_vis.jpg")
            cv2.imwrite(str(output_file), img_vis)
            success_count += 1

        except Exception:
            fail_count += 1
            continue

    print("\nVisualization complete!")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
