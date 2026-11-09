#!/usr/bin/env python3
"""
可视化 AnimalPose YOLO 数据集中的标注关键点和骨骼连接
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
DATASET_ROOT = '/home/ng204/GH/ultralytics+rtmo/datasets/animalpose_yolo'
TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, 'images/test')
TEST_LABELS_DIR = os.path.join(DATASET_ROOT, 'labels/test')
OUTPUT_DIR = '/home/ng204/GH/ultralytics+rtmo/runs/keshihua/animalpose'

# 17个关键点名称（标准 AnimalPose）
KPT_NAMES = [
    'nose',        # 0
    'left_eye',    # 1
    'right_eye',   # 2
    'left_ear',    # 3
    'right_ear',   # 4
    'neck',        # 5
    'left_shoulder',   # 6
    'right_shoulder',  # 7
    'left_elbow',      # 8
    'right_elbow',     # 9
    'left_wrist',      # 10
    'right_wrist',     # 11
    'left_hip',        # 12
    'right_hip',       # 13
    'left_knee',       # 14
    'right_knee',      # 15
    'left_ankle',      # 16
    'right_ankle',     # 17
]

# 骨骼连接（标准 AnimalPose）
SKELETON_PAIRS = [
    (0, 1),   # nose - left_eye
    (0, 2),   # nose - right_eye
    (1, 3),   # left_eye - left_ear
    (2, 4),   # right_eye - right_ear
    (0, 5),   # nose - neck
    (5, 6),   # neck - left_shoulder
    (5, 7),   # neck - right_shoulder
    (6, 8),   # left_shoulder - left_elbow
    (7, 9),   # right_shoulder - right_elbow
    (8, 10),  # left_elbow - left_wrist
    (9, 11),  # right_elbow - right_wrist
    (5, 12),  # neck - left_hip
    (5, 13),  # neck - right_hip
    (12, 14), # left_hip - left_knee
    (13, 15), # right_hip - right_knee
    (14, 16), # left_knee - left_ankle
    (15, 17), # right_knee - right_ankle
]

# 关键点颜色
KEYPOINT_COLORS = [
    (0, 255, 0),      # 0 - green
    (255, 0, 0),      # 1 - blue
    (0, 0, 255),      # 2 - red
    (255, 255, 0),    # 3 - cyan
    (0, 255, 255),    # 4 - yellow
    (128, 0, 128),    # 5 - purple
    (255, 0, 255),    # 6 - magenta
    (0, 128, 255),    # 7 - orange
    (128, 128, 0),    # 8 - dark cyan
    (128, 0, 0),      # 9 - dark blue
    (0, 128, 128),    # 10 - dark yellow
    (255, 128, 0),    # 11 - light blue
    (128, 255, 0),    # 12 - light green
    (0, 255, 128),    # 13 - light cyan
    (255, 0, 128),    # 14 - light magenta
    (128, 128, 255),  # 15 - light red
    (255, 128, 128),  # 16 - light blue
    (128, 255, 128),  # 17 - light green
]

SKELETON_COLOR = (0, 255, 0)  # green


def draw_pose(img, keypoints, conf_threshold=0.5):
    """
    在图像上绘制关键点和骨骼

    Args:
        img: 输入图像
        keypoints: 关键点数组，shape为(num_keypoints, 3)，其中3个值为(x, y, confidence)
        conf_threshold: 置信度阈值

    Returns:
        绘制了关键点和骨骼的图像
    """
    h, w = img.shape[:2]

    # 首先绘制骨骼连接
    for start_idx, end_idx in SKELETON_PAIRS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_kpt = keypoints[start_idx]
            end_kpt = keypoints[end_idx]

            # 检查置信度
            if start_kpt[2] > conf_threshold and end_kpt[2] > conf_threshold:
                start_pos = (int(start_kpt[0]), int(start_kpt[1]))
                end_pos = (int(end_kpt[0]), int(end_kpt[1]))

                # 确保坐标在图像范围内
                if (0 <= start_pos[0] < w and 0 <= start_pos[1] < h and
                    0 <= end_pos[0] < w and 0 <= end_pos[1] < h):
                    cv2.line(img, start_pos, end_pos, SKELETON_COLOR, 2)

    # 然后绘制关键点
    for idx, kpt in enumerate(keypoints):
        if kpt[2] > conf_threshold:  # 只绘制置信度足够高的关键点
            x, y, conf = int(kpt[0]), int(kpt[1]), kpt[2]

            # 确保坐标在图像范围内
            if 0 <= x < w and 0 <= y < h:
                color = KEYPOINT_COLORS[idx % len(KEYPOINT_COLORS)]
                # 绘制圆形关键点
                cv2.circle(img, (x, y), 5, color, -1)
                # 绘制外圆圈
                cv2.circle(img, (x, y), 6, (255, 255, 255), 1)

    return img


def parse_yolo_labels(label_path, img_width, img_height):
    """
    解析 YOLO 格式的标注文件

    Args:
        label_path: 标注文件路径
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        关键点数组，shape为(num_keypoints, 3)
    """
    keypoints = []

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:  # class_id, x, y, confidence
                class_id = int(parts[0])
                x = float(parts[1]) * img_width
                y = float(parts[2]) * img_height
                conf = float(parts[3])
                keypoints.append([x, y, conf])

    except Exception as e:
        logger.warning(f"无法解析标注文件 {label_path}: {e}")

    return np.array(keypoints) if keypoints else np.zeros((17, 3))


def main():
    """主函数"""
    logger.info("开始可视化 AnimalPose YOLO 数据集标注")
    logger.info(f"测试图像目录: {TEST_IMAGES_DIR}")
    logger.info(f"测试标注目录: {TEST_LABELS_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")

    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 检查目录是否存在
    if not Path(TEST_IMAGES_DIR).exists():
        logger.error(f"测试图像目录不存在: {TEST_IMAGES_DIR}")
        return

    if not Path(TEST_LABELS_DIR).exists():
        logger.error(f"测试标注目录不存在: {TEST_LABELS_DIR}")
        return

    # 获取所有测试图像
    test_images = sorted(Path(TEST_IMAGES_DIR).glob('*.jpg'))
    if not test_images:
        test_images = sorted(Path(TEST_IMAGES_DIR).glob('*.png'))

    logger.info(f"找到 {len(test_images)} 张测试图像")

    if len(test_images) == 0:
        logger.warning(f"没有找到测试图像在 {TEST_IMAGES_DIR}")
        return

    # 处理每张图像
    for idx, img_path in enumerate(test_images):
        try:
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"无法读取图像: {img_path}")
                continue

            h, w = img.shape[:2]

            # 查找对应的标注文件
            label_path = Path(TEST_LABELS_DIR) / (img_path.stem + '.txt')

            if label_path.exists():
                # 解析标注
                keypoints = parse_yolo_labels(str(label_path), w, h)

                if len(keypoints) > 0:
                    # 绘制关键点和骨骼
                    img = draw_pose(img, keypoints)

            # 保存结果
            output_file = Path(OUTPUT_DIR) / img_path.name
            cv2.imwrite(str(output_file), img)

            if (idx + 1) % 10 == 0:
                logger.info(f"已处理 {idx + 1}/{len(test_images)} 张图像")

        except Exception as e:
            logger.error(f"处理图像 {img_path} 时出错: {e}")
            continue

    logger.info(f"完成 AnimalPose YOLO 数据集的可视化，结果保存至 {OUTPUT_DIR}")
    logger.info("=" * 60)
    logger.info("所有数据集的可视化已完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
