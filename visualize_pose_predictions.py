#!/usr/bin/env python3
"""
可视化多个姿态估计模型在测试集上的预测结果
使用模型检测出的关键点和骨骼连接进行可视化
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
MODELS = {
    'fish3': '/home/ng204/GH/ultralytics+rtmo/runs/pose/fish3/weights/best.pt',
    'fish3l': '/home/ng204/GH/ultralytics+rtmo/runs/pose/fish3l/weights/best.pt',
    'fish3m': '/home/ng204/GH/ultralytics+rtmo/runs/pose/fish3m/weights/best.pt',
    'fish3n': '/home/ng204/GH/ultralytics+rtmo/runs/pose/fish3n/weights/best.pt',
    'fish3s': '/home/ng204/GH/ultralytics+rtmo/runs/pose/fish3s/weights/best.pt',
}

TEST_IMAGES_DIR = '/home/ng204/GH/ultralytics+rtmo/datasets/animalpose/images/test'
OUTPUT_BASE_DIR = '/home/ng204/GH/ultralytics+rtmo/runs/keshihua/fish'

# 鱼类关键点连接关系（骨骼）- 根据 animalpose.yaml 定义
# 0: mouth, 1: eye_left, 2: eye_right, 3: dorsal_mid, 4: caudal_base, 5: caudal_tip
SKELETON_PAIRS = [
    (0, 1),  # mouth ↔ eye_left
    (0, 2),  # mouth ↔ eye_right
    (1, 2),  # eye_left ↔ eye_right (构成三角)
    (0, 3),  # mouth ↔ dorsal_mid
    (3, 4),  # dorsal_mid ↔ caudal_base
    (4, 5),  # caudal_base ↔ caudal_tip
]

# 关键点颜色 (BGR格式)
KEYPOINT_COLORS = [
    (0, 255, 0),      # mouth - green
    (255, 0, 0),      # eye_left - blue
    (0, 0, 255),      # eye_right - red
    (255, 255, 0),    # dorsal_mid - cyan
    (0, 255, 255),    # caudal_base - yellow
    (128, 0, 128),    # caudal_tip - purple
]

# 骨骼颜色
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


def visualize_model(model_name, model_path, output_dir):
    """
    使用指定模型对测试集进行可视化

    Args:
        model_name: 模型名称
        model_path: 模型权重路径
        output_dir: 输出目录
    """
    logger.info(f"加载模型: {model_name} from {model_path}")

    # 加载模型
    model = YOLO(model_path)

    # 获取所有测试图像
    test_images = sorted(Path(TEST_IMAGES_DIR).glob('*.png'))
    logger.info(f"找到 {len(test_images)} 张测试图像")

    if len(test_images) == 0:
        logger.warning(f"没有找到测试图像在 {TEST_IMAGES_DIR}")
        return

    # 创建输出目录
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_path}")

    # 处理每张图像
    for idx, img_path in enumerate(test_images):
        try:
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"无法读取图像: {img_path}")
                continue

            # 运行推理
            results = model(img, verbose=False)

            if results and len(results) > 0:
                result = results[0]

                # 获取关键点
                if result.keypoints is not None and len(result.keypoints) > 0:
                    for keypoints in result.keypoints:
                        kpts = keypoints.xy[0].cpu().numpy() if hasattr(keypoints.xy[0], 'cpu') else keypoints.xy[0]

                        # 如果有置信度信息，获取它
                        if hasattr(keypoints, 'conf') and keypoints.conf is not None:
                            conf = keypoints.conf[0].cpu().numpy() if hasattr(keypoints.conf[0], 'cpu') else keypoints.conf[0]
                            # 组合坐标和置信度
                            kpts_with_conf = np.hstack([kpts, conf.reshape(-1, 1)])
                        else:
                            # 如果没有置信度，使用全1
                            kpts_with_conf = np.hstack([kpts, np.ones((len(kpts), 1))])

                        # 绘制关键点和骨骼
                        img = draw_pose(img, kpts_with_conf)

            # 保存结果
            output_file = output_path / img_path.name
            cv2.imwrite(str(output_file), img)

            if (idx + 1) % 10 == 0:
                logger.info(f"已处理 {idx + 1}/{len(test_images)} 张图像")

        except Exception as e:
            logger.error(f"处理图像 {img_path} 时出错: {e}")
            continue

    logger.info(f"完成 {model_name} 的可视化，结果保存至 {output_path}")


def main():
    """主函数"""
    logger.info("开始可视化多个姿态估计模型")
    logger.info(f"测试图像目录: {TEST_IMAGES_DIR}")
    logger.info(f"输出基目录: {OUTPUT_BASE_DIR}")

    # 检查测试图像目录是否存在
    if not Path(TEST_IMAGES_DIR).exists():
        logger.error(f"测试图像目录不存在: {TEST_IMAGES_DIR}")
        return

    # 处理每个模型
    for model_name, model_path in MODELS.items():
        if not Path(model_path).exists():
            logger.error(f"模型权重文件不存在: {model_path}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"正在处理模型: {model_name}")
        logger.info(f"{'='*60}")

        visualize_model(model_name, model_path, OUTPUT_BASE_DIR)

    logger.info(f"\n{'='*60}")
    logger.info("所有模型的可视化已完成！")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
