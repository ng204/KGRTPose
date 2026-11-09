#!/usr/bin/env python3
"""
验证 AnimalPose 标签解析格式
"""

import cv2
import numpy as np
from pathlib import Path

def parse_and_visualize_sample():
    """解析一个样本标签并可视化"""

    images_dir = Path('/home/ng204/GH/ultralytics+rtmo/datasets/animalpose_yolo/images/test')
    labels_dir = Path('/home/ng204/GH/ultralytics+rtmo/datasets/animalpose_yolo/labels/test')

    # 获取前3张图片进行验证
    image_files = sorted(list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.jpg')))[:3]

    for img_file in image_files:
        print(f"\n{'='*60}")
        print(f"Image: {img_file.name}")

        # 读取图片
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]
        print(f"Image dimensions: {w}x{h}")

        # 读取标签
        label_file = labels_dir / (img_file.stem + '.txt')
        with open(label_file, 'r') as f:
            parts = f.read().strip().split()

        print(f"Total values in label: {len(parts)}")
        print(f"First 10 values: {parts[:10]}")

        # 尝试不同的解析格式
        class_id = int(parts[0])
        print(f"\nClass ID: {class_id}")

        # 格式: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, ...]
        if len(parts) == 65:  # 1 + 4 + 60
            print("\n✓ Format matches: [class, bbox, kpt0_x, kpt0_y, kpt0_v, ...]")

            bbox = [float(parts[i]) for i in range(1, 5)]
            print(f"BBox (normalized): x={bbox[0]:.3f}, y={bbox[1]:.3f}, w={bbox[2]:.3f}, h={bbox[3]:.3f}")
            print(f"BBox (pixels): x={bbox[0]*w:.1f}, y={bbox[1]*h:.1f}, w={bbox[2]*w:.1f}, h={bbox[3]*h:.1f}")

            # 解析前5个关键点
            print("\nFirst 5 keypoints:")
            keypoints = []
            for i in range(20):
                idx = 5 + i * 3  # 跳过 class(1) + bbox(4)
                if idx + 2 < len(parts):
                    x = float(parts[idx]) * w
                    y = float(parts[idx + 1]) * h
                    v = float(parts[idx + 2])
                    keypoints.append((x, y, v))
                    if i < 5:
                        print(f"  kpt{i}: x={x:.1f}, y={y:.1f}, v={v:.2f} {'✓ valid' if (0 <= x <= w and 0 <= y <= h and v >= 0) else '✗ INVALID'}")

            # 在图片上绘制前5个关键点
            img_vis = img.copy()
            for idx, (x, y, v) in enumerate(keypoints[:5]):
                if v > 0:
                    cv2.circle(img_vis, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(img_vis, str(idx), (int(x)+5, int(y)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 保存验证图片
            output_file = Path(f'/tmp/validate_{img_file.stem}.jpg')
            cv2.imwrite(str(output_file), img_vis)
            print(f"\nVisualization saved to: {output_file}")

if __name__ == '__main__':
    parse_and_visualize_sample()
