#!/usr/bin/env python3
"""诊断 AnimalPose 标签格式."""

from pathlib import Path

import cv2

# 检查一个标签文件
label_file = Path("/home/ng204/GH/ultralytics+rtmo/datasets/animalpose_yolo/labels/test/ca100.txt")
image_file = Path("/home/ng204/GH/ultralytics+rtmo/datasets/animalpose_yolo/images/test/ca100.jpeg")

# 读取标签
with open(label_file) as f:
    line = f.readline().strip()
    parts = line.split()

print(f"Label file: {label_file.name}")
print(f"Total values: {len(parts)}")
print(f"Raw values: {parts[:20]}...")  # 打印前20个值

# 读取图片
img = cv2.imread(str(image_file))
h, w = img.shape[:2]
print(f"\nImage dimensions: {w}x{h}")

# 分析格式
print("\n--- Format Analysis ---")
print("If format is [class, kpt0_x, kpt0_y, kpt0_v, ...] with 20 kpts:")
print("  Expected values: 1 + 20*3 = 61")
print(f"  Actual values: {len(parts)}")

print("\nIf format is [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, ...] with 20 kpts:")
print("  Expected values: 1 + 4 + 20*3 = 65")
print(f"  Actual values: {len(parts)}")

print("\nIf format is [class, kpt0_vis, kpt0_x, kpt0_y, ...] with 20 kpts:")
print("  Expected values: 1 + 20*3 = 61")
print(f"  Actual values: {len(parts)}")

# 尝试不同的解析方式
print("\n--- Trying different parsing strategies ---")

# 策略1: [class, kpt0_x, kpt0_y, kpt0_v, ...]
print("\nStrategy 1: [class, kpt0_x, kpt0_y, kpt0_v, ...]")
print(f"Class: {parts[0]}")
print("First 5 keypoints:")
for i in range(5):
    idx = 1 + i * 3
    if idx + 2 < len(parts):
        x = float(parts[idx]) * w
        y = float(parts[idx + 1]) * h
        v = float(parts[idx + 2])
        print(f"  kpt{i}: x={x:.1f}, y={y:.1f}, v={v:.3f}")

# 策略2: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, ...]
print("\nStrategy 2: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_x, kpt0_y, kpt0_v, ...]")
print(f"Class: {parts[0]}")
print(f"BBox: x={parts[1]}, y={parts[2]}, w={parts[3]}, h={parts[4]}")
print("First 5 keypoints:")
for i in range(5):
    idx = 5 + i * 3
    if idx + 2 < len(parts):
        x = float(parts[idx]) * w
        y = float(parts[idx + 1]) * h
        v = float(parts[idx + 2])
        print(f"  kpt{i}: x={x:.1f}, y={y:.1f}, v={v:.3f}")

# 策略3: [class, kpt0_vis, kpt0_x, kpt0_y, ...]
print("\nStrategy 3: [class, kpt0_vis, kpt0_x, kpt0_y, ...]")
print(f"Class: {parts[0]}")
print("First 5 keypoints:")
for i in range(5):
    idx = 1 + i * 3
    if idx + 2 < len(parts):
        v = float(parts[idx])
        x = float(parts[idx + 1]) * w
        y = float(parts[idx + 2]) * h
        print(f"  kpt{i}: v={v:.3f}, x={x:.1f}, y={y:.1f}")

# 策略4: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_vis, kpt0_x, kpt0_y, ...]
print("\nStrategy 4: [class, bbox_x, bbox_y, bbox_w, bbox_h, kpt0_vis, kpt0_x, kpt0_y, ...]")
print(f"Class: {parts[0]}")
print(f"BBox: x={parts[1]}, y={parts[2]}, w={parts[3]}, h={parts[4]}")
print("First 5 keypoints:")
for i in range(5):
    idx = 5 + i * 3
    if idx + 2 < len(parts):
        v = float(parts[idx])
        x = float(parts[idx + 1]) * w
        y = float(parts[idx + 2]) * h
        print(f"  kpt{i}: v={v:.3f}, x={x:.1f}, y={y:.1f}")
