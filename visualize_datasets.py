#!/usr/bin/env python3
"""
可视化 animalpose_yolo 和 atrw 数据集
将图片和关键点标签结合在一起
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
from collections import defaultdict

# 配置
ANIMALPOSE_ROOT = "/home/ng204/GH/ultralytics+rtmo/datasets/animalpose_yolo"
ATRW_ROOT = "/home/ng204/GH/ultralytics+rtmo/datasets/atrw"
OUTPUT_ROOT = "/home/ng204/GH/ultralytics+rtmo/runs/keshihua/2"
IMAGES_PER_DATASET = 30

# 创建输出目录
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/animalpose_yolo", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/atrw", exist_ok=True)


def get_keypoint_colors(num_kpts):
    """生成关键点颜色"""
    colors = []
    for i in range(num_kpts):
        hue = int(180 * i / num_kpts)
        colors.append(cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
    return colors


def parse_yolo_label(label_path, img_h, img_w):
    """
    解析 YOLO 格式标签
    格式: class_id x_center y_center w h kpt1_x kpt1_y kpt1_vis kpt2_x ...
    """
    keypoints = []
    try:
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                values = list(map(float, line.split()))
                # 跳过 class_id, bbox (4个值)
                # 从第5个值开始是关键点 (每个3个值: x, y, vis)
                kpt_data = values[5:]  # 跳过 class_id 和 bbox (1+4=5)

                for i in range(0, len(kpt_data), 3):
                    if i + 2 < len(kpt_data):
                        x_norm = kpt_data[i]
                        y_norm = kpt_data[i + 1]
                        vis = kpt_data[i + 2]

                        # 反归一化到像素坐标
                        x_pixel = int(x_norm * img_w)
                        y_pixel = int(y_norm * img_h)

                        keypoints.append({
                            'x': x_pixel,
                            'y': y_pixel,
                            'vis': vis,
                            'idx': i // 3
                        })
    except Exception as e:
        print(f"Error parsing {label_path}: {e}")

    return keypoints


def draw_keypoints(img, keypoints, colors, radius=5, thickness=2):
    """在图片上绘制关键点"""
    img_copy = img.copy()

    for kpt in keypoints:
        x, y = kpt['x'], kpt['y']
        vis = kpt['vis']
        idx = kpt['idx']

        # 只绘制 visible 的关键点
        if vis > 0.5:
            color = tuple(map(int, colors[idx % len(colors)]))
            # 绘制圆点
            cv2.circle(img_copy, (x, y), radius, color, -1)
            # 绘制边框
            cv2.circle(img_copy, (x, y), radius, (255, 255, 255), thickness)

    return img_copy


def draw_skeleton(img, keypoints, edges=None, colors=None):
    """绘制骨骼连接线"""
    img_copy = img.copy()

    if edges is None:
        edges = []

    if colors is None:
        colors = get_keypoint_colors(len(keypoints))

    # 按索引存储关键点
    kpt_dict = {kpt['idx']: kpt for kpt in keypoints if kpt['vis'] > 0.5}

    # 绘制骨骼
    for start_idx, end_idx in edges:
        if start_idx in kpt_dict and end_idx in kpt_dict:
            start = kpt_dict[start_idx]
            end = kpt_dict[end_idx]

            color = tuple(map(int, colors[start_idx % len(colors)]))
            cv2.line(img_copy, (start['x'], start['y']), (end['x'], end['y']), color, 2)

    return img_copy


def visualize_animalpose_yolo(num_images=30):
    """可视化 animalpose_yolo 数据集"""
    print(f"\n{'='*60}")
    print(f"正在可视化 animalpose_yolo (选取 {num_images} 张图片)")
    print(f"{'='*60}")

    images_dir = f"{ANIMALPOSE_ROOT}/images/train"
    labels_dir = f"{ANIMALPOSE_ROOT}/labels/train"

    # 获取所有图片
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

    # 随机选择30张
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    # 关键点颜色 (6个关键点)
    num_kpts = 6
    colors = get_keypoint_colors(num_kpts)

    # 鱼类骨骼边（根据之前的 STGR 定义）
    edges = [(0, 1), (0, 2), (1, 2), (0, 3), (3, 4), (4, 5)]

    for idx, img_file in enumerate(selected_images, 1):
        img_path = os.path.join(images_dir, img_file)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠ 无法读取: {img_file}")
            continue

        h, w = img.shape[:2]

        # 解析标签
        keypoints = parse_yolo_label(label_path, h, w)

        if not keypoints:
            print(f"  ⚠ 无效标签: {img_file}")
            continue

        # 绘制骨骼
        img_vis = draw_skeleton(img, keypoints, edges, colors)

        # 绘制关键点
        img_vis = draw_keypoints(img_vis, keypoints, colors)

        # 添加文本信息
        info_text = f"AnimalPose YOLO | {len(keypoints)} keypoints"
        cv2.putText(img_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

        # 保存
        output_path = f"{OUTPUT_ROOT}/animalpose_yolo/{idx:03d}_{img_file}"
        cv2.imwrite(output_path, img_vis)

        print(f"  ✓ [{idx:2d}/{num_images}] {img_file} → {os.path.basename(output_path)}")

    print(f"✅ animalpose_yolo 可视化完成，共 {len(selected_images)} 张图片")


def visualize_atrw(num_images=30):
    """可视化 ATRW 数据集"""
    print(f"\n{'='*60}")
    print(f"正在可视化 ATRW (选取 {num_images} 张图片)")
    print(f"{'='*60}")

    images_dir = f"{ATRW_ROOT}/images/train"
    labels_dir = f"{ATRW_ROOT}/labels/train"

    # 获取所有图片
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

    # 随机选择30张
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    # 老虎关键点数（通常是17个，与COCO pose兼容）
    # 先检查标签格式
    num_kpts = 17
    colors = get_keypoint_colors(num_kpts)

    # COCO skeleton
    coco_edges = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    # 转换为0-indexed
    edges = [[e[0]-1, e[1]-1] for e in coco_edges]

    for idx, img_file in enumerate(selected_images, 1):
        img_path = os.path.join(images_dir, img_file)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠ 无法读取: {img_file}")
            continue

        h, w = img.shape[:2]

        # 解析标签
        keypoints = parse_yolo_label(label_path, h, w)

        if not keypoints:
            print(f"  ⚠ 无效标签: {img_file}")
            continue

        # 绘制骨骼
        img_vis = draw_skeleton(img, keypoints, edges, colors)

        # 绘制关键点
        img_vis = draw_keypoints(img_vis, keypoints, colors)

        # 添加文本信息
        info_text = f"ATRW | {len(keypoints)} keypoints"
        cv2.putText(img_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

        # 保存
        output_path = f"{OUTPUT_ROOT}/atrw/{idx:03d}_{img_file}"
        cv2.imwrite(output_path, img_vis)

        print(f"  ✓ [{idx:2d}/{num_images}] {img_file} → {os.path.basename(output_path)}")

    print(f"✅ ATRW 可视化完成，共 {len(selected_images)} 张图片")


def main():
    print(f"\n🎨 数据集可视化工具")
    print(f"输出目录: {OUTPUT_ROOT}")

    # 设置随机种子以确保可重复
    random.seed(42)

    # 可视化两个数据集
    visualize_animalpose_yolo(IMAGES_PER_DATASET)
    visualize_atrw(IMAGES_PER_DATASET)

    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"📊 可视化统计")
    print(f"{'='*60}")

    animalpose_files = len([f for f in os.listdir(f"{OUTPUT_ROOT}/animalpose_yolo") if f.endswith('.png') or f.endswith('.jpg')])
    atrw_files = len([f for f in os.listdir(f"{OUTPUT_ROOT}/atrw") if f.endswith('.png') or f.endswith('.jpg')])

    print(f"✓ AnimalPose YOLO: {animalpose_files} 张图片")
    print(f"✓ ATRW: {atrw_files} 张图片")
    print(f"✓ 总计: {animalpose_files + atrw_files} 张图片")
    print(f"✓ 保存位置: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
