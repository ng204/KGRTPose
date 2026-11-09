#!/usr/bin/env python3
"""
检查鱼类位姿数据标注质量的脚本
"""
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def check_annotation_quality():
    dataset_root = Path('datasets/animalpose')

    print("="*80)
    print("🔍 鱼类位姿数据集标注质量检查")
    print("="*80)

    # 检查目录结构
    print("\n📁 目录结构：")
    for split in ['train', 'val', 'test']:
        img_dir = dataset_root / 'images' / split
        lab_dir = dataset_root / 'labels' / split
        if img_dir.exists():
            img_count = len(list(img_dir.glob('*.*')))
            lab_count = len(list(lab_dir.glob('*.txt'))) if lab_dir.exists() else 0
            print(f"  {split:6s}: {img_count:4d} images, {lab_count:4d} labels")

    # 详细检查每个split
    for split in ['train', 'val', 'test']:
        lab_dir = dataset_root / 'labels' / split
        if not lab_dir.exists():
            continue

        print(f"\n{'='*80}")
        print(f"📊 检查 {split.upper()} 集合标注质量")
        print(f"{'='*80}")

        label_files = sorted(lab_dir.glob('*.txt'))
        print(f"总标注文件数: {len(label_files)}")

        if not label_files:
            print("⚠️  无标注文件")
            continue

        # 统计变量
        stats = {
            'total_annotations': 0,
            'total_keypoints': 0,
            'visible_keypoints': 0,
            'invisible_keypoints': 0,
            'out_of_bounds': [],
            'coordinate_ranges': defaultdict(list),
            'visibility_per_kpt': [0] * 6,  # 6个关键点
            'empty_files': 0,
            'format_errors': 0,
            'bbox_issues': [],
            'coordinate_anomalies': []
        }

        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                stats['empty_files'] += 1
                continue

            for line_idx, line in enumerate(lines):
                try:
                    parts = line.strip().split()

                    # 格式: class_id x1 y1 x2 y2 kpt0_x kpt0_y kpt0_vis ... kpt5_x kpt5_y kpt5_vis
                    # 总共: 1 + 4 + 6*3 = 23个值
                    if len(parts) < 23:
                        stats['format_errors'] += 1
                        stats['coordinate_anomalies'].append(
                            f"{label_file.name}:{line_idx+1} - 字段数不足 ({len(parts)}/23)"
                        )
                        continue

                    stats['total_annotations'] += 1

                    # 解析bbox和kpts
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    kpts_raw = [float(x) for x in parts[5:23]]

                    # 检查bbox范围
                    x1, y1, x2, y2 = bbox
                    if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                        stats['bbox_issues'].append(
                            f"{label_file.name}: bbox超出范围 [{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}]"
                        )

                    # 检查6个关键点 (x, y, visibility)
                    for kpt_idx in range(6):
                        kpt_x = kpts_raw[kpt_idx * 3]
                        kpt_y = kpts_raw[kpt_idx * 3 + 1]
                        kpt_vis = int(kpts_raw[kpt_idx * 3 + 2])

                        stats['total_keypoints'] += 1

                        # 记录可见性
                        if kpt_vis > 0:
                            stats['visible_keypoints'] += 1
                            stats['visibility_per_kpt'][kpt_idx] += 1
                        else:
                            stats['invisible_keypoints'] += 1

                        # 记录坐标范围
                        stats['coordinate_ranges'][kpt_idx].append((kpt_x, kpt_y))

                        # 检查坐标有效性
                        if kpt_vis > 0:  # 仅检查可见的关键点
                            if kpt_x < 0 or kpt_x > 1 or kpt_y < 0 or kpt_y > 1:
                                stats['out_of_bounds'].append(
                                    f"{label_file.name}:kpt{kpt_idx} - ({kpt_x:.4f}, {kpt_y:.4f})"
                                )
                            # 检查是否过于接近边界 (>0.95 or <0.05的可能是问题)
                            if kpt_x > 1.01 or kpt_y > 1.01:
                                stats['coordinate_anomalies'].append(
                                    f"{label_file.name}:kpt{kpt_idx} - 超出(>1.01): ({kpt_x:.4f}, {kpt_y:.4f})"
                                )
                except (ValueError, IndexError) as e:
                    stats['format_errors'] += 1
                    stats['coordinate_anomalies'].append(
                        f"{label_file.name}:{line_idx+1} - 解析错误: {str(e)}"
                    )

        # 打印统计结果
        print(f"\n✅ 基本统计：")
        print(f"   标注总数: {stats['total_annotations']}")
        print(f"   关键点总数: {stats['total_keypoints']}")
        print(f"   空文件: {stats['empty_files']}")
        print(f"   格式错误: {stats['format_errors']}")

        print(f"\n📍 关键点可见性分布：")
        kpt_names = ['mouth', 'eye_left', 'eye_right', 'dorsal_mid', 'caudal_base', 'caudal_tip']
        for kpt_idx, name in enumerate(kpt_names):
            vis = stats['visibility_per_kpt'][kpt_idx]
            vis_rate = (vis / stats['total_keypoints'] * 100) if stats['total_keypoints'] > 0 else 0
            print(f"   kpt{kpt_idx} ({name:12s}): {vis:6d}/{stats['total_keypoints']:6d} ({vis_rate:5.1f}%)")

        vis_rate = (stats['visible_keypoints'] / stats['total_keypoints'] * 100) if stats['total_keypoints'] > 0 else 0
        print(f"   {'总体可见率':18s}: {stats['visible_keypoints']:6d}/{stats['total_keypoints']:6d} ({vis_rate:5.1f}%)")

        # 坐标范围分析
        print(f"\n📐 坐标范围分析：")
        for kpt_idx, name in enumerate(kpt_names):
            if stats['coordinate_ranges'][kpt_idx]:
                coords = np.array(stats['coordinate_ranges'][kpt_idx])
                x_vals = coords[:, 0]
                y_vals = coords[:, 1]
                # 过滤掉(0,0)的情况（标注错误）
                x_vals_valid = x_vals[~((x_vals == 0) & (y_vals == 0))]
                y_vals_valid = y_vals[~((x_vals == 0) & (y_vals == 0))]

                if len(x_vals_valid) > 0:
                    print(f"   kpt{kpt_idx} ({name:12s}): X=[{x_vals_valid.min():.3f}, {x_vals_valid.max():.3f}], Y=[{y_vals_valid.min():.3f}, {y_vals_valid.max():.3f}]")

        # 异常情况汇总
        print(f"\n⚠️  异常检测：")

        if stats['bbox_issues']:
            print(f"\n   ❌ BBox超出范围 ({len(stats['bbox_issues'])} 条):")
            for issue in stats['bbox_issues'][:5]:  # 只显示前5个
                print(f"      {issue}")
            if len(stats['bbox_issues']) > 5:
                print(f"      ... 还有 {len(stats['bbox_issues'])-5} 条")

        if stats['out_of_bounds']:
            print(f"\n   ❌ 坐标超出范围 ({len(stats['out_of_bounds'])} 条):")
            for issue in stats['out_of_bounds'][:5]:
                print(f"      {issue}")
            if len(stats['out_of_bounds']) > 5:
                print(f"      ... 还有 {len(stats['out_of_bounds'])-5} 条")

        if stats['coordinate_anomalies']:
            print(f"\n   ⚠️  异常坐标值 ({len(stats['coordinate_anomalies'])} 条):")
            for issue in stats['coordinate_anomalies'][:5]:
                print(f"      {issue}")
            if len(stats['coordinate_anomalies']) > 5:
                print(f"      ... 还有 {len(stats['coordinate_anomalies'])-5} 条")

        if not stats['bbox_issues'] and not stats['out_of_bounds'] and not stats['coordinate_anomalies']:
            print("   ✅ 未发现坐标异常")

    # 最后的建议
    print(f"\n{'='*80}")
    print("💡 建议与总结")
    print(f"{'='*80}")
    print("""
✅ 如果上述检查都通过：
   1. 数据标注质量良好，可以直接进行训练优化
   2. 建议按优化方案继续训练到 1000-1200 epoch

❌ 如果发现坐标异常或格式错误：
   1. 需要先修复这些标注文件
   2. 使用坐标映射/验证脚本进行修复
   3. 重新训练模型

📌 关键点可见性说明：
   - 如果某个关键点可见率<70%，可能影响性能
   - 可以考虑加权训练或数据增强来补偿
    """)

if __name__ == '__main__':
    check_annotation_quality()
