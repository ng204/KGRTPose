#!/usr/bin/env python3
"""
修复缺少class_id和bbox的标注文件
格式：应该是 class_id x1 y1 x2 y2 kpt0_x kpt0_y kpt0_vis ... kpt5_x kpt5_y kpt5_vis
但0001.txt等文件缺少前5个字段
"""
import os
from pathlib import Path

def fix_missing_fields():
    dataset_root = Path('datasets/animalpose')

    print("="*80)
    print("🔧 修复缺少class_id和bbox的标注文件")
    print("="*80)

    stats = {'files_fixed': 0, 'annotations_fixed': 0}

    for split in ['train', 'val', 'test']:
        lab_dir = dataset_root / 'labels' / split
        if not lab_dir.exists():
            continue

        print(f"\n🔍 处理 {split.upper()} 集合...")

        label_files = sorted(lab_dir.glob('*.txt'))

        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                if not lines:
                    continue

                fixed_lines = []
                file_modified = False

                for line in lines:
                    parts = line.strip().split()

                    if len(parts) == 0:
                        fixed_lines.append(line)
                        continue

                    # 检查是否缺少class_id和bbox（正常应该23个字段）
                    # 缺少的话应该是 18 个字段（6 kpts * 3）
                    if len(parts) == 18:
                        # 缺少class_id和bbox字段，需要添加
                        # class_id = 0 (鱼类)
                        # 为bbox自动生成一个合理的值，基于关键点位置

                        kpts = []
                        for i in range(0, 18, 3):
                            kpts.append((float(parts[i]), float(parts[i+1])))

                        # 计算关键点的包围盒
                        xs = [kpt[0] for kpt in kpts]
                        ys = [kpt[1] for kpt in kpts]

                        x_min = min(xs)
                        x_max = max(xs)
                        y_min = min(ys)
                        y_max = max(ys)

                        # 添加一些边距
                        margin = 0.05
                        x_min = max(0, x_min - margin)
                        x_max = min(1, x_max + margin)
                        y_min = max(0, y_min - margin)
                        y_max = min(1, y_max + margin)

                        # 重构行
                        fixed_line = f"0 {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f} {' '.join(parts)}\n"
                        fixed_lines.append(fixed_line)
                        stats['annotations_fixed'] += 1
                        file_modified = True
                    elif len(parts) == 23:
                        # 已经是正确格式
                        fixed_lines.append(line)
                    else:
                        # 其他格式，保持原样
                        fixed_lines.append(line)

                # 写回文件
                if file_modified:
                    with open(label_file, 'w') as f:
                        f.writelines(fixed_lines)
                    stats['files_fixed'] += 1

            except Exception as e:
                print(f"  ⚠️  处理 {label_file.name} 失败: {e}")

    # 打印统计
    print(f"\n{'='*80}")
    print("✅ 修复完成！")
    print(f"{'='*80}")
    print(f"修复的文件数: {stats['files_fixed']}")
    print(f"修复的标注数: {stats['annotations_fixed']}")
    print(f"\n💾 修复后的文件已保存")

if __name__ == '__main__':
    fix_missing_fields()
