#!/usr/bin/env python3
"""
彻底修复test集标注文件中超出范围的坐标
强制将所有 > 1.0 的坐标裁剪到 1.0
"""
from pathlib import Path

test_dir = Path('/home/ng204/GH/ultralytics+rtmo/datasets/animalpose/labels/test')

if not test_dir.exists():
    print(f"❌ Test目录不存在: {test_dir}")
    exit(1)

fixed_count = 0
total_coords_fixed = 0

for label_file in sorted(test_dir.glob('*.txt')):
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        file_modified = False

        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue

            parts = line.strip().split()
            fixed_parts = []

            for part in parts:
                try:
                    val = float(part)
                    # 修复：任何 > 1.0 的值都裁剪到 1.0
                    if val > 1.0:
                        val = 1.0
                        total_coords_fixed += 1
                        file_modified = True
                    # 任何 < 0.0 的值都设为 0.0
                    elif val < 0.0:
                        val = 0.0
                        total_coords_fixed += 1
                        file_modified = True

                    # 保持原始格式（整数或浮点）
                    if '.' in part:
                        fixed_parts.append(f"{val:.6f}")
                    else:
                        fixed_parts.append(str(int(val)))
                except ValueError:
                    # 不是数字就保持原样
                    fixed_parts.append(part)

            new_lines.append(' '.join(fixed_parts) + '\n')

        # 写回文件
        if file_modified:
            with open(label_file, 'w') as f:
                f.writelines(new_lines)
            fixed_count += 1
            print(f"✓ 已修复: {label_file.name}")

    except Exception as e:
        print(f"❌ 错误 {label_file.name}: {e}")

print(f"\n{'='*60}")
print(f"✅ 修复完成!")
print(f"{'='*60}")
print(f"修复的文件数: {fixed_count}/142")
print(f"修复的坐标总数: {total_coords_fixed}")
print(f"{'='*60}")

# 清除缓存
import os
cache_file = Path('/home/ng204/GH/ultralytics+rtmo/datasets/animalpose/labels/test.cache')
if cache_file.exists():
    os.remove(cache_file)
    print(f"🗑️  已删除缓存: test.cache")
