#!/usr/bin/env python3
"""修复test集中超出范围的坐标."""

from pathlib import Path


def fix_coords_in_file(label_path):
    """修复单个标签文件中的坐标."""
    try:
        with open(label_path) as f:
            lines = f.readlines()

        modified = False
        fixed_lines = []

        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                fixed_lines.append(line)
                continue

            # 尝试修复所有数值字段
            fixed_parts = []
            for i, part in enumerate(parts):
                try:
                    val = float(part)
                    if val > 1.0:
                        val = 1.0
                        modified = True
                    elif val < 0.0:
                        val = 0.0
                        modified = True
                    fixed_parts.append(f"{val:.6f}")
                except ValueError:
                    fixed_parts.append(part)

            fixed_lines.append("\t".join(fixed_parts) + "\n")

        if modified:
            with open(label_path, "w") as f:
                f.writelines(fixed_lines)
            return True
        return False

    except Exception as e:
        print(f"❌ 错误处理 {label_path}: {e}")
        return False


# 处理test集
test_dir = Path("/home/ng204/GH/ultralytics+rtmo/datasets/animalpose/labels/test")
if test_dir.exists():
    print("🔧 修复test集坐标...")
    fixed_count = 0
    for label_file in sorted(test_dir.glob("*.txt")):
        if fix_coords_in_file(label_file):
            fixed_count += 1
            print(f"  ✓ 修复: {label_file.name}")

    print(f"\n✅ 已修复 {fixed_count} 个test集文件")

    # 清除test集缓存
    cache_file = test_dir.parent / "test.cache"
    if cache_file.exists():
        cache_file.unlink()
        print("🗑️  已删除缓存: test.cache")
