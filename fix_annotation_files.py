#!/usr/bin/env python3
"""
自动修复标注文件的脚本
1. 将 visibility 字段转换为整数 (>0.5 -> 1, <=0.5 -> 0)
2. 裁剪超出范围的坐标到 [0, 1]
3. 修复颠倒的 bbox.
"""

from pathlib import Path


def fix_annotation_files():
    dataset_root = Path("datasets/animalpose")

    print("=" * 80)
    print("🔧 开始修复标注文件")
    print("=" * 80)

    stats = {
        "files_processed": 0,
        "annotations_fixed": 0,
        "visibility_fixed": 0,
        "coords_clipped": 0,
        "bbox_fixed": 0,
    }

    for split in ["train", "val", "test"]:
        lab_dir = dataset_root / "labels" / split
        if not lab_dir.exists():
            continue

        print(f"\n🔍 处理 {split.upper()} 集合...")

        label_files = sorted(lab_dir.glob("*.txt"))

        for label_file in label_files:
            try:
                with open(label_file) as f:
                    lines = f.readlines()

                if not lines:
                    continue

                fixed_lines = []
                file_modified = False

                for line in lines:
                    try:
                        parts = line.strip().split()

                        if len(parts) < 23:
                            # 格式太短，跳过
                            fixed_lines.append(line)
                            continue

                        # 解析
                        class_id = parts[0]
                        bbox = [float(x) for x in parts[1:5]]
                        kpts_raw = [float(x) for x in parts[5:23]]

                        # 修复 bbox (检查是否颠倒)
                        x1, y1, x2, y2 = bbox
                        if x1 > x2:
                            x1, x2 = x2, x1
                            stats["bbox_fixed"] += 1
                            file_modified = True
                        if y1 > y2:
                            y1, y2 = y2, y1
                            stats["bbox_fixed"] += 1
                            file_modified = True

                        # 裁剪 bbox 到 [0, 1]
                        x1 = max(0, min(1, x1))
                        y1 = max(0, min(1, y1))
                        x2 = max(0, min(1, x2))
                        y2 = max(0, min(1, y2))

                        bbox_str = f"{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}"

                        # 修复关键点
                        kpts_fixed = []
                        for kpt_idx in range(6):
                            kpt_x = kpts_raw[kpt_idx * 3]
                            kpt_y = kpts_raw[kpt_idx * 3 + 1]
                            kpt_vis_raw = kpts_raw[kpt_idx * 3 + 2]

                            # 将 visibility 转换为整数 (>0.5 -> 1, <=0.5 -> 0)
                            if kpt_vis_raw > 0.5:
                                kpt_vis = 1
                            else:
                                kpt_vis = 0

                            # 检查是否修改了 visibility
                            if kpt_vis_raw != kpt_vis:
                                stats["visibility_fixed"] += 1
                                file_modified = True

                            # 裁剪坐标
                            kpt_x_orig = kpt_x
                            kpt_y_orig = kpt_y
                            kpt_x = max(0, min(1, kpt_x))
                            kpt_y = max(0, min(1, kpt_y))

                            if kpt_x != kpt_x_orig or kpt_y != kpt_y_orig:
                                stats["coords_clipped"] += 1
                                file_modified = True

                            kpts_fixed.append(f"{kpt_x:.6f} {kpt_y:.6f} {kpt_vis}")

                        # 重构行
                        fixed_line = f"{class_id} {bbox_str} {' '.join(kpts_fixed)}\n"
                        fixed_lines.append(fixed_line)

                        if file_modified:
                            stats["annotations_fixed"] += 1
                            file_modified = False

                    except Exception:
                        # 出错的行保持原样
                        fixed_lines.append(line)

                # 写回文件
                with open(label_file, "w") as f:
                    f.writelines(fixed_lines)

                stats["files_processed"] += 1

            except Exception as e:
                print(f"  ⚠️  处理 {label_file.name} 失败: {e}")

    # 打印统计
    print(f"\n{'=' * 80}")
    print("✅ 修复完成！")
    print(f"{'=' * 80}")
    print(f"处理的文件数: {stats['files_processed']}")
    print(f"修复的标注数: {stats['annotations_fixed']}")
    print(f"修复的 visibility 字段: {stats['visibility_fixed']}")
    print(f"裁剪的坐标: {stats['coords_clipped']}")
    print(f"修复的 bbox: {stats['bbox_fixed']}")

    print("\n💾 修复后的文件已保存到原位置")
    print("建议：重新运行 check_annotation_quality.py 验证修复结果\n")


if __name__ == "__main__":
    fix_annotation_files()
