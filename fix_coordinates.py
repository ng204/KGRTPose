#!/usr/bin/env python3
"""
Fix out-of-bounds coordinates in RTMO pose estimation labels.
Clamps all coordinates to [0, 1] range.
"""

from pathlib import Path


def fix_label_file(label_path):
    """Read a label file, fix out-of-bounds coordinates, and save it back. Format: keypoint_id class_id x1 y1 c1 x2 y2
    c2 ...
    """
    try:
        with open(label_path) as f:
            lines = f.readlines()

        fixed_lines = []
        changed = False

        for line in lines:
            line = line.strip()
            if not line:
                fixed_lines.append(line)
                continue

            parts = line.split()
            if len(parts) < 3:
                fixed_lines.append(line)
                continue

            # First element is keypoint_id, second is class_id
            keypoint_id = parts[0]

            # Process the rest of the values (should be alternating x, y, c)
            fixed_parts = [keypoint_id]

            for i, val_str in enumerate(parts[1:]):
                try:
                    val = float(val_str)
                    # Clamp to [0, 1] range
                    if val < 0.0:
                        val = 0.0
                        changed = True
                    elif val > 1.0:
                        val = 1.0
                        changed = True

                    fixed_parts.append(f"{val:.6f}" if "." in val_str else str(int(val)))
                except ValueError:
                    # If it's not a float, keep as is (might be a class label or confidence)
                    fixed_parts.append(val_str)

            fixed_lines.append("\t".join(fixed_parts))

        # Write back only if changes were made
        if changed:
            with open(label_path, "w") as f:
                for line in fixed_lines:
                    f.write(line + "\n")
            return True, len([p for p in fixed_parts if p not in [keypoint_id]])
        return False, 0

    except Exception as e:
        print(f"Error processing {label_path}: {e}")
        return False, 0


def main():
    dataset_root = Path("/home/ng204/GH/ultralytics+rtmo/datasets/animalpose")
    label_dirs = [
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]

    total_files = 0
    fixed_files = 0

    for label_dir in label_dirs:
        if not label_dir.exists():
            print(f"Label directory not found: {label_dir}")
            continue

        print(f"\nProcessing {label_dir.name} labels...")

        for label_file in sorted(label_dir.glob("*.txt")):
            total_files += 1
            fixed, _count = fix_label_file(label_file)
            if fixed:
                fixed_files += 1
                print(f"  Fixed: {label_file.name}")

    print(f"\n{'=' * 60}")
    print(f"Summary: Fixed {fixed_files}/{total_files} label files")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
