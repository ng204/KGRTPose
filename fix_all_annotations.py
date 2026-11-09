#!/usr/bin/env python3
"""
Complete annotation fix script for animalpose dataset.
Fixes:
1. Out-of-bounds coordinates (clamps to [0, 1])
2. Class label mismatches (fixes class IDs)
3. Invalid visibility values.
"""

from pathlib import Path


def fix_annotation_files():
    dataset_root = Path("datasets/animalpose")

    print("=" * 80)
    print("🔧 Fixing all annotation issues...")
    print("=" * 80)

    stats = {
        "files_processed": 0,
        "coords_clipped": 0,
        "class_id_fixed": 0,
        "visibility_fixed": 0,
    }

    for split in ["train", "val"]:
        label_dir = dataset_root / "labels" / split
        if not label_dir.exists():
            print(f"Label directory not found: {label_dir}")
            continue

        print(f"\n🔍 Processing {split.upper()} set...")

        label_files = sorted(label_dir.glob("*.txt"))

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

                        if len(parts) < 2:
                            fixed_lines.append(line)
                            continue

                        # First element is class_id
                        class_id = int(float(parts[0]))

                        # Fix class_id if it's not 0
                        if class_id != 0:
                            class_id = 0
                            stats["class_id_fixed"] += 1
                            file_modified = True

                        # Process remaining coordinates
                        fixed_values = [str(class_id)]

                        for i, val_str in enumerate(parts[1:]):
                            try:
                                val = float(val_str)

                                # Clamp to [0, 1] for coordinates
                                # (every 3 values: x, y, visibility)
                                coord_idx = i % 3
                                if coord_idx < 2:  # x or y coordinate
                                    orig_val = val
                                    val = max(0.0, min(1.0, val))
                                    if val != orig_val:
                                        stats["coords_clipped"] += 1
                                        file_modified = True
                                    fixed_values.append(f"{val:.6f}")
                                else:  # visibility value
                                    # Clamp visibility to [0, 2] range
                                    orig_val = val
                                    val = max(0.0, min(2.0, val))
                                    if val != orig_val:
                                        stats["visibility_fixed"] += 1
                                        file_modified = True
                                    # Keep visibility as integer if it was
                                    if val == int(val):
                                        fixed_values.append(str(int(val)))
                                    else:
                                        fixed_values.append(f"{val:.6f}")
                            except ValueError:
                                fixed_values.append(val_str)

                        fixed_line = " ".join(fixed_values) + "\n"
                        fixed_lines.append(fixed_line)

                    except Exception:
                        # Keep problematic lines as-is
                        fixed_lines.append(line)

                # Write back if modified
                if file_modified:
                    with open(label_file, "w") as f:
                        f.writelines(fixed_lines)
                    print(f"  ✓ Fixed: {label_file.name}")

                stats["files_processed"] += 1

            except Exception as e:
                print(f"  ⚠️  Error processing {label_file.name}: {e}")

    # Remove cache files to force re-scanning
    cache_dir = dataset_root / "labels"
    for cache_file in cache_dir.glob("*.cache"):
        try:
            cache_file.unlink()
            print(f"  🗑️  Removed cache: {cache_file.name}")
        except Exception as e:
            print(f"  ⚠️  Could not remove {cache_file.name}: {e}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("✅ Annotation fixes complete!")
    print(f"{'=' * 80}")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Coordinates clipped: {stats['coords_clipped']}")
    print(f"Class IDs fixed: {stats['class_id_fixed']}")
    print(f"Visibility values fixed: {stats['visibility_fixed']}")
    print("\n💾 Fixed files saved in place")
    print("⚠️  Caches cleared — they will be rebuilt on next training run\n")


if __name__ == "__main__":
    fix_annotation_files()
