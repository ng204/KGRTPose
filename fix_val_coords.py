#!/usr/bin/env python3
"""Fix out-of-bounds coordinates in validation set labels."""

from pathlib import Path


def fix_label_file(label_path):
    """Fix a single label file by clamping all float values to [0, 1]."""
    try:
        with open(label_path) as f:
            lines = f.readlines()

        fixed_lines = []
        changed = False

        for line in lines:
            line = line.strip()
            if not line:
                fixed_lines.append("")
                continue

            parts = line.split()
            fixed_parts = []

            for part in parts:
                try:
                    val = float(part)
                    # Clamp to [0, 1]
                    val = max(0.0, min(1.0, val))
                    if val != float(part):
                        changed = True
                    # Format with 6 decimals
                    fixed_parts.append(f"{val:.6f}")
                except ValueError:
                    # Keep non-float values as-is
                    fixed_parts.append(part)

            fixed_lines.append(" ".join(fixed_parts))

        # Write back if changed
        if changed:
            with open(label_path, "w") as f:
                f.write("\n".join(fixed_lines))
            return True
        return False

    except Exception as e:
        print(f"❌ Error in {label_path}: {e}")
        return False


def main():
    val_dir = Path("/home/ng204/GH/ultralytics+rtmo/datasets/animalpose/labels/val")

    if not val_dir.exists():
        print(f"❌ Validation directory not found: {val_dir}")
        return

    print("🔧 Fixing validation set coordinates...")
    fixed_count = 0
    total_count = 0

    for label_file in sorted(val_dir.glob("*.txt")):
        total_count += 1
        if fix_label_file(label_file):
            fixed_count += 1
            print(f"  ✓ Fixed {label_file.name}")

    print(f"\n{'=' * 60}")
    print(f"✅ Fixed {fixed_count}/{total_count} validation label files")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
