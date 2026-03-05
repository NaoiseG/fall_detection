import argparse
from pathlib import Path
import shutil
import sys

import numpy as np


def _replacement_value(labels: np.ndarray):
    if labels.dtype.kind in ("U", "S"):
        return "2"
    if labels.dtype.kind == "O":
        for v in labels:
            if v != 20 and v != "20":
                return "2" if isinstance(v, str) else 2
        return "2"
    return 2


def _replace_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    if labels.dtype.kind in ("U", "S", "O"):
        mask = (labels == "20") | (labels == 20)
    else:
        mask = labels == 20

    count = int(np.sum(mask))
    if count == 0:
        return labels, 0

    out = labels.copy()
    out[mask] = _replacement_value(labels)
    return out, count


def process_file(path: Path, dry_run: bool, backup: bool) -> int:
    with np.load(path, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}

    labels = arrays.get("frame_labels")
    if labels is None:
        return 0

    new_labels, count = _replace_labels(labels)
    if count == 0:
        return 0

    if dry_run:
        return count

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    arrays["frame_labels"] = new_labels
    tmp_path = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(str(tmp_path), **arrays)
    tmp_path.replace(path)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Replace label 20 with label 2 in NPZ frame_labels."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="..\\..\\Datasets\\UPFall_keypoints_alpha\\outputs_npz",
        help="Root directory to scan for keypoints.npz files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report counts without modifying files.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a .bak copy before overwriting each NPZ.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    npz_paths = list(root.rglob("keypoints.npz"))
    print(f"Found {len(npz_paths)} NPZ files under {root}")

    total_files = 0
    total_labels = 0
    for p in npz_paths:
        count = process_file(p, dry_run=bool(args.dry_run), backup=bool(args.backup))
        if count > 0:
            total_files += 1
            total_labels += count
            print(f"{p}: replaced {count}")

    action = "Would replace" if args.dry_run else "Replaced"
    print(f"{action} {total_labels} labels across {total_files} files.")


if __name__ == "__main__":
    main()
