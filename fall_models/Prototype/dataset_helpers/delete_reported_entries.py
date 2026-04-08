#!/usr/bin/env python3
"""
Delete reported UP-Fall entries listed in a CSV under a mirrored root tree.

The CSV is expected to contain a ``file`` column like:
    keypoints/UPFall_keypoints/yolo11l/base/Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera2/keypoints.npz

By default, each row is mapped to the camera directory under ``--root``:
    <root>/Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera2

Use ``--target file`` if you only want to delete the reported file itself
instead of the whole camera directory.

Examples:
    python dataset_helpers/delete_reported_entries.py bad_keypoints_report.csv ^
        --root /scratch/keypoints/UPFall_keypoints/yolo11l/base

    python dataset_helpers/delete_reported_entries.py bad_keypoints_report.csv ^
        --root /scratch/keypoints/UPFall_keypoints/yolo11l/base --execute

    python dataset_helpers/delete_reported_entries.py bad_keypoints_report.csv ^
        --root /scratch/keypoints/UPFall_keypoints/yolo11l/base --target file --execute
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class TargetRecord:
    row_number: int
    relative_path: Path
    absolute_path: Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Delete entries listed in a CSV under a supplied mirrored root. "
            "Defaults to preview mode; pass --execute to actually delete."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV report.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Root directory whose layout mirrors the keypoint tree below the "
            "Subject*/Activity*/Trial*/Camera* level."
        ),
    )
    parser.add_argument(
        "--target",
        choices=("camera-dir", "file"),
        default="camera-dir",
        help=(
            "Delete the camera directory described by each CSV row, or only the "
            "reported file itself."
        ),
    )
    parser.add_argument(
        "--file-column",
        default="file",
        help="CSV column containing the reported path.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete targets. Without this flag, only print what would be deleted.",
    )
    return parser


def csv_path_to_posix_parts(path_text: str) -> List[str]:
    normalized = path_text.replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in ("", ".")]
    if not parts:
        raise ValueError("empty path value")
    return parts


def build_camera_relative_path(row: Dict[str, str], file_column: str) -> Path:
    metadata_parts = [
        (row.get("subject") or "").strip(),
        (row.get("activity") or "").strip(),
        (row.get("trial") or "").strip(),
        (row.get("camera_dir") or "").strip(),
    ]
    if all(metadata_parts):
        return Path(*metadata_parts)

    file_value = (row.get(file_column) or "").strip()
    if not file_value:
        raise ValueError(f"missing '{file_column}' value")

    parts = csv_path_to_posix_parts(file_value)

    for index, part in enumerate(parts):
        if part.startswith("Subject"):
            if index + 3 >= len(parts):
                raise ValueError(
                    f"could not infer camera directory from '{file_value}'"
                )
            return Path(*parts[index:-1])

    if len(parts) < 2:
        raise ValueError(f"could not infer camera directory from '{file_value}'")

    return Path(*parts[:-1])


def build_target_relative_path(
    row: Dict[str, str],
    file_column: str,
    target_kind: str,
) -> Path:
    camera_relative_path = build_camera_relative_path(row, file_column)
    if target_kind == "camera-dir":
        return camera_relative_path

    file_value = (row.get(file_column) or "").strip()
    if file_value:
        file_name = PurePosixPath(file_value.replace("\\", "/")).name
    else:
        file_name = "keypoints.npz"

    if not file_name:
        raise ValueError(f"could not infer file name from '{file_value}'")

    return camera_relative_path / file_name


def resolve_target_under_root(root: Path, relative_path: Path) -> Path:
    absolute_path = (root / relative_path).resolve()
    try:
        absolute_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"refusing to operate outside root: {absolute_path}"
        ) from exc
    return absolute_path


def iter_target_records(
    csv_path: Path,
    root: Path,
    file_column: str,
    target_kind: str,
) -> Iterable[TargetRecord]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        for row_number, row in enumerate(reader, start=2):
            relative_path = build_target_relative_path(row, file_column, target_kind)
            absolute_path = resolve_target_under_root(root, relative_path)
            yield TargetRecord(
                row_number=row_number,
                relative_path=relative_path,
                absolute_path=absolute_path,
            )


def delete_path(target: Path) -> None:
    if target.is_dir():
        shutil.rmtree(target)
        return
    target.unlink()


def main() -> int:
    args = build_arg_parser().parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    root = args.root.expanduser().resolve()

    if not csv_path.is_file():
        raise SystemExit(f"CSV does not exist or is not a file: {csv_path}")
    if not root.is_dir():
        raise SystemExit(f"Root does not exist or is not a directory: {root}")

    try:
        records = list(
            iter_target_records(
                csv_path=csv_path,
                root=root,
                file_column=args.file_column,
                target_kind=args.target,
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    deduped_records: List[TargetRecord] = []
    seen_paths = set()
    duplicate_count = 0

    for record in records:
        dedupe_key = str(record.absolute_path)
        if dedupe_key in seen_paths:
            duplicate_count += 1
            print(f"[skip-duplicate] row={record.row_number} target={record.relative_path}")
            continue
        seen_paths.add(dedupe_key)
        deduped_records.append(record)

    existing_records: List[TargetRecord] = []
    missing_records: List[TargetRecord] = []
    for record in deduped_records:
        if record.absolute_path.exists():
            existing_records.append(record)
            label = "delete" if args.execute else "would-delete"
            print(f"[{label}] row={record.row_number} target={record.relative_path}")
        else:
            missing_records.append(record)
            print(f"[missing] row={record.row_number} target={record.relative_path}")

    deleted_count = 0
    if args.execute:
        for record in existing_records:
            delete_path(record.absolute_path)
            deleted_count += 1

    print()
    print(
        "Summary: "
        f"rows={len(records)} "
        f"unique_targets={len(deduped_records)} "
        f"existing={len(existing_records)} "
        f"missing={len(missing_records)} "
        f"duplicates={duplicate_count} "
        f"deleted={deleted_count if args.execute else 0} "
        f"mode={args.target} "
        f"execute={args.execute}"
    )

    if not args.execute:
        print("Preview only. Re-run with --execute to delete the listed targets.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
