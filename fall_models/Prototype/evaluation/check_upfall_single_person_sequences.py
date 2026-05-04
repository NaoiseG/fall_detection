"""Check UP-Fall frame sequences for exactly one visible person.

Scans UP-Fall frame folders for activities 1-5 and subjects 16-17 by default.
Each camera frame folder is treated as one video sequence, matching the
directory layout used by dataset_helpers/get_keypoints_files.py.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


DEFAULT_FRAME_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class SequenceCheck:
    path: Path
    sampled_frames: int
    bad_frames: int
    first_bad_frame: Path | None
    first_bad_person_count: int | None

    @property
    def has_only_one_person(self) -> bool:
        return self.sampled_frames > 0 and self.bad_frames == 0


def parse_int_list(value: str, label: str) -> list[int]:
    """Parse comma-separated integers and ranges, e.g. 1-5,8."""
    items: set[int] = set()
    for chunk in (part.strip() for part in value.split(",")):
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            try:
                start = int(start_text.strip())
                end = int(end_text.strip())
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid {label} range: {chunk}") from exc
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid {label} range: {chunk}")
            items.update(range(start, end + 1))
        else:
            try:
                items.add(int(chunk))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid {label}: {chunk}") from exc

    if not items:
        raise argparse.ArgumentTypeError(f"{label.capitalize()} cannot be empty.")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError(f"{label.capitalize()} must be positive integers.")
    return sorted(items)


def parse_subjects(value: str) -> list[int]:
    return parse_int_list(value, "subjects")


def parse_activities(value: str) -> list[int]:
    return parse_int_list(value, "activities")


def parse_cameras(value: str) -> list[int]:
    return parse_int_list(value, "cameras")


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_frame_files(sequence_dir: Path, extensions: Iterable[str]) -> list[Path]:
    normalized_extensions = {ext.lower() for ext in extensions}
    frames = [
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    return sorted(frames, key=natural_sort_key)


def find_sequence_dirs(
    upfall_root: Path,
    subjects: Iterable[int],
    activities: Iterable[int],
    cameras: Iterable[int] | None,
    extensions: Iterable[str],
) -> list[Path]:
    camera_suffixes = None if cameras is None else tuple(f"Camera{camera}" for camera in cameras)
    sequence_dirs: list[Path] = []

    for subject in subjects:
        subject_root = upfall_root / f"Subject{subject}"
        if not subject_root.is_dir():
            print(f"Missing subject folder, skipping: {subject_root}")
            continue

        for activity in activities:
            activity_root = subject_root / f"Activity{activity}"
            if not activity_root.is_dir():
                print(f"Missing activity folder, skipping: {activity_root}")
                continue

            for candidate in activity_root.rglob("*Camera*"):
                if not candidate.is_dir():
                    continue
                if camera_suffixes is not None and not candidate.name.endswith(camera_suffixes):
                    continue
                if list_frame_files(candidate, extensions):
                    sequence_dirs.append(candidate)

    return sorted(set(sequence_dirs), key=lambda path: path.as_posix())


def person_class_id(model: YOLO) -> int:
    names = getattr(model, "names", None) or {}
    items = names.items() if hasattr(names, "items") else enumerate(names)
    for class_id, name in items:
        if str(name).lower() == "person":
            return int(class_id)
    return 0


def count_people(result, person_id: int) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return 0
    classes = boxes.cls.detach().cpu().tolist()
    return sum(1 for class_id in classes if int(class_id) == person_id)


def check_sequence(
    model: YOLO,
    sequence_dir: Path,
    person_id: int,
    frame_step: int,
    extensions: Iterable[str],
    conf: float,
    imgsz: int | None,
    max_det: int | None,
    stop_on_first_failure: bool,
) -> SequenceCheck:
    frames = list_frame_files(sequence_dir, extensions)
    sampled_frames = frames[::frame_step]
    bad_frames = 0
    first_bad_frame: Path | None = None
    first_bad_person_count: int | None = None

    for frame_path in sampled_frames:
        predict_kwargs = {
            "source": str(frame_path),
            "conf": conf,
            "verbose": False,
        }
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        if max_det is not None:
            predict_kwargs["max_det"] = max_det

        results = model.predict(**predict_kwargs)
        people = count_people(results[0], person_id) if results else 0

        if people != 1:
            bad_frames += 1
            if first_bad_frame is None:
                first_bad_frame = frame_path
                first_bad_person_count = people
            if stop_on_first_failure:
                break

    return SequenceCheck(
        path=sequence_dir,
        sampled_frames=len(sampled_frames),
        bad_frames=bad_frames,
        first_bad_frame=first_bad_frame,
        first_bad_person_count=first_bad_person_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an Ultralytics YOLO model on UP-Fall frame folders and print "
            "which sequences contain exactly one detected person in every sampled frame."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to YOLO weights, e.g. .pt or .engine.")
    parser.add_argument("--upfall-root", type=Path, required=True, help="Root of the UP-Fall dataset.")
    parser.add_argument("--subjects", type=parse_subjects, default=[16, 17], help="Subject IDs/ranges. Default: 16-17.")
    parser.add_argument("--activities", type=parse_activities, default=[1, 2, 3, 4, 5], help="Activity IDs/ranges. Default: 1-5.")
    parser.add_argument(
        "--cameras",
        type=parse_cameras,
        default=None,
        help="Camera IDs/ranges to scan. Default: all Camera folders found.",
    )
    parser.add_argument("--frame-step", type=int, default=3, help="Check every Nth frame. Default: 3.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold. Default: 0.25.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional YOLO inference image size.")
    parser.add_argument("--max-det", type=int, default=None, help="Optional YOLO max_det override.")
    parser.add_argument(
        "--frame-extensions",
        nargs="+",
        default=list(DEFAULT_FRAME_EXTENSIONS),
        help="Frame image extensions to include. Default: .png .jpg .jpeg .bmp.",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Continue scanning a sequence after the first non-one-person sampled frame.",
    )
    args = parser.parse_args()

    if args.frame_step <= 0:
        raise SystemExit("--frame-step must be a positive integer.")
    if YOLO is None:
        raise SystemExit("Missing dependency: ultralytics. Install it with `pip install ultralytics`.")

    upfall_root = args.upfall_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if not upfall_root.is_dir():
        raise SystemExit(f"UP-Fall root does not exist or is not a directory: {upfall_root}")
    if not model_path.exists():
        raise SystemExit(f"YOLO model path does not exist: {model_path}")

    print(f"UP-Fall root: {upfall_root}")
    print(f"YOLO model:   {model_path}")
    print(f"Subjects:     {args.subjects}")
    print(f"Activities:   {args.activities}")
    print(f"Cameras:      {args.cameras if args.cameras is not None else 'all'}")
    print(f"Frame step:   {args.frame_step}")

    sequence_dirs = find_sequence_dirs(
        upfall_root=upfall_root,
        subjects=args.subjects,
        activities=args.activities,
        cameras=args.cameras,
        extensions=args.frame_extensions,
    )
    print(f"Sequences found: {len(sequence_dirs)}")
    if not sequence_dirs:
        return

    model = YOLO(str(model_path))
    person_id = person_class_id(model)
    print(f"Person class id: {person_id}")

    checks: list[SequenceCheck] = []
    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        print(f"\n[{index}/{len(sequence_dirs)}] {sequence_dir}")
        check = check_sequence(
            model=model,
            sequence_dir=sequence_dir,
            person_id=person_id,
            frame_step=args.frame_step,
            extensions=args.frame_extensions,
            conf=args.conf,
            imgsz=args.imgsz,
            max_det=args.max_det,
            stop_on_first_failure=not args.full_scan,
        )
        checks.append(check)

        if check.has_only_one_person:
            print(f"  ONLY_ONE_PERSON sampled_frames={check.sampled_frames}")
        else:
            bad_detail = ""
            if check.first_bad_frame is not None:
                bad_detail = (
                    f" first_bad_frame={check.first_bad_frame.name}"
                    f" person_count={check.first_bad_person_count}"
                )
            print(
                "  NOT_ONLY_ONE_PERSON"
                f" sampled_frames={check.sampled_frames}"
                f" bad_frames={check.bad_frames}"
                f"{bad_detail}"
            )

    passing = [check for check in checks if check.has_only_one_person]
    print("\nSequences containing only one person in every sampled frame:")
    if passing:
        for check in passing:
            print(check.path)
    else:
        print("None")

    print(f"\nSummary: {len(passing)}/{len(checks)} sequences passed.")


if __name__ == "__main__":
    main()
