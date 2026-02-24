"""
Entry-point script to run ViTPose extraction on the full UP-Fall directory tree.

This file contains the main() logic plus a minimal ViTPose pipeline using
HuggingFace Transformers (RTDetr for person detection + ViTPose for keypoints).

Usage examples:
  python dataset_helpers/get_keypoints_files_ViTpose.py --camera 1
  python dataset_helpers/get_keypoints_files_ViTpose.py --subjects 12-12
  python dataset_helpers/get_keypoints_files_ViTpose.py --subjects 2,4,7
  python dataset_helpers/get_keypoints_files_ViTpose.py --camera 2 --subjects 1-3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "transformers is required for ViTPose extraction. "
        "Install with: pip install transformers"
    ) from exc


# ----------------------------- CONFIG -----------------------------

@dataclass
class VitPoseExportConfig:
    # HF model names
    detector_model: str = "PekingU/rtdetr_r50vd_coco_o365"
    pose_model: str = "usyd-community/vitpose-base"

    # thresholds
    person_threshold: float = 0.3
    pose_threshold: float = 0.3
    draw_kpt_threshold: float = 0.3

    fps: int = 30
    max_people: int = 1
    num_kpts: int = 17
    video_codec: str = "mp4v"
    save_csv: bool = False
    render_video: bool = True
    device: Optional[str] = None  # "cuda" or "cpu"; None => auto


# ----------------------------- PATH HELPERS -----------------------------

def find_camera_folders_subjects(root: str, camera: int = 1, subjects: List[int] | range = range(1, 6)) -> List[str]:
    folders = []
    for s in subjects:
        subj_root = Path(root) / f"Subject{s}"
        if not subj_root.exists():
            continue
        pat = subj_root / "**" / f"*Camera{camera}"
        folders.extend([str(p) for p in glob.glob(str(pat), recursive=True) if os.path.isdir(p)])
    return folders


def parse_subjects(subjects_str: Optional[str]) -> List[int] | range:
    if subjects_str is None or str(subjects_str).strip() == "":
        return range(1, 6)

    raw = str(subjects_str).strip()
    if "," in raw and "-" in raw:
        raise ValueError("subjects must be a comma list or a range, not both")

    if "-" in raw:
        parts = [p.strip() for p in raw.split("-")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("invalid subjects range, expected START-END")
        if not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError("subjects range must be numeric")
        start = int(parts[0])
        end = int(parts[1])
        if start <= 0 or end <= 0:
            raise ValueError("subjects must be positive integers")
        if start > end:
            raise ValueError("subjects range start must be <= end")
        return range(start, end + 1)

    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("subjects list cannot be empty")
    subjects = []
    for p in parts:
        if not p.isdigit():
            raise ValueError("subjects list must be numeric")
        val = int(p)
        if val <= 0:
            raise ValueError("subjects must be positive integers")
        subjects.append(val)
    return sorted(set(subjects))


def build_arg_parser(default_upfall_root: Path, default_output_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ViTPose extraction on UP-Fall frames."
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma list (e.g., 1,2,3) or range (e.g., 1-5). Default: 1-5.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        required=True,
        help="Camera index to process (e.g., 1 for Camera1).",
    )
    parser.add_argument(
        "--upfall-root",
        type=str,
        default=str(default_upfall_root),
        help="Root directory of the UP-Fall dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(default_output_root),
        help="Root directory where keypoint outputs are written.",
    )
    return parser


# ----------------------------- SORTING -----------------------------

FRAME_TS_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2}(?:\.\d+)?)"
)


def parse_frame_timestamp(path: str) -> pd.Timestamp:
    name = os.path.splitext(os.path.basename(path))[0]
    m = FRAME_TS_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse timestamp from {path}")
    dt = f"{m.group('date')}T{m.group('h')}:{m.group('m')}:{m.group('s')}"
    return pd.to_datetime(dt)


def frame_time_key(path: str) -> pd.Timestamp:
    return parse_frame_timestamp(path)


def list_frames(frames_dir: str, pattern: str = "*.png") -> List[str]:
    paths = glob.glob(os.path.join(frames_dir, pattern))
    paths = sorted(paths, key=frame_time_key)
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir} matching {pattern}")
    return paths


# ----------------------------- IO HELPERS -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_video_writer(out_path: str, fps: int, frame_size: Tuple[int, int], codec: str = "mp4v") -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    w, h = frame_size
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))


# ----------------------------- LABELS -----------------------------

def load_window_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=1, encoding="utf-8-sig")

    df.columns = [str(c).strip() for c in df.columns]

    if "Timestamp" not in df.columns:
        raise ValueError(f"'Timestamp' not found. Columns: {df.columns[:10].tolist()}")

    if "Tag" not in df.columns:
        raise ValueError(f"'Tag' not found. Columns: {df.columns[-10:].tolist()}")

    window_start = pd.to_datetime(
        df["Timestamp"],
        format="%Y-%m-%dT%H:%M:%S.%f",
        errors="coerce"
    )

    out = pd.DataFrame({
        "window_start": window_start,
        "label": df["Tag"].astype(str),
    })

    out = (
        out.dropna(subset=["window_start"])
           .sort_values("window_start")
           .reset_index(drop=True)
    )

    out["window_id"] = np.arange(len(out), dtype=np.int64)
    return out


# ----------------------------- DATA STRUCTURES -----------------------------

def init_pose_arrays(num_frames: int, max_people: int, num_kpts: int) -> Dict[str, np.ndarray]:
    kpts_xy = np.full((num_frames, max_people, num_kpts, 2), np.nan, dtype=np.float32)
    kpts_conf = np.full((num_frames, max_people, num_kpts), np.nan, dtype=np.float32)
    person_conf = np.full((num_frames, max_people), np.nan, dtype=np.float32)
    return {"kpts_xy": kpts_xy, "kpts_conf": kpts_conf, "person_conf": person_conf}


# ----------------------------- VITPOSE WRAPPER -----------------------------

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def _pose_to_arrays(person_pose: Dict[str, Any], num_kpts: int) -> Tuple[np.ndarray, np.ndarray]:
    xy = np.full((num_kpts, 2), np.nan, dtype=np.float32)
    conf = np.full((num_kpts,), np.nan, dtype=np.float32)

    kp = _to_numpy(person_pose.get("keypoints"))
    scores = _to_numpy(person_pose.get("scores"))
    labels = _to_numpy(person_pose.get("labels"))

    if kp is None:
        return xy, conf

    kp = np.asarray(kp, dtype=np.float32).reshape(-1, 2)
    if labels is None:
        count = min(num_kpts, kp.shape[0])
        xy[:count] = kp[:count]
        if scores is not None:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            conf[:count] = scores[:count]
        return xy, conf

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    for idx, lab in enumerate(labels):
        if 0 <= lab < num_kpts and idx < kp.shape[0]:
            xy[lab] = kp[idx]
            if scores is not None and idx < scores.shape[0]:
                conf[lab] = scores[idx]

    return xy, conf


class VitPoseRunner:
    def __init__(self, config: VitPoseExportConfig):
        self.config = config
        if config.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.device.lower()
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
        self.device = device

        self.person_image_processor = AutoProcessor.from_pretrained(config.detector_model)
        self.person_model = RTDetrForObjectDetection.from_pretrained(config.detector_model)
        self.person_model.to(self.device).eval()

        self.pose_image_processor = AutoProcessor.from_pretrained(config.pose_model)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(config.pose_model)
        self.pose_model.to(self.device).eval()

    def _detect_people(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        inputs = self.person_image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.person_model(**inputs)

        results = self.person_image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)]),
            threshold=self.config.person_threshold,
        )
        result = results[0]

        labels = result["labels"]
        mask = labels == 0  # person class in COCO
        boxes = result["boxes"][mask]
        scores = result["scores"][mask]

        if boxes.numel() == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        boxes = boxes.detach().cpu().numpy().astype(np.float32)
        scores = scores.detach().cpu().numpy().astype(np.float32)

        # Convert boxes from VOC (x1,y1,x2,y2) to COCO (x1,y1,w,h)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        return boxes, scores

    def _estimate_pose(self, image: Image.Image, boxes_xywh: np.ndarray) -> List[Dict[str, Any]]:
        if boxes_xywh.size == 0:
            return []

        inputs = self.pose_image_processor(image, boxes=[boxes_xywh], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.pose_model(**inputs)

        pose_results = self.pose_image_processor.post_process_pose_estimation(
            outputs, boxes=[boxes_xywh], threshold=self.config.pose_threshold
        )
        return pose_results[0] if pose_results else []

    def infer(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        boxes, scores = self._detect_people(image)
        if boxes.shape[0] == 0:
            return []

        if boxes.shape[0] > self.config.max_people:
            order = np.argsort(-scores)[: self.config.max_people]
            boxes = boxes[order]
            scores = scores[order]

        pose_results = self._estimate_pose(image, boxes)

        count = min(len(pose_results), len(scores))
        people: List[Dict[str, Any]] = []
        for i in range(count):
            xy, conf = _pose_to_arrays(pose_results[i], self.config.num_kpts)
            pconf = float(scores[i]) if i < len(scores) else float(np.nanmean(conf))
            people.append({"kpts_xy": xy, "kpts_conf": conf, "person_conf": pconf})
        return people

    def render(self, frame_bgr: np.ndarray, people: List[Dict[str, Any]]) -> np.ndarray:
        if not people:
            return frame_bgr

        out = frame_bgr.copy()
        for person in people:
            xy = person["kpts_xy"]
            conf = person["kpts_conf"]

            for a, b in COCO_SKELETON:
                if a >= xy.shape[0] or b >= xy.shape[0]:
                    continue
                if np.any(np.isnan(xy[a])) or np.any(np.isnan(xy[b])):
                    continue
                if not np.isfinite(conf[a]) or not np.isfinite(conf[b]):
                    continue
                if conf[a] < self.config.draw_kpt_threshold or conf[b] < self.config.draw_kpt_threshold:
                    continue
                pt1 = (int(xy[a][0]), int(xy[a][1]))
                pt2 = (int(xy[b][0]), int(xy[b][1]))
                cv2.line(out, pt1, pt2, (0, 255, 0), 2)

            for k in range(xy.shape[0]):
                if np.any(np.isnan(xy[k])):
                    continue
                if not np.isfinite(conf[k]):
                    continue
                if conf[k] < self.config.draw_kpt_threshold:
                    continue
                pt = (int(xy[k][0]), int(xy[k][1]))
                cv2.circle(out, pt, 3, (0, 0, 255), -1)

        return out


# ----------------------------- CORE PIPELINE -----------------------------

def run_pose_on_frames_vitpose(
    frames_dir: str,
    out_dir: str,
    windows_csv: str,
    config: VitPoseExportConfig,
    pattern: str = "*.png",
    runner: Optional[VitPoseRunner] = None,
) -> Tuple[str, str, Optional[str]]:
    """
    Processes a directory of frames.
    Writes:
      - annotated video (mp4)
      - keypoints (npz)
      - optional csv
    Returns paths to outputs.
    """
    ensure_dir(out_dir)

    frame_paths = list_frames(frames_dir, pattern=pattern)
    num_frames = len(frame_paths)

    if runner is None:
        runner = VitPoseRunner(config)

    first_bgr = cv2.imread(frame_paths[0])
    if first_bgr is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    h, w = first_bgr.shape[:2]

    out_video = os.path.join(out_dir, "pose_out.mp4")
    out_npz = os.path.join(out_dir, "keypoints.npz")
    out_csv = os.path.join(out_dir, "keypoints.csv") if config.save_csv else None

    writer = make_video_writer(out_video, config.fps, (w, h), codec=config.video_codec)

    arrays = init_pose_arrays(num_frames, config.max_people, config.num_kpts)

    csv_rows = []
    if config.save_csv:
        csv_rows.append(["frame", "person", "kpt", "x", "y", "kpt_conf", "person_conf", "frame_path"])

    windows_df = load_window_labels(windows_csv)
    frame_dts = [parse_frame_timestamp(p) for p in frame_paths]

    frames_df = pd.DataFrame({
        "frame_idx": np.arange(num_frames),
        "frame_path": frame_paths,
        "frame_dt": frame_dts,
    }).sort_values("frame_dt").reset_index(drop=True)

    frames_df = pd.merge_asof(
        frames_df.sort_values("frame_dt"),
        windows_df.sort_values("window_start"),
        left_on="frame_dt",
        right_on="window_start",
        direction="backward",
        allow_exact_matches=True,
    )

    frames_df["label"] = frames_df["label"].fillna("unknown")
    frames_df["window_id"] = frames_df["window_id"].fillna(-1).astype(np.int64)

    frame_labels = frames_df["label"].to_numpy()
    frame_window_ids = frames_df["window_id"].to_numpy()

    for i, p in enumerate(frame_paths):
        frame_bgr = cv2.imread(p)
        if frame_bgr is None:
            print(f"Skipping unreadable frame: {p}")
            continue

        people = runner.infer(frame_bgr)

        if config.render_video:
            annotated = runner.render(frame_bgr, people)
            writer.write(annotated)
        else:
            writer.write(frame_bgr)

        if not people:
            continue

        people_sorted = sorted(people, key=lambda x: float(x["person_conf"]), reverse=True)

        for j, person in enumerate(people_sorted[:config.max_people]):
            arrays["kpts_xy"][i, j] = person["kpts_xy"]
            arrays["kpts_conf"][i, j] = person["kpts_conf"]
            arrays["person_conf"][i, j] = float(person["person_conf"])

            if config.save_csv:
                for k in range(config.num_kpts):
                    x, y = arrays["kpts_xy"][i, j, k]
                    kconf = arrays["kpts_conf"][i, j, k]
                    pconf = arrays["person_conf"][i, j]
                    csv_rows.append([i, j, k, float(x), float(y), float(kconf), float(pconf), p])

    writer.release()

    np.savez_compressed(
        out_npz,
        kpts_xy=arrays["kpts_xy"],
        kpts_conf=arrays["kpts_conf"],
        person_conf=arrays["person_conf"],
        frame_paths=np.array(frame_paths, dtype=object),
        frame_labels=frame_labels,
        window_ids=frame_window_ids,
        frame_timestamps=np.array(frame_dts, dtype="datetime64[ns]"),
        fps=np.array([config.fps], dtype=np.int32),
        model_path=np.array([config.pose_model], dtype=object),
    )

    if config.save_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerows(csv_rows)

    return out_video, out_npz, out_csv


# ----------------------------- MAIN -----------------------------

def main() -> None:
    # --- Configure paths for your PC ---
    UPFALL_ROOT = Path("../../Datasets/UPFall")  # change if needed
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints_vitpose/outputs_npz")  # change if needed

    parser = build_arg_parser(UPFALL_ROOT, OUTPUT_ROOT)
    args = parser.parse_args()

    try:
        subjects = parse_subjects(args.subjects)
    except ValueError as exc:
        parser.error(str(exc))

    upfall_root = Path(args.upfall_root)
    output_root = Path(args.output_root)

    cfg = VitPoseExportConfig(
        detector_model="PekingU/rtdetr_r50vd_coco_o365",
        pose_model="usyd-community/vitpose-base",
        person_threshold=0.3,
        pose_threshold=0.3,
        draw_kpt_threshold=0.3,
        fps=30,
        max_people=1,
        save_csv=False,
        render_video=True,
    )

    runner = VitPoseRunner(cfg)

    camera_folders = find_camera_folders_subjects(
        root=str(upfall_root),
        camera=args.camera,
        subjects=subjects,
    )

    print("Camera folders found:", len(camera_folders))
    total = len(camera_folders)
    results = []

    for i, frames_dir in enumerate(camera_folders, 1):
        print(f"\n[{i}/{total}] Processing: {frames_dir}")

        trial_dir = os.path.dirname(frames_dir)
        matches = glob.glob(os.path.join(trial_dir, "*Features1&0.5.csv"))
        if not matches:
            print("  -> no Features1&0.5.csv found, skipping")
            continue
        windows_csv = matches[0]

        rel = os.path.relpath(frames_dir, str(upfall_root))
        out_dir = output_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "keypoints.npz").exists():
            print("  -> already exists, skipping")
            continue

        out_video, out_npz, _ = run_pose_on_frames_vitpose(
            frames_dir=frames_dir,
            out_dir=str(out_dir),
            windows_csv=windows_csv,
            config=cfg,
            pattern="*.png",
            runner=runner,
        )

        print(f"  OK wrote {out_npz}")
        results.append(out_npz)

    print("\nDone.")
    print("Processed sequences:", len(results))


if __name__ == "__main__":
    main()
