import json
import os
import glob
import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import torch

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------- CONFIG -----------------------------

@dataclass
class PoseExportConfig:
    model_path: str = "pose_models/ultralytics/yolo11l-pose.pt"
    conf_thres: float = 0.25
    conf_min: float = 0.75
    fps: int = 30
    max_people: int = 1
    detector_max_det: int = 10
    num_kpts: int = 17               # COCO keypoints for Ultralytics pose models
    video_codec: str = "mp4v"        # mp4v is widely supported
    save_csv: bool = False
    max_jump_px: Optional[float] = None
    max_jump_diag_frac: float = 0.25
    max_lost: int = 10
    switch_margin_px: float = 20.0
    reset_on_max_lost: bool = False
    target_x_frac: float = 0.5
    target_y_frac: float = 0.5
    draw_kpt_threshold: float = 0.30
    draw_no_target_text: bool = True


# ----------------------------- SORTING -----------------------------

def frame_time_key(path: str) -> pd.Timestamp:
    return parse_frame_timestamp(path)

def list_frames(frames_dir: str, pattern: str = "*.png") -> List[str]:
    paths = glob.glob(os.path.join(frames_dir, pattern))
    paths = sorted(paths, key=frame_time_key)
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir} matching {pattern}")
    return paths

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


# ----------------------------- IO HELPERS -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_video_writer(out_path: str, fps: int, frame_size: Tuple[int, int], codec: str = "mp4v") -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    w, h = frame_size
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def is_engine_weights_path(weights_path: Path) -> bool:
    return weights_path.suffix.lower() == ".engine"


def resolve_yolo_predict_device(device: str, yolo_is_engine: bool) -> Any:
    """
    Ultralytics accepts string/int device selectors.
    For TensorRT engines, a numeric GPU index is often the most compatible.
    """
    device_str = str(device).strip()
    if not yolo_is_engine:
        return device_str

    d = device_str.lower()
    if d == "cuda":
        return 0
    if d.startswith("cuda:"):
        idx = d.split(":", 1)[1].strip()
        if idx.isdigit():
            return int(idx)
    return device_str


def _has_ultralytics_engine_metadata(engine_path: Path) -> bool:
    """
    Ultralytics TensorRT loader expects engine files to begin with:
      [4-byte little-endian metadata length][JSON metadata][serialized TRT engine]
    """
    try:
        file_size = int(engine_path.stat().st_size)
        if file_size <= 4:
            return False

        with engine_path.open("rb") as f:
            raw_len = f.read(4)
            if len(raw_len) != 4:
                return False
            meta_len = int.from_bytes(raw_len, byteorder="little", signed=False)

            max_reasonable = min(file_size - 4, 8 * 1024 * 1024)
            if meta_len <= 0 or meta_len > int(max_reasonable):
                return False

            meta_raw = f.read(meta_len)
            if len(meta_raw) != int(meta_len):
                return False

        meta = json.loads(meta_raw.decode("utf-8"))
        return isinstance(meta, dict)
    except Exception:
        return False


def ensure_ultralytics_engine_header(engine_path: Path) -> Path:
    """
    Wrap metadata-less TRT engines with an empty Ultralytics metadata header.
    """
    if _has_ultralytics_engine_metadata(engine_path):
        return engine_path

    stem = engine_path.stem
    if not stem.endswith(".ultra"):
        stem = f"{stem}.ultra"
    wrapped_path = engine_path.with_name(f"{stem}.engine")

    try:
        src_stat = engine_path.stat()
        if wrapped_path.exists():
            dst_stat = wrapped_path.stat()
            if int(dst_stat.st_mtime) >= int(src_stat.st_mtime) and int(dst_stat.st_size) > int(src_stat.st_size):
                return wrapped_path
    except OSError:
        pass

    meta_raw = b"{}"
    with engine_path.open("rb") as src, wrapped_path.open("wb") as dst:
        dst.write(int(len(meta_raw)).to_bytes(4, byteorder="little", signed=False))
        dst.write(meta_raw)
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return wrapped_path


# ----------------------------- DATA STRUCTURES -----------------------------

def init_pose_arrays(num_frames: int, max_people: int, num_kpts: int) -> Dict[str, np.ndarray]:
    kpts_xy = np.full((num_frames, max_people, num_kpts, 2), np.nan, dtype=np.float32)
    kpts_conf = np.full((num_frames, max_people, num_kpts), np.nan, dtype=np.float32)
    person_conf = np.full((num_frames, max_people), np.nan, dtype=np.float32)
    return {"kpts_xy": kpts_xy, "kpts_conf": kpts_conf, "person_conf": person_conf}


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


def _to_numpy_or_none(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def extract_box_centers_conf(r) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Safely extract:
      - box centers: (P, 2)
      - box conf: (P,) or None
      - boxes xyxy: (P, 4)
    """
    if r.boxes is None:
        empty_centers = np.empty((0, 2), dtype=np.float32)
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        return empty_centers, None, empty_boxes

    boxes_xyxy = _to_numpy_or_none(getattr(r.boxes, "xyxy", None))
    if boxes_xyxy is None:
        empty_centers = np.empty((0, 2), dtype=np.float32)
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        return empty_centers, None, empty_boxes

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[0] == 0 or boxes_xyxy.shape[1] < 4:
        empty_centers = np.empty((0, 2), dtype=np.float32)
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        return empty_centers, None, empty_boxes

    boxes_xyxy = boxes_xyxy[:, :4]
    centers = np.column_stack((
        0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]),
        0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]),
    )).astype(np.float32)

    box_conf = _to_numpy_or_none(getattr(r.boxes, "conf", None))
    if box_conf is not None:
        box_conf = np.asarray(box_conf, dtype=np.float32).reshape(-1)
        if box_conf.shape[0] < boxes_xyxy.shape[0]:
            pad = boxes_xyxy.shape[0] - box_conf.shape[0]
            box_conf = np.pad(box_conf, (0, pad), mode="constant", constant_values=np.nan)
        elif box_conf.shape[0] > boxes_xyxy.shape[0]:
            box_conf = box_conf[: boxes_xyxy.shape[0]]

    return centers, box_conf, boxes_xyxy


def select_person_idx(
    box_centers: np.ndarray,
    box_conf: Optional[np.ndarray],
    prev_center: Optional[np.ndarray],
    target_center: np.ndarray,
    conf_min: float,
    max_jump_px: float,
    switch_margin_px: float,
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    Temporal target selection:
      - Acquire (no prev_center): prefer conf >= conf_min, closest to target center.
      - Track (has prev_center): closest to prev_center with max-jump gate.
        If multiple candidates are similarly close, keep the current target lock.
    """
    num_people = int(box_centers.shape[0])
    if num_people == 0:
        return None, prev_center

    if prev_center is None:
        candidate_idx = np.arange(num_people, dtype=np.int32)
        if box_conf is not None and box_conf.shape[0] >= num_people:
            high_conf = np.where(np.isfinite(box_conf[:num_people]) & (box_conf[:num_people] >= conf_min))[0]
            if high_conf.size > 0:
                candidate_idx = high_conf.astype(np.int32, copy=False)

        dists = np.linalg.norm(box_centers[candidate_idx] - target_center[None, :], axis=1)
        if dists.size == 0:
            return None, prev_center

        best_rel = int(np.argmin(dists))
        best_idx = int(candidate_idx[best_rel])
        return best_idx, box_centers[best_idx].astype(np.float32, copy=True)

    dists = np.linalg.norm(box_centers - prev_center[None, :], axis=1)
    if dists.size == 0:
        return None, prev_center

    valid = np.where(np.isfinite(dists) & (dists <= max_jump_px))[0]
    if valid.size == 0:
        return None, prev_center

    ranked = valid[np.argsort(dists[valid])]
    best_idx = int(ranked[0])

    if ranked.size >= 2 and switch_margin_px > 0.0:
        second_idx = int(ranked[1])
        margin = float(dists[second_idx] - dists[best_idx])
        if margin < switch_margin_px:
            return None, prev_center

    return best_idx, box_centers[best_idx].astype(np.float32, copy=True)


def draw_selected_pose(
    frame: np.ndarray,
    kpts_xy: Optional[np.ndarray],
    kpts_conf: Optional[np.ndarray],
    box_xyxy: Optional[np.ndarray],
    draw_kpt_threshold: float,
    draw_no_target_text: bool,
) -> np.ndarray:
    out = frame.copy()
    if kpts_xy is None:
        if draw_no_target_text:
            cv2.putText(
                out,
                "NO TARGET",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        return out

    xy = np.asarray(kpts_xy, dtype=np.float32)
    conf = None
    if kpts_conf is not None:
        conf = np.asarray(kpts_conf, dtype=np.float32).reshape(-1)

    if box_xyxy is not None:
        box_xyxy = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)
        if box_xyxy.shape[0] >= 4 and np.all(np.isfinite(box_xyxy[:4])):
            x1, y1, x2, y2 = np.round(box_xyxy[:4]).astype(int).tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for a, b in COCO_SKELETON:
        if a >= xy.shape[0] or b >= xy.shape[0]:
            continue
        if not np.isfinite(xy[a]).all() or not np.isfinite(xy[b]).all():
            continue
        if conf is not None:
            if a >= conf.shape[0] or b >= conf.shape[0]:
                continue
            if not np.isfinite(conf[a]) or not np.isfinite(conf[b]):
                continue
            if conf[a] < draw_kpt_threshold or conf[b] < draw_kpt_threshold:
                continue

        pt1 = tuple(np.round(xy[a]).astype(int).tolist())
        pt2 = tuple(np.round(xy[b]).astype(int).tolist())
        cv2.line(out, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    for k in range(xy.shape[0]):
        if not np.isfinite(xy[k]).all():
            continue
        if conf is not None:
            if k >= conf.shape[0] or not np.isfinite(conf[k]):
                continue
            if conf[k] < draw_kpt_threshold:
                continue
        pt = tuple(np.round(xy[k]).astype(int).tolist())
        cv2.circle(out, pt, 3, (0, 0, 255), -1, cv2.LINE_AA)

    return out


def extract_pose_for_frame(r) -> Optional[Dict[str, Any]]:
    """
    Return dict with:
      xy: (P, K, 2)
      kc: (P, K)
      box_conf: (P,) or None
      box_centers: (P, 2)
      boxes_xyxy: (P, 4)
    """
    if r.keypoints is None or getattr(r.keypoints, "xy", None) is None:
        return None

    xy = _to_numpy_or_none(r.keypoints.xy)
    if xy is None:
        return None
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[0] == 0:
        return None

    kc = _to_numpy_or_none(getattr(r.keypoints, "conf", None))
    if kc is not None:
        kc = np.asarray(kc, dtype=np.float32)

    box_centers, box_conf, boxes_xyxy = extract_box_centers_conf(r)

    return {
        "xy": xy,
        "kc": kc,
        "box_conf": box_conf,
        "box_centers": box_centers,
        "boxes_xyxy": boxes_xyxy,
    }


# ----------------------------- CORE PIPELINE -----------------------------

def run_pose_on_frames(
    frames_dir: str,
    out_dir: str,
    windows_csv: str,
    config: PoseExportConfig,
    pattern: str = "*.png"
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

    weights_path = Path(config.model_path).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"Pose model not found: {weights_path}")

    yolo_is_engine = is_engine_weights_path(weights_path)
    runtime_weights_path = weights_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if yolo_is_engine:
        if not str(device).lower().startswith("cuda"):
            raise RuntimeError(
                f"TensorRT .engine model requires CUDA, but resolved device is '{device}'."
            )
        runtime_weights_path = ensure_ultralytics_engine_header(weights_path)

    try:
        model = YOLO(str(runtime_weights_path), task="pose")
    except TypeError:
        model = YOLO(str(runtime_weights_path))

    if not yolo_is_engine:
        model.to(device)

    yolo_predict_device = resolve_yolo_predict_device(device=device, yolo_is_engine=yolo_is_engine)
    use_half_yolo = bool(yolo_is_engine and str(device).lower().startswith("cuda"))
    print("Using device:", device)
    if yolo_is_engine:
        print(f"Using TensorRT engine: {weights_path}")
        if runtime_weights_path != weights_path:
            print(f"Wrapped engine path: {runtime_weights_path}")

    first = read_image(frame_paths[0])
    h, w = first.shape[:2]

    out_video = os.path.join(out_dir, "pose_out.mp4")
    out_npz = os.path.join(out_dir, "keypoints.npz")
    out_csv = os.path.join(out_dir, "keypoints.csv") if config.save_csv else None

    writer = make_video_writer(out_video, config.fps, (w, h), codec=config.video_codec)

    arrays = init_pose_arrays(num_frames, config.max_people, config.num_kpts)

    csv_rows = []
    if config.save_csv:
        csv_rows.append(["frame", "person", "kpt", "x", "y", "kpt_conf", "person_conf", "frame_path"])
    

    #Get frame labels
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

    target_center = np.array(
        [w * config.target_x_frac, h * config.target_y_frac],
        dtype=np.float32,
    )
    frame_diag = float(np.hypot(float(w), float(h)))
    max_jump_px = float(config.max_jump_px) if config.max_jump_px is not None else float(config.max_jump_diag_frac * frame_diag)
    prev_center: Optional[np.ndarray] = None
    lost_count = 0

    for i, p in enumerate(frame_paths):
        frame = cv2.imread(p)
        if frame is None:
            print(f"Skipping unreadable frame: {p}")
            continue

        # Keep detector candidate count independent from exported people count.
        max_det = max(1, int(config.detector_max_det))
        if yolo_is_engine:
            results = model.predict(
                source=frame,
                conf=config.conf_thres,
                verbose=False,
                device=yolo_predict_device,
                half=use_half_yolo,
                max_det=max_det,
            )
        else:
            results = model.predict(
                source=frame,
                conf=config.conf_thres,
                verbose=False,
                device=yolo_predict_device,
                max_det=max_det,
            )
        r = results[0]
        selected_xy: Optional[np.ndarray] = None
        selected_kc: Optional[np.ndarray] = None
        selected_box_xyxy: Optional[np.ndarray] = None
        selected_person_conf = float("nan")

        pose = extract_pose_for_frame(r)
        if pose is None:
            lost_count += 1
        else:
            xy = pose["xy"]
            kc = pose["kc"]
            box_conf = pose["box_conf"]
            box_centers = pose["box_centers"]
            boxes_xyxy = pose["boxes_xyxy"]

            # Match pose rows to bbox rows before selection.
            num_candidates = min(int(xy.shape[0]), int(box_centers.shape[0]), int(boxes_xyxy.shape[0]))
            if num_candidates <= 0:
                lost_count += 1
            else:
                xy = xy[:num_candidates]
                box_centers = box_centers[:num_candidates]
                boxes_xyxy = boxes_xyxy[:num_candidates]
                if box_conf is not None:
                    box_conf = box_conf[:num_candidates]
                if kc is not None and kc.ndim >= 2:
                    kc = kc[:num_candidates]
                else:
                    kc = None

                idx, new_center = select_person_idx(
                    box_centers=box_centers,
                    box_conf=box_conf,
                    prev_center=prev_center,
                    target_center=target_center,
                    conf_min=config.conf_min,
                    max_jump_px=max_jump_px,
                    switch_margin_px=config.switch_margin_px,
                )

                if idx is None:
                    lost_count += 1
                else:
                    prev_center = new_center
                    lost_count = 0

                    selected_xy = xy[idx]
                    selected_box_xyxy = boxes_xyxy[idx]
                    if kc is not None and idx < kc.shape[0]:
                        selected_kc = kc[idx]

                    if box_conf is not None and idx < box_conf.shape[0] and np.isfinite(box_conf[idx]):
                        selected_person_conf = float(box_conf[idx])
                    elif selected_kc is not None:
                        selected_person_conf = float(np.nanmean(selected_kc))

                    if config.max_people > 0:
                        j = 0
                        xy_sel = np.asarray(selected_xy, dtype=np.float32)
                        xy_count = min(config.num_kpts, int(xy_sel.shape[0]))
                        arrays["kpts_xy"][i, j, :xy_count] = xy_sel[:xy_count]

                        if selected_kc is not None:
                            kc_sel = np.asarray(selected_kc, dtype=np.float32).reshape(-1)
                            kc_count = min(config.num_kpts, int(kc_sel.shape[0]))
                            arrays["kpts_conf"][i, j, :kc_count] = kc_sel[:kc_count]

                        arrays["person_conf"][i, j] = selected_person_conf

                        if config.save_csv:
                            for k in range(config.num_kpts):
                                x, y = arrays["kpts_xy"][i, j, k]
                                kconf = arrays["kpts_conf"][i, j, k]
                                pconf = arrays["person_conf"][i, j]
                                csv_rows.append([i, j, k, float(x), float(y), float(kconf), float(pconf), p])

        if config.reset_on_max_lost and lost_count > config.max_lost:
            prev_center = None

        annotated = draw_selected_pose(
            frame=frame,
            kpts_xy=selected_xy,
            kpts_conf=selected_kc,
            box_xyxy=selected_box_xyxy,
            draw_kpt_threshold=config.draw_kpt_threshold,
            draw_no_target_text=config.draw_no_target_text,
        )
        writer.write(annotated)

    writer.release()

    #Save to .npz file
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
        model_path=np.array([config.model_path], dtype=object),
    )

    if config.save_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerows(csv_rows)

    return out_video, out_npz, out_csv

def load_window_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=1, encoding="utf-8-sig")

    # Clean column names
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

def find_camera_folders(root: str, camera: int = 1) -> List[str]:
    pat = os.path.join(root, "**", f"*Camera{camera}")
    folders = [p for p in glob.glob(pat, recursive=True) if os.path.isdir(p)]
    folders = [p for p in folders if re.search(rf"Camera{camera}$", os.path.basename(p))]
    return folders


def find_features_csv(trial_dir: str, features_suffix: str = "Features1&0.5.csv") -> str:
    cands = glob.glob(os.path.join(trial_dir, f"*{features_suffix}"))
    if not cands:
        raise FileNotFoundError(f"No *{features_suffix} found in {trial_dir}")
    if len(cands) > 1:
        # usually only one; pick first but warn
        print(f"Warning: multiple feature CSVs in {trial_dir}, using {cands[0]}")
    return cands[0]

def batch_process_upfall_tree(
    upfall_root: str,
    output_root: str,
    config: PoseExportConfig,
    camera: int = 1,
    features_suffix: str = "Features1&0.5.csv",
    pattern: str = "*.png",
) -> List[Tuple[str, str, str]]:
    ensure_dir(output_root)

    cam_folders = find_camera_folders(upfall_root, camera=camera)
    results = []

    for frames_dir in cam_folders:
        trial_dir = os.path.dirname(frames_dir)  # .../TrialZ/
        try:
            windows_csv = find_features_csv(trial_dir, features_suffix=features_suffix)
        except FileNotFoundError as e:
            print(f"Skipping {frames_dir}: {e}")
            continue

        # Make a unique output folder name from the relative path
        rel = os.path.relpath(frames_dir, upfall_root)
        out_dir = os.path.join(output_root, rel)
        ensure_dir(out_dir)

        # Skip if already processed (resume-friendly)
        if os.path.exists(os.path.join(out_dir, "keypoints.npz")):
            continue

        out_video, out_npz, _ = run_pose_on_frames(
            frames_dir=frames_dir,
            out_dir=out_dir,
            windows_csv=windows_csv,
            config=config,
            pattern=pattern,
        )
        results.append((frames_dir, out_video, out_npz))

    return results




# ----------------------------- EXAMPLE USAGE -----------------------------

# if __name__ == "__main__":
#     cfg = PoseExportConfig(
#         model_path="yolo11l-pose.pt",
#         conf_thres=0.25,
#         fps=30,
#         max_people=1,
#         save_csv=False
#     )
#     frames = "../../Datasets/UP-Fall/Subject1Activity1Trial1Camera1"
#     windows_csv = "../../Datasets/UP-Fall/Subject1Activity1Trial1Features1&0.5.csv"

#     # Single folder
#     run_pose_on_frames(frames_dir=frames, out_dir="outputs/frames_run", windows_csv=windows_csv, config=cfg)

#     # Many folders
#     # batch_process_folders(input_root="all_sequences", output_root="outputs", config=cfg)

