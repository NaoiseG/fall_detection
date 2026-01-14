import os
import glob
import re
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
    model_path: str = "Prototype/yolo11l-pose.pt"
    conf_thres: float = 0.25
    fps: int = 30
    max_people: int = 1
    num_kpts: int = 17               # COCO keypoints for Ultralytics pose models
    video_codec: str = "mp4v"        # mp4v is widely supported
    save_csv: bool = False


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


# ----------------------------- DATA STRUCTURES -----------------------------

def init_pose_arrays(num_frames: int, max_people: int, num_kpts: int) -> Dict[str, np.ndarray]:
    kpts_xy = np.full((num_frames, max_people, num_kpts, 2), np.nan, dtype=np.float32)
    kpts_conf = np.full((num_frames, max_people, num_kpts), np.nan, dtype=np.float32)
    person_conf = np.full((num_frames, max_people), np.nan, dtype=np.float32)
    return {"kpts_xy": kpts_xy, "kpts_conf": kpts_conf, "person_conf": person_conf}


def choose_people_order(r) -> np.ndarray:
    """
    Choose ordering for people in frame.
    Prefer box confidence if available, otherwise mean keypoint confidence.
    """
    if r.boxes is not None and r.boxes.conf is not None:
        conf = r.boxes.conf.detach().cpu().numpy()
        return np.argsort(-conf)

    if r.keypoints is not None and r.keypoints.conf is not None:
        kc = r.keypoints.conf.detach().cpu().numpy()
        return np.argsort(-np.nanmean(kc, axis=1))

    return np.array([], dtype=int)


def extract_pose_for_frame(r) -> Optional[Dict[str, Any]]:
    """
    Return dict with:
      xy: (P, K, 2)
      kc: (P, K)
      box_conf: (P,) or None
    """
    if r.keypoints is None:
        return None

    xy = r.keypoints.xy.detach().cpu().numpy()  # (people, kpts, 2)
    kc = None
    if r.keypoints.conf is not None:
        kc = r.keypoints.conf.detach().cpu().numpy()

    box_conf = None
    if r.boxes is not None and r.boxes.conf is not None:
        box_conf = r.boxes.conf.detach().cpu().numpy()

    return {"xy": xy, "kc": kc, "box_conf": box_conf}


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

    model = YOLO(config.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Using device:", device)

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

    for i, p in enumerate(frame_paths):
        frame = cv2.imread(p)
        if frame is None:
            print(f"Skipping unreadable frame: {p}")
            continue

        results = model(frame, conf=config.conf_thres, verbose=False)
        r = results[0]

        annotated = r.plot()
        writer.write(annotated)

        pose = extract_pose_for_frame(r)
        if pose is None:
            continue

        xy = pose["xy"]
        kc = pose["kc"]
        box_conf = pose["box_conf"]

        order = choose_people_order(r)
        for j, idx in enumerate(order[:config.max_people]):
            arrays["kpts_xy"][i, j] = xy[idx].astype(np.float32)

            if kc is not None:
                arrays["kpts_conf"][i, j] = kc[idx].astype(np.float32)
            else:
                arrays["kpts_conf"][i, j] = np.nan

            if box_conf is not None and idx < len(box_conf):
                arrays["person_conf"][i, j] = float(box_conf[idx])
            elif kc is not None:
                arrays["person_conf"][i, j] = float(np.nanmean(kc[idx]))
            else:
                arrays["person_conf"][i, j] = np.nan

            if config.save_csv:
                for k in range(config.num_kpts):
                    x, y = arrays["kpts_xy"][i, j, k]
                    kconf = arrays["kpts_conf"][i, j, k]
                    pconf = arrays["person_conf"][i, j]
                    csv_rows.append([i, j, k, float(x), float(y), float(kconf), float(pconf), p])

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

