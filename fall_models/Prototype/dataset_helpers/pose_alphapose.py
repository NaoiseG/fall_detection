import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch


# ----------------------------- CONFIG -----------------------------

@dataclass
class AlphaPoseExportConfig:
    # AlphaPose repo root (relative to project root by default)
    alphapose_root: str = "pose_models/AlphaPose"

    # FastPose COCO-17 config + checkpoint
    cfg_path: str = "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
    checkpoint: str = "pretrained_models/fast_res50_256x192.pth"

    # YOLOv3-SPP detector files
    detector_cfg: str = "detector/yolo/cfg/yolov3-spp.cfg"
    detector_weights: str = "detector/yolo/data/yolov3-spp.weights"

    conf_thres: float = 0.1
    conf_min: float = 0.75
    nms_thres: float = 0.6
    fps: int = 30
    max_people: int = 1
    num_kpts: int = 17
    video_codec: str = "mp4v"
    save_csv: bool = False
    max_jump_px: Optional[float] = None
    max_jump_diag_frac: float = 0.25
    max_lost: int = 10
    target_x_frac: float = 0.5
    target_y_frac: float = 0.5
    draw_kpt_threshold: float = 0.30
    draw_no_target_text: bool = True
    draw_confidence_text: bool = True
    min_box_area: int = 0
    flip: bool = False
    vis_fast: bool = True
    render_video: bool = True
    device: Optional[str] = None  # "cuda" or "cpu"


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


def _coerce_int_id(value: Any) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0
        return int(value.view(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0
        return _coerce_int_id(value[0])
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_ids(ids: Any, count: int) -> List[int]:
    if ids is None:
        return [0] * count
    if torch.is_tensor(ids):
        if ids.numel() == 0:
            return [0] * count
        values = [_coerce_int_id(v) for v in ids.view(-1)]
    elif isinstance(ids, np.ndarray):
        if ids.size == 0:
            return [0] * count
        values = [_coerce_int_id(v) for v in ids.reshape(-1)]
    elif isinstance(ids, (list, tuple)):
        values = [_coerce_int_id(v) for v in ids]
    else:
        values = [_coerce_int_id(ids)]

    if len(values) < count:
        values.extend([0] * (count - len(values)))
    elif len(values) > count:
        values = values[:count]
    return values


# ----------------------------- LABELS -----------------------------

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


def select_person_idx(
    box_centers: np.ndarray,
    box_conf: Optional[np.ndarray],
    prev_center: Optional[np.ndarray],
    target_center: np.ndarray,
    conf_min: float,
    max_jump_px: float,
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    Temporal target selection:
      - Acquire (no prev_center): prefer conf >= conf_min, closest to target center.
      - Track (has prev_center): closest to prev_center with max-jump gate.
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

    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    if not np.isfinite(best_dist) or best_dist > max_jump_px:
        return None, prev_center

    return best_idx, box_centers[best_idx].astype(np.float32, copy=True)


def draw_selected_pose(
    frame: np.ndarray,
    kpts_xy: Optional[np.ndarray],
    kpts_conf: Optional[np.ndarray],
    box_xyxy: Optional[np.ndarray],
    person_conf: float,
    draw_kpt_threshold: float,
    draw_no_target_text: bool,
    draw_confidence_text: bool,
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

    text_origin = (12, 30)
    if box_xyxy is not None:
        box_xyxy = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)
        if box_xyxy.shape[0] >= 4 and np.all(np.isfinite(box_xyxy[:4])):
            x1, y1, x2, y2 = np.round(box_xyxy[:4]).astype(int).tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text_origin = (x1, max(20, y1 - 8))

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

    if draw_confidence_text and np.isfinite(person_conf):
        cv2.putText(
            out,
            f"CONF {float(person_conf):.3f}",
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


# ----------------------------- ALPHAPOSE WRAPPER -----------------------------

class AlphaPoseRunner:
    def __init__(self, config: AlphaPoseExportConfig):
        self.config = config
        self.root = Path(config.alphapose_root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"AlphaPose root not found: {self.root}")

        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        # Resolve model and detector paths to absolute locations
        self.cfg_path = self._resolve_path(config.cfg_path)
        self.checkpoint_path = self._resolve_path(config.checkpoint)
        self.detector_cfg_path = self._resolve_path(config.detector_cfg)
        self.detector_weights_path = self._resolve_path(config.detector_weights)

        if not Path(self.cfg_path).exists():
            raise FileNotFoundError(f"AlphaPose cfg not found: {self.cfg_path}")
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"AlphaPose checkpoint not found: {self.checkpoint_path}")
        if not Path(self.detector_cfg_path).exists():
            raise FileNotFoundError(f"YOLOv3-SPP cfg not found: {self.detector_cfg_path}")
        if not Path(self.detector_weights_path).exists():
            raise FileNotFoundError(
                f"YOLOv3-SPP weights not found: {self.detector_weights_path}. "
                "Download yolov3-spp.weights and place it at that path."
            )

        from alphapose.utils.config import update_config
        from alphapose.models import builder
        from alphapose.utils.transforms import flip, flip_heatmap, get_func_heatmap_to_coord
        from alphapose.utils.pPose_nms import pose_nms
        from alphapose.utils.presets import SimpleTransform
        from detector.apis import get_detector
        from detector import yolo_cfg as yolo_cfg_mod

        self.flip = flip
        self.flip_heatmap = flip_heatmap
        self.pose_nms = pose_nms
        self.SimpleTransform = SimpleTransform

        self.cfg = update_config(self.cfg_path)

        # Override detector cfg to absolute paths and requested thresholds
        yolo_cfg_mod.cfg.CONFIG = self.detector_cfg_path
        yolo_cfg_mod.cfg.WEIGHTS = self.detector_weights_path
        yolo_cfg_mod.cfg.CONFIDENCE = float(config.conf_thres)
        yolo_cfg_mod.cfg.NMS_THRES = float(config.nms_thres)

        # Device selection
        if config.device is None:
            use_cuda = torch.cuda.is_available()
        else:
            use_cuda = config.device.lower() == "cuda"
        if use_cuda and not torch.cuda.is_available():
            use_cuda = False

        gpus = [0] if use_cuda else [-1]
        device = torch.device(f"cuda:{gpus[0]}" if gpus[0] >= 0 else "cpu")

        # Minimal opt namespace required by AlphaPose detector/vis
        self.opt = _Opt(
            detector="yolo",
            gpus=gpus,
            device=device,
            tracking=False,
            pose_track=False,
            pose_flow=False,
            min_box_area=config.min_box_area,
            flip=config.flip,
            vis_fast=config.vis_fast,
            showbox=False,
        )

        # Build detector + pose model
        self.detector = get_detector(self.opt)
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(self.checkpoint_path, map_location=device))
        self.pose_model.to(device).eval()

        self.pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)
        self.norm_type = self.cfg.LOSS.get("NORM_TYPE", None)
        self.hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self.eval_joints = list(range(self.cfg.DATA_PRESET.NUM_JOINTS))

        if config.num_kpts != self.cfg.DATA_PRESET.NUM_JOINTS:
            raise ValueError(
                f"config.num_kpts={config.num_kpts} does not match model joints "
                f"{self.cfg.DATA_PRESET.NUM_JOINTS}"
            )

        # Preprocess transform for crops
        self.transformation = self.SimpleTransform(
            self.pose_dataset,
            scale_factor=0,
            input_size=self.cfg.DATA_PRESET.IMAGE_SIZE,
            output_size=self.cfg.DATA_PRESET.HEATMAP_SIZE,
            rot=0,
            sigma=self.cfg.DATA_PRESET.SIGMA,
            train=False,
            add_dpg=False,
            gpu_device=device,
        )

        loss_type = self.cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss")
        num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
        self.hand_face_num = None
        if loss_type == "MSELoss":
            self.vis_thres = [0.4] * num_joints
        elif "JointRegression" in loss_type:
            self.vis_thres = [0.05] * num_joints
        elif loss_type == "Combined":
            if num_joints == 68:
                self.hand_face_num = 42
            else:
                self.hand_face_num = 110
            self.vis_thres = [0.4] * (num_joints - self.hand_face_num) + [0.05] * self.hand_face_num

        self.use_heatmap_loss = (loss_type == "MSELoss")

    def _resolve_path(self, p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str(self.root / path)

    def infer(self, image_rgb: np.ndarray, image_name: str) -> Dict[str, Any]:
        """
        Returns a dict with:
          - im_res: AlphaPose-style result for visualization
          - people: list of per-person dicts with numpy keypoints for export
        """
        if image_rgb is None:
            return {"im_res": {"imgname": image_name, "result": []}, "people": []}

        img = self.detector.image_preprocess(image_rgb)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        im_dim = image_rgb.shape[1], image_rgb.shape[0]
        im_dim_list = torch.FloatTensor(im_dim).repeat(1, 2)

        with torch.no_grad():
            dets = self.detector.images_detection(img, im_dim_list)
            if isinstance(dets, int) or dets is None or dets.shape[0] == 0:
                return {"im_res": {"imgname": image_name, "result": []}, "people": []}
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()

            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)

            boxes = boxes[dets[:, 0] == 0]
            scores = scores[dets[:, 0] == 0]
            ids = ids[dets[:, 0] == 0]
            if boxes.nelement() == 0:
                return {"im_res": {"imgname": image_name, "result": []}, "people": []}

            inps = torch.zeros(boxes.size(0), 3, *self.cfg.DATA_PRESET.IMAGE_SIZE)
            cropped_boxes = torch.zeros(boxes.size(0), 4)

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(image_rgb, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            inps = inps.to(self.opt.device)
            if self.config.flip:
                inps = torch.cat((inps, self.flip(inps)))

            hm = self.pose_model(inps)
            if self.config.flip:
                hm_flip = self.flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
            hm = hm.cpu()

            pose_coords = []
            pose_scores = []
            for i in range(hm.shape[0]):
                bbox = cropped_boxes[i].tolist()
                if isinstance(self.heatmap_to_coord, list):
                    if self.hand_face_num is None:
                        raise ValueError("Combined loss requires hand_face_num to be set")
                    pose_coords_body, pose_scores_body = self.heatmap_to_coord[0](
                        hm[i][self.eval_joints[:-self.hand_face_num]],
                        bbox,
                        hm_shape=self.hm_size,
                        norm_type=self.norm_type,
                    )
                    pose_coords_fh, pose_scores_fh = self.heatmap_to_coord[1](
                        hm[i][self.eval_joints[-self.hand_face_num:]],
                        bbox,
                        hm_shape=self.hm_size,
                        norm_type=self.norm_type,
                    )
                    pose_coord = np.concatenate((pose_coords_body, pose_coords_fh), axis=0)
                    pose_score = np.concatenate((pose_scores_body, pose_scores_fh), axis=0)
                else:
                    pose_coord, pose_score = self.heatmap_to_coord(
                        hm[i][self.eval_joints], bbox, hm_shape=self.hm_size, norm_type=self.norm_type
                    )
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            boxes, scores, ids, preds_img, preds_scores, _ = self.pose_nms(
                boxes, scores, ids, preds_img, preds_scores, self.config.min_box_area,
                use_heatmap_loss=self.use_heatmap_loss
            )
            norm_ids = _normalize_ids(ids, len(scores))

            result_list = []
            people = []
            for k in range(len(scores)):
                kp = preds_img[k]
                kp_score = preds_scores[k]
                proposal_score = torch.mean(kp_score) + scores[k] + 1.25 * torch.max(kp_score)
                box = boxes[k]
                box_xyxy = np.asarray(box.cpu().numpy() if torch.is_tensor(box) else box, dtype=np.float32).reshape(-1)
                if box_xyxy.shape[0] < 4:
                    continue
                box_xyxy = box_xyxy[:4]
                score_k = scores[k]
                if torch.is_tensor(score_k):
                    score_k_val = float(score_k.reshape(-1)[0].item())
                else:
                    score_k_val = float(score_k)
                box_xywh = [
                    float(box_xyxy[0]),
                    float(box_xyxy[1]),
                    float(box_xyxy[2] - box_xyxy[0]),
                    float(box_xyxy[3] - box_xyxy[1]),
                ]
                result_list.append(
                    {
                        "keypoints": kp,
                        "kp_score": kp_score,
                        "proposal_score": float(proposal_score),
                        "idx": norm_ids[k],
                        "box": box_xywh,
                    }
                )

                people.append(
                    {
                        "kpts_xy": kp.cpu().numpy().astype(np.float32),
                        "kpts_conf": kp_score.cpu().numpy().astype(np.float32).reshape(-1),
                        "box_xyxy": box_xyxy.astype(np.float32, copy=False),
                        "box_center": np.array(
                            [
                                0.5 * (box_xyxy[0] + box_xyxy[2]),
                                0.5 * (box_xyxy[1] + box_xyxy[3]),
                            ],
                            dtype=np.float32,
                        ),
                        "box_conf": score_k_val,
                        "person_conf": float(proposal_score),
                    }
                )

            im_res = {"imgname": image_name, "result": result_list}
            return {"im_res": im_res, "people": people}

    def render(self, frame_bgr: np.ndarray, im_res: Dict[str, Any]) -> np.ndarray:
        if not im_res or not im_res.get("result"):
            return frame_bgr

        if self.config.vis_fast:
            from alphapose.utils.vis import vis_frame_fast as vis_frame
        else:
            from alphapose.utils.vis import vis_frame

        return vis_frame(frame_bgr, im_res, self.opt, self.vis_thres.copy())


class _Opt:
    def __init__(
        self,
        detector: str,
        gpus: List[int],
        device: torch.device,
        tracking: bool,
        pose_track: bool,
        pose_flow: bool,
        min_box_area: int,
        flip: bool,
        vis_fast: bool,
        showbox: bool,
    ):
        self.detector = detector
        self.gpus = gpus
        self.device = device
        self.tracking = tracking
        self.pose_track = pose_track
        self.pose_flow = pose_flow
        self.min_box_area = min_box_area
        self.flip = flip
        self.vis_fast = vis_fast
        self.showbox = showbox
        self.sp = True
        self.save_img = False
        self.vis = False
        self.format = None
        self.eval = False


# ----------------------------- CORE PIPELINE -----------------------------

def run_pose_on_frames_alphapose(
    frames_dir: str,
    out_dir: str,
    windows_csv: str,
    config: AlphaPoseExportConfig,
    pattern: str = "*.png",
    runner: Optional[AlphaPoseRunner] = None,
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
        runner = AlphaPoseRunner(config)

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

    target_center = np.array(
        [w * config.target_x_frac, h * config.target_y_frac],
        dtype=np.float32,
    )
    frame_diag = float(np.hypot(float(w), float(h)))
    max_jump_px = (
        float(config.max_jump_px)
        if config.max_jump_px is not None
        else float(config.max_jump_diag_frac * frame_diag)
    )
    prev_center: Optional[np.ndarray] = None
    lost_count = 0

    for i, p in enumerate(frame_paths):
        frame_bgr = cv2.imread(p)
        if frame_bgr is None:
            print(f"Skipping unreadable frame: {p}")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_name = os.path.basename(p)
        out = runner.infer(frame_rgb, image_name=image_name)
        people = out["people"]
        selected_xy: Optional[np.ndarray] = None
        selected_kc: Optional[np.ndarray] = None
        selected_box_xyxy: Optional[np.ndarray] = None
        selected_person_conf = float("nan")

        if not people:
            lost_count += 1
        else:
            xy = np.asarray([person["kpts_xy"] for person in people], dtype=np.float32)
            kc = np.asarray([person["kpts_conf"] for person in people], dtype=np.float32)
            box_centers = np.asarray([person["box_center"] for person in people], dtype=np.float32)
            boxes_xyxy = np.asarray([person["box_xyxy"] for person in people], dtype=np.float32)
            box_conf = np.asarray(
                [float(person.get("box_conf", np.nan)) for person in people],
                dtype=np.float32,
            )

            num_candidates = min(
                int(xy.shape[0]),
                int(kc.shape[0]),
                int(box_centers.shape[0]),
                int(boxes_xyxy.shape[0]),
            )
            if num_candidates <= 0:
                lost_count += 1
            else:
                xy = xy[:num_candidates]
                kc = kc[:num_candidates]
                box_centers = box_centers[:num_candidates]
                boxes_xyxy = boxes_xyxy[:num_candidates]
                box_conf = box_conf[:num_candidates]

                idx, new_center = select_person_idx(
                    box_centers=box_centers,
                    box_conf=box_conf,
                    prev_center=prev_center,
                    target_center=target_center,
                    conf_min=config.conf_min,
                    max_jump_px=max_jump_px,
                )

                if idx is None:
                    lost_count += 1
                else:
                    prev_center = new_center
                    lost_count = 0

                    selected_xy = xy[idx]
                    selected_kc = kc[idx]
                    selected_box_xyxy = boxes_xyxy[idx]

                    if idx < box_conf.shape[0] and np.isfinite(box_conf[idx]):
                        selected_person_conf = float(box_conf[idx])
                    elif selected_kc is not None:
                        selected_person_conf = float(np.nanmean(selected_kc))
                    elif idx < len(people):
                        selected_person_conf = float(people[idx].get("person_conf", np.nan))

                    if config.max_people > 0:
                        j = 0
                        xy_sel = np.asarray(selected_xy, dtype=np.float32)
                        xy_count = min(config.num_kpts, int(xy_sel.shape[0]))
                        arrays["kpts_xy"][i, j, :xy_count] = xy_sel[:xy_count]

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

        if lost_count > config.max_lost:
            prev_center = None

        if config.render_video:
            annotated = draw_selected_pose(
                frame=frame_bgr,
                kpts_xy=selected_xy,
                kpts_conf=selected_kc,
                box_xyxy=selected_box_xyxy,
                person_conf=selected_person_conf,
                draw_kpt_threshold=config.draw_kpt_threshold,
                draw_no_target_text=config.draw_no_target_text,
                draw_confidence_text=config.draw_confidence_text,
            )
            writer.write(annotated)
        else:
            writer.write(frame_bgr)

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
        model_path=np.array([str(Path(config.checkpoint))], dtype=object),
    )

    if config.save_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerows(csv_rows)

    return out_video, out_npz, out_csv
