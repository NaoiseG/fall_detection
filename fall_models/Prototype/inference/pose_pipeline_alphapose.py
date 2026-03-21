from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from dataset_helpers.pose_alphapose import AlphaPoseExportConfig, AlphaPoseRunner
from inference.pose_pipeline import K, PoseFrameOutput, select_person_idx


def _maybe_cuda_sync(sync_cuda: bool) -> None:
    if bool(sync_cuda) and torch.cuda.is_available():
        torch.cuda.synchronize()


def _to_numpy_or_none(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


@dataclass
class AlphaPosePipelineConfig:
    alphapose_root: Path
    device: str

    cfg_path: str = "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
    checkpoint: str = "pretrained_models/fast_res50_256x192.pth"
    detector_cfg: str = "detector/yolo/cfg/yolov3-spp.cfg"
    detector_weights: str = "detector/yolo/data/yolov3-spp.weights"
    alphapose_conf: float = 0.10
    alphapose_nms: float = 0.60
    min_box_area: int = 0
    flip: bool = False
    max_det: int = 0

    frame_step: int = 1
    track_conf_min: float = 0.75
    track_max_jump_px: float = 0.0
    track_max_jump_diag_frac: float = 0.25
    track_max_lost: int = 10
    track_target_x_frac: float = 0.5
    track_target_y_frac: float = 0.5


class AlphaPosePipeline:
    """
    AlphaPose-backed pose pipeline with the same output contract as PosePipeline.
    Tracking semantics intentionally mirror the shared YOLO path for cleaner
    detector-backbone comparisons in benchmark runs.
    """

    def __init__(self, config: AlphaPosePipelineConfig) -> None:
        self.config = config
        runner_device = "cuda" if str(config.device).lower().startswith("cuda") else "cpu"

        runner_cfg = AlphaPoseExportConfig(
            alphapose_root=str(Path(config.alphapose_root).expanduser()),
            cfg_path=str(config.cfg_path),
            checkpoint=str(config.checkpoint),
            detector_cfg=str(config.detector_cfg),
            detector_weights=str(config.detector_weights),
            conf_thres=float(config.alphapose_conf),
            conf_min=float(config.track_conf_min),
            nms_thres=float(config.alphapose_nms),
            fps=30,
            max_people=1,
            num_kpts=int(K),
            save_csv=False,
            min_box_area=int(config.min_box_area),
            flip=bool(config.flip),
            vis_fast=True,
            render_video=False,
            device=str(runner_device),
        )
        self._runner = AlphaPoseRunner(runner_cfg)
        self.reset_tracking_state()

    @property
    def checkpoint_path(self) -> Path:
        return Path(self._runner.checkpoint_path)

    @property
    def detector_weights_path(self) -> Path:
        return Path(self._runner.detector_weights_path)

    def reset_tracking_state(self) -> None:
        self._sampled_count = 0
        self._last_xy = np.zeros((K, 2), dtype=np.float32)
        self._last_cf = np.zeros((K,), dtype=np.float32)
        self._prev_center: Optional[np.ndarray] = None
        self._target_center: Optional[np.ndarray] = None
        self._lost_count = 0
        self._max_jump_px_runtime: Optional[float] = None

    def _init_geometry_if_needed(self, frame: np.ndarray) -> None:
        if self._target_center is not None and self._max_jump_px_runtime is not None:
            return

        h, w = frame.shape[:2]
        frame_diag = float(np.hypot(float(w), float(h)))
        self._target_center = np.array(
            [
                float(self.config.track_target_x_frac) * float(w),
                float(self.config.track_target_y_frac) * float(h),
            ],
            dtype=np.float32,
        )

        if float(self.config.track_max_jump_px) > 0.0:
            self._max_jump_px_runtime = float(self.config.track_max_jump_px)
        else:
            self._max_jump_px_runtime = float(self.config.track_max_jump_diag_frac) * float(frame_diag)

    def _limit_candidates(
        self,
        xy_all: np.ndarray,
        cf_all: Optional[np.ndarray],
        box_centers: np.ndarray,
        box_conf: Optional[np.ndarray],
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        max_det = int(self.config.max_det)
        num_candidates = int(xy_all.shape[0])
        if max_det <= 0 or num_candidates <= max_det:
            return xy_all, cf_all, box_centers, box_conf

        if box_conf is not None and int(box_conf.shape[0]) >= num_candidates:
            scores = np.nan_to_num(box_conf[:num_candidates], nan=-1.0, neginf=-1.0, posinf=1.0e6)
        elif cf_all is not None and cf_all.ndim == 2 and int(cf_all.shape[0]) >= num_candidates:
            scores = np.nanmean(np.clip(cf_all[:num_candidates], 0.0, 1.0), axis=1)
        else:
            scores = np.zeros((num_candidates,), dtype=np.float32)

        keep = np.argsort(scores)[::-1][:max_det]
        keep = np.asarray(keep, dtype=np.int64)
        xy_limited = xy_all[keep]
        centers_limited = box_centers[keep]
        cf_limited = cf_all[keep] if cf_all is not None else None
        conf_limited = box_conf[keep] if box_conf is not None else None
        return xy_limited, cf_limited, centers_limited, conf_limited

    def process_frame(self, frame_bgr: np.ndarray, raw_frame_idx: int, sync_cuda_timing: bool = False) -> PoseFrameOutput:
        do_sample = (int(raw_frame_idx) % int(self.config.frame_step)) == 0
        if not do_sample:
            return PoseFrameOutput(
                raw_frame_idx=int(raw_frame_idx),
                sampled=False,
                sampled_frame_idx=None,
                found=bool(np.any(self._last_cf > 0.0)),
                keypoints_xy=self._last_xy.copy(),
                keypoints_conf=self._last_cf.copy(),
                pose_infer_ms=0.0,
                track_ms=0.0,
            )

        self._init_geometry_if_needed(frame_bgr)
        target_center = self._target_center
        max_jump_px = self._max_jump_px_runtime
        if target_center is None or max_jump_px is None:
            raise RuntimeError("AlphaPose pipeline geometry failed to initialize.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_name = f"frame_{int(raw_frame_idx):06d}.png"

        _maybe_cuda_sync(sync_cuda_timing)
        t_pose0 = time.perf_counter()
        out = self._runner.infer(frame_rgb, image_name=image_name)
        _maybe_cuda_sync(sync_cuda_timing)
        pose_infer_ms = (time.perf_counter() - t_pose0) * 1000.0

        t_track0 = time.perf_counter()
        xy_zeros = np.zeros((K, 2), dtype=np.float32)
        cf_zeros = np.zeros((K,), dtype=np.float32)

        found = False
        out_xy = xy_zeros
        out_cf = cf_zeros

        people = out.get("people", []) if isinstance(out, dict) else []
        if people:
            xy_all = _to_numpy_or_none([person.get("kpts_xy") for person in people])
            cf_all = _to_numpy_or_none([person.get("kpts_conf") for person in people])
            box_centers = _to_numpy_or_none([person.get("box_center") for person in people])
            box_conf = _to_numpy_or_none([person.get("box_conf", np.nan) for person in people])

            if xy_all is not None:
                xy_all = np.asarray(xy_all, dtype=np.float32)
            if cf_all is not None:
                cf_all = np.asarray(cf_all, dtype=np.float32)
            if box_centers is not None:
                box_centers = np.asarray(box_centers, dtype=np.float32)
            if box_conf is not None:
                box_conf = np.asarray(box_conf, dtype=np.float32).reshape(-1)

            if (
                xy_all is not None
                and xy_all.ndim == 3
                and int(xy_all.shape[0]) > 0
                and int(xy_all.shape[1]) == int(K)
            ):
                num_candidates = int(xy_all.shape[0])

                if box_centers is not None and box_centers.ndim == 2 and int(box_centers.shape[1]) == 2:
                    num_candidates = min(num_candidates, int(box_centers.shape[0]))
                else:
                    box_centers = np.mean(xy_all, axis=1).astype(np.float32, copy=False)

                if cf_all is not None and cf_all.ndim == 2:
                    num_candidates = min(num_candidates, int(cf_all.shape[0]))
                else:
                    cf_all = None

                if box_conf is not None:
                    num_candidates = min(num_candidates, int(box_conf.shape[0]))

                if num_candidates > 0:
                    xy_all = xy_all[:num_candidates]
                    box_centers = box_centers[:num_candidates]
                    if cf_all is not None:
                        cf_all = cf_all[:num_candidates]
                    if box_conf is not None:
                        box_conf = box_conf[:num_candidates]

                    xy_all, cf_all, box_centers, box_conf = self._limit_candidates(
                        xy_all=xy_all,
                        cf_all=cf_all,
                        box_centers=box_centers,
                        box_conf=box_conf,
                    )

                    idx, new_center = select_person_idx(
                        box_centers=box_centers,
                        box_conf=box_conf,
                        prev_center=self._prev_center,
                        target_center=target_center,
                        conf_min=float(self.config.track_conf_min),
                        max_jump_px=float(max_jump_px),
                    )
                    if idx is not None:
                        found = True
                        out_xy = xy_all[idx].astype(np.float32, copy=False)
                        if cf_all is not None and idx < cf_all.shape[0] and int(cf_all[idx].shape[0]) == int(K):
                            out_cf = np.clip(cf_all[idx].astype(np.float32, copy=False), 0.0, 1.0)
                        else:
                            out_cf = np.ones((K,), dtype=np.float32)
                        self._prev_center = new_center

        if found:
            self._lost_count = 0
        else:
            self._lost_count += 1
            if int(self._lost_count) > int(self.config.track_max_lost):
                self._prev_center = None

        track_ms = (time.perf_counter() - t_track0) * 1000.0

        self._last_xy = out_xy.copy()
        self._last_cf = out_cf.copy()

        sampled_idx = int(self._sampled_count)
        self._sampled_count += 1

        return PoseFrameOutput(
            raw_frame_idx=int(raw_frame_idx),
            sampled=True,
            sampled_frame_idx=int(sampled_idx),
            found=bool(found),
            keypoints_xy=out_xy,
            keypoints_conf=out_cf,
            pose_infer_ms=float(pose_infer_ms),
            track_ms=float(track_ms),
        )
