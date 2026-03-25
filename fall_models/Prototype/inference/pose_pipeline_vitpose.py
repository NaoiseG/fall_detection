from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dataset_helpers.get_keypoints_files_ViTpose import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_POSE_MODEL,
    VitPoseExportConfig,
    VitPoseRunner,
    normalize_model_source,
)
from inference.pose_pipeline import K, PoseFrameOutput, select_person_idx


def _maybe_cuda_sync(sync_cuda: bool) -> None:
    if bool(sync_cuda) and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class VitPosePipelineConfig:
    device: str

    detector_model: str = DEFAULT_DETECTOR_MODEL
    detector_processor: Optional[str] = None
    pose_model: str = DEFAULT_POSE_MODEL
    pose_processor: Optional[str] = None
    vitpose_conf: float = 0.25
    vitpose_pose_threshold: float = 0.30
    max_det: int = 10

    frame_step: int = 1
    track_conf_min: float = 0.75
    track_max_jump_px: float = 0.0
    track_max_jump_diag_frac: float = 0.25
    track_max_lost: int = 10
    track_target_x_frac: float = 0.5
    track_target_y_frac: float = 0.5


class VitPosePipeline:
    """
    ViTPose-backed pose pipeline with the same output contract and tracking
    semantics as the shared YOLO/AlphaPose benchmark paths.
    """

    def __init__(self, config: VitPosePipelineConfig) -> None:
        self.config = config
        self.detector_model_source = normalize_model_source(str(config.detector_model))
        self.pose_model_source = normalize_model_source(str(config.pose_model))
        detector_processor = (
            normalize_model_source(str(config.detector_processor))
            if config.detector_processor is not None and str(config.detector_processor).strip()
            else None
        )
        pose_processor = (
            normalize_model_source(str(config.pose_processor))
            if config.pose_processor is not None and str(config.pose_processor).strip()
            else None
        )
        runner_device = "cuda" if str(config.device).lower().startswith("cuda") else "cpu"

        runner_cfg = VitPoseExportConfig(
            detector_model=str(self.detector_model_source),
            detector_processor=detector_processor,
            pose_model=str(self.pose_model_source),
            pose_processor=pose_processor,
            person_threshold=float(config.vitpose_conf),
            pose_threshold=float(config.vitpose_pose_threshold),
            fps=30,
            max_people=1,
            detector_max_det=int(config.max_det) if int(config.max_det) > 0 else 1_000_000,
            num_kpts=int(K),
            save_csv=False,
            render_video=False,
            device=str(runner_device),
        )
        self._runner = VitPoseRunner(runner_cfg)
        self.reset_tracking_state()

    @property
    def detector_model_path(self) -> Path:
        return Path(str(self.detector_model_source))

    @property
    def pose_model_path(self) -> Path:
        return Path(str(self.pose_model_source))

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
            raise RuntimeError("ViTPose pipeline geometry failed to initialize.")

        _maybe_cuda_sync(sync_cuda_timing)
        t_pose0 = time.perf_counter()
        people = self._runner.infer(frame_bgr)
        _maybe_cuda_sync(sync_cuda_timing)
        pose_infer_ms = (time.perf_counter() - t_pose0) * 1000.0

        t_track0 = time.perf_counter()
        xy_zeros = np.zeros((K, 2), dtype=np.float32)
        cf_zeros = np.zeros((K,), dtype=np.float32)

        found = False
        out_xy = xy_zeros
        out_cf = cf_zeros

        if people:
            xy_all = np.asarray([person.get("kpts_xy") for person in people], dtype=np.float32)
            cf_all = np.asarray([person.get("kpts_conf") for person in people], dtype=np.float32)
            box_centers = np.asarray([person.get("box_center") for person in people], dtype=np.float32)
            box_conf = np.asarray([person.get("box_conf", np.nan) for person in people], dtype=np.float32).reshape(-1)

            if xy_all.ndim == 3 and int(xy_all.shape[0]) > 0 and int(xy_all.shape[1]) == int(K):
                num_candidates = int(xy_all.shape[0])

                if box_centers.ndim == 2 and int(box_centers.shape[1]) == 2:
                    num_candidates = min(num_candidates, int(box_centers.shape[0]))
                else:
                    box_centers = np.mean(xy_all, axis=1).astype(np.float32, copy=False)

                if cf_all.ndim == 2:
                    num_candidates = min(num_candidates, int(cf_all.shape[0]))
                else:
                    cf_all = None

                num_candidates = min(num_candidates, int(box_conf.shape[0]))

                if num_candidates > 0:
                    xy_all = xy_all[:num_candidates]
                    box_centers = box_centers[:num_candidates]
                    box_conf = box_conf[:num_candidates]
                    if cf_all is not None:
                        cf_all = cf_all[:num_candidates]

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
