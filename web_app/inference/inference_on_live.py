"""
Live webcam -> pose model -> temporal classifier inference.

Shared model-loading and inference utilities are imported directly from
inference_on_video.py to keep this file minimal.

Temporal sampling: the sample buffer is time-stamped at the selected FPS
(default 18 fps).  Windows cover T / selected_fps seconds.  If live processing
cannot sustain the selected FPS, inference raises an error instead of silently
falling behind.
"""

from __future__ import annotations

import base64
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _sleep_until(target_t: float) -> None:
    """Sleep until an absolute perf_counter timestamp with reduced overshoot."""
    while True:
        remaining_s = float(target_t) - time.perf_counter()
        if remaining_s <= 0.0:
            return
        if remaining_s > 0.002:
            time.sleep(max(0.0, remaining_s - 0.001))
        else:
            time.sleep(0)


# Allow running from any working directory.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.inference_on_video import (  # noqa: E402
    AsyncVideoWriter,
    K,
    SKELETON,
    build_temporal_model,
    clean_state_dict,
    draw_hud,
    draw_pose,
    expected_in_features,
    infer_one_window,
    load_checkpoint,
    load_class_names,
    make_window_features,
    open_video_writer,
    pick_device,
    pose_on_frame,
    resolve_ckpt_and_arch,
)

# FPS the classifiers were trained on.  Windows cover T/TRAINING_FPS seconds.
TRAINING_FPS: float = 18.0


# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------

def list_available_cameras(max_index: int = 4) -> List[Dict[str, Any]]:
    """Probe camera indices 0..max_index and return the ones that work.

    Uses the DirectShow backend on Windows for faster probing (avoids the
    multi-second timeout on missing indices with the default MSMF backend).
    """
    if sys.platform.startswith("linux") and not any(Path("/dev").glob("video*")):
        return []

    cameras: List[Dict[str, Any]] = []
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append(
                    {
                        "index": idx,
                        "label": f"Camera {idx}",
                        "width": w,
                        "height": h,
                    }
                )
        cap.release()
    return cameras


# ---------------------------------------------------------------------------
# Temporal interpolation helper
# ---------------------------------------------------------------------------

def _interpolate_temporal(
    target_times: np.ndarray,
    buf_times: np.ndarray,
    buf_xy: np.ndarray,
    buf_cf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate keypoints to T uniformly-spaced time points.

    Args:
        target_times: (T,) desired output timestamps.
        buf_times:    (N,) timestamps of buffered samples (sorted ascending).
        buf_xy:       (N, K, 2) keypoint coordinates.
        buf_cf:       (N, K)    keypoint confidences.

    Returns:
        xy_out: (T, K, 2), cf_out: (T, K)
    """
    T = len(target_times)
    K_joints = buf_xy.shape[1]
    xy_out = np.empty((T, K_joints, 2), dtype=np.float32)
    cf_out = np.empty((T, K_joints), dtype=np.float32)

    for i, t in enumerate(target_times):
        idx = int(np.searchsorted(buf_times, t))
        idx = max(1, min(idx, len(buf_times) - 1))
        t0, t1 = buf_times[idx - 1], buf_times[idx]
        dt = t1 - t0
        alpha = 0.0 if dt < 1e-9 else float(np.clip((t - t0) / dt, 0.0, 1.0))
        xy_out[i] = buf_xy[idx - 1] * (1.0 - alpha) + buf_xy[idx] * alpha
        cf_out[i] = buf_cf[idx - 1] * (1.0 - alpha) + buf_cf[idx] * alpha

    return xy_out, cf_out


# ---------------------------------------------------------------------------
# Main live inference entry point
# ---------------------------------------------------------------------------

def run_inference_live(
    *,
    camera_index: int,
    stop_event: threading.Event,
    classification_model_path: Path,
    keypoint_model_path: Path,
    on_packet: Optional[Callable[[Dict[str, Any]], None]] = None,
    frame_source: Optional[Callable[[], Optional[np.ndarray]]] = None,
    source_label: str = "server",
    save_path: Optional[Path] = None,
    arch: Optional[str] = None,
    labels_file: Optional[str] = None,
    device: Optional[str] = None,
    keypoint_backend: Optional[str] = None,
    half: int = 0,
    imgsz: int = 640,
    yolo_conf: float = 0.25,
    max_people: int = 1,
    track_conf_min: float = 0.75,
    track_max_jump_px: float = 0.0,
    track_max_jump_diag_frac: float = 0.25,
    track_max_lost: int = 10,
    track_target_x_frac: float = 0.5,
    track_target_y_frac: float = 0.5,
    T: int = 0,
    stride: int = 0,
    training_fps: float = TRAINING_FPS,
    normalize_mode: Optional[str] = None,
    missing_mode: Optional[str] = None,
    interp_mode: Optional[str] = None,
    interp_group: int = 0,
    rp_center_mode: Optional[str] = None,
    rp_img_w: int = 0,
    rp_img_h: int = 0,
    jpeg_quality: int = 80,
    **_unused: Any,
) -> int:
    """Run live inference from a webcam, emitting the same SSE packet format as
    run_inference_stream_packets in inference_on_video.py.

    Temporal sampling: each classifier window covers exactly T / training_fps
    real seconds.  The web API passes the user-selected FPS as training_fps for
    live mode, preserving T while changing the real-time window duration.
    """
    training_fps = max(1.0, float(training_fps))

    run_device = pick_device(device)
    use_half_requested = bool(int(half)) and run_device.startswith("cuda")
    use_half_keypoint = use_half_requested
    use_half_temporal = use_half_requested
    print(
        f"[live][runtime] device={run_device} half={int(use_half_requested)} "
        f"camera_index={camera_index} source={source_label}"
    )

    # ------------------------------------------------------------------ model
    ckpt_path, resolved_arch = resolve_ckpt_and_arch(
        str(classification_model_path), arch
    )
    print(f"[live][model] arch={resolved_arch} ckpt={ckpt_path.as_posix()}")

    state, meta = load_checkpoint(ckpt_path)
    state = clean_state_dict(state)

    T_raw = int(T) if int(T) > 0 else int(meta.get("T", meta.get("T_used", 64)) or 64)
    stride_raw = int(stride) if int(stride) > 0 else int(meta.get("stride", 16) or 16)
    T_final = max(1, T_raw)
    stride_final = max(1, min(stride_raw, T_final))

    print(f"[live][window] T={T_final} stride={stride_final} training_fps={training_fps:.1f}")

    use_conf = bool(meta.get("use_conf", True))
    normalize = bool(meta.get("normalize", True))
    normalize_mode_final = str(normalize_mode) if normalize_mode else str(
        meta.get("normalize_mode") or "center_scale"
    )
    add_vel = bool(meta.get("add_vel", True))
    add_acc = bool(meta.get("add_acc", True))
    add_global = bool(meta.get("add_global", True))
    add_mask = bool(meta.get("add_mask_channel", True))
    conf_thres = float(meta.get("conf_thres", 0.2))
    max_interp_gap = int(meta.get("max_interp_gap", 5))
    missing_mode_final = str(missing_mode) if missing_mode else str(
        meta.get("missing_mode") or "conf_thres"
    )
    interp_mode_final = str(interp_mode) if interp_mode else str(
        meta.get("interp_mode") or "short_gap_hold"
    )
    interp_group_final = int(interp_group) if int(interp_group) > 0 else int(
        meta.get("interp_group", 100) or 100
    )
    rp_center_mode_final = str(rp_center_mode) if rp_center_mode else str(
        meta.get("rp_center_mode") or "auto"
    )
    rp_img_w_final: Optional[int] = int(rp_img_w) if int(rp_img_w) > 0 else (
        int(meta["rp_img_w"]) if meta.get("rp_img_w") is not None else None
    )
    rp_img_h_final: Optional[int] = int(rp_img_h) if int(rp_img_h) > 0 else (
        int(meta["rp_img_h"]) if meta.get("rp_img_h") is not None else None
    )
    min_valid_frac = float(meta.get("min_valid_frac", 0.3))

    num_classes = int(meta.get("num_classes", 0) or 0)
    in_features_meta = int(meta.get("in_features", 0) or 0)
    if num_classes <= 0:
        raise ValueError(
            "Checkpoint missing num_classes. Use a checkpoint from training/train_models.py."
        )

    merge_fall_11_to_7 = int(num_classes) == 11
    display_num_classes = 7 if merge_fall_11_to_7 else int(num_classes)
    class_names = load_class_names(
        num_classes=display_num_classes, meta=meta, labels_file=labels_file
    )
    standing_label = next(
        (n for n in class_names if "stand" in str(n).lower()), "standing"
    )

    in_features = expected_in_features(
        use_conf=use_conf,
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        add_mask=add_mask,
    )
    if in_features_meta > 0 and int(in_features) != int(in_features_meta):
        raise ValueError(
            f"Feature mismatch: expected in_features={in_features}, "
            f"ckpt expects {in_features_meta}"
        )

    node_features_meta = meta.get("node_features", None)
    if node_features_meta is None:
        nf = int(in_features // K)
        node_features_meta = nf if nf * K == int(in_features) else None

    model = build_temporal_model(
        arch=resolved_arch,
        in_features=int(in_features),
        num_classes=int(num_classes),
        device=run_device,
        T_used=int(T_final),
        node_features=int(node_features_meta) if node_features_meta is not None else None,
    )
    missing_keys, unexpected = model.load_state_dict(state, strict=False)
    if missing_keys:
        print("[live][WARN] missing keys:", missing_keys[:8])
    if unexpected:
        print("[live][WARN] unexpected keys:", unexpected[:8])
    model.eval()
    if use_half_temporal:
        try:
            model.half()
        except Exception as err:
            use_half_temporal = False
            model.float()
            print(f"[live][WARN] FP16 temporal model failed, using FP32: {err}")

    from inference.helpers.keypoint_runtime import KeypointRuntime

    keypoint_runtime = KeypointRuntime(
        model_path=Path(keypoint_model_path).expanduser(),
        device=run_device,
        backend=keypoint_backend,
    )
    print(f"[live][pose] backend={keypoint_runtime.backend}")

    # ----------------------------------------------------------------- camera
    cap: Optional[cv2.VideoCapture] = None
    if frame_source is None:
        backend_flag = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(camera_index), backend_flag)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {camera_index}.")

        def read_frame() -> Optional[np.ndarray]:
            assert cap is not None
            ok, frame_bgr = cap.read()
            return frame_bgr if ok else None
    else:
        read_frame = frame_source

    writer: Optional[AsyncVideoWriter] = None
    out_w = 0
    out_h = 0
    source_ready = False
    try:
        # Time window parameters.
        window_duration_s = T_final / training_fps
        stride_duration_s = stride_final / training_fps
        # Keep enough history for the current window plus two strides.
        buffer_keep_s = window_duration_s + stride_duration_s * 2.0 + 2.0

        # Buffer: list of (timestamp, xy (K,2), cf (K,))
        sample_buf: List[Tuple[float, np.ndarray, np.ndarray]] = []

        most_recent_pred: Tuple[int, float, Optional[float]] = (-1, 0.0, None)
        next_win_start_t: Optional[float] = None

        # Tracking state.
        track_prev_center: Optional[np.ndarray] = None
        track_target_center: Optional[np.ndarray] = None
        track_max_jump_px_final: Optional[float] = None
        track_lost_count = 0

        fps_ema: Optional[float] = None
        ema_alpha = 0.1
        display_frame_idx = 0

        capture_interval_s = 1.0 / training_fps
        next_capture_t = time.perf_counter()
        consecutive_overruns = 0
        max_consecutive_overruns = 3

        while not stop_event.is_set():
            # ---------------------------------------- throttle to training_fps
            now = time.perf_counter()
            if next_capture_t > now:
                _sleep_until(next_capture_t)

            t_frame_start = time.perf_counter()
            if t_frame_start - next_capture_t > capture_interval_s:
                next_capture_t = t_frame_start
            next_capture_t += capture_interval_s

            frame = read_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            if frame.ndim != 3 or frame.shape[2] != 3:
                continue
            t_frame_start = time.perf_counter()

            if not source_ready:
                # Use first readable frame to set normalisation and output size.
                frame_h, frame_w = frame.shape[:2]
                out_w = int(frame_w)
                out_h = int(frame_h)
                if save_path is not None:
                    writer = AsyncVideoWriter(
                        open_video_writer(
                            save_path=Path(save_path).expanduser(),
                            fps=float(training_fps),
                            frame_size=(out_w, out_h),
                        )
                    )
                if rp_img_w_final is None:
                    rp_img_w_final = frame_w
                if rp_img_h_final is None:
                    rp_img_h_final = frame_h
                print(f"[live][camera] source={source_label} index={camera_index} res={frame_w}x{frame_h}")
                source_ready = True

            # ------------------------------------- initialise tracking targets
            if track_target_center is None or track_max_jump_px_final is None:
                h_img, w_img = frame.shape[:2]
                track_target_center = np.array(
                    [
                        float(w_img) * float(track_target_x_frac),
                        float(h_img) * float(track_target_y_frac),
                    ],
                    dtype=np.float32,
                )
                frame_diag = float(np.hypot(w_img, h_img))
                track_max_jump_px_final = (
                    float(track_max_jump_px)
                    if float(track_max_jump_px) > 0.0
                    else float(track_max_jump_diag_frac) * frame_diag
                )

            # ------------------------------------------- keypoint extraction
            xy, cf, new_center, found = pose_on_frame(
                keypoint_runtime=keypoint_runtime,
                frame_bgr=frame,
                imgsz=int(imgsz),
                yolo_conf=float(yolo_conf),
                max_people=int(max_people),
                use_half=use_half_keypoint,
                prev_center=track_prev_center,
                target_center=track_target_center,
                conf_min=float(track_conf_min),
                max_jump_px=float(track_max_jump_px_final),
            )
            if found:
                track_prev_center = new_center
                track_lost_count = 0
            else:
                track_lost_count += 1
                if track_lost_count > int(track_max_lost):
                    track_prev_center = None

            t_sample = time.perf_counter()
            sample_buf.append((t_sample, xy.copy(), cf.copy()))

            # ---------------------------------------- initialise window start
            if next_win_start_t is None:
                next_win_start_t = sample_buf[0][0]

            # -------------------------------- compute windows that are ready
            while next_win_start_t is not None:
                win_end_t = next_win_start_t + window_duration_s
                if t_sample < win_end_t:
                    break  # Not enough data yet.

                # Extract only the buffer slice covering this window.
                win_buf = [
                    s for s in sample_buf
                    if next_win_start_t - 0.001 <= s[0] <= win_end_t + 0.001
                ]
                if len(win_buf) < 2:
                    # Safety: need at least 2 points to interpolate.
                    next_win_start_t += stride_duration_s
                    continue

                buf_times = np.array([s[0] for s in win_buf], dtype=np.float64)
                buf_xy = np.stack([s[1] for s in win_buf], axis=0)
                buf_cf = np.stack([s[2] for s in win_buf], axis=0)

                target_times = np.linspace(
                    float(next_win_start_t), float(win_end_t), T_final,
                    dtype=np.float64,
                )
                xy_seq, cf_seq = _interpolate_temporal(
                    target_times, buf_times, buf_xy, buf_cf
                )

                window_feat = make_window_features(
                    xy_seq=xy_seq,
                    conf_seq=cf_seq,
                    T=int(T_final),
                    use_conf=use_conf,
                    normalize=normalize,
                    normalize_mode=normalize_mode_final,
                    add_vel=add_vel,
                    add_acc=add_acc,
                    add_global=add_global,
                    add_mask=add_mask,
                    conf_thres=conf_thres,
                    max_interp_gap=max_interp_gap,
                    missing_mode=missing_mode_final,
                    interp_mode=interp_mode_final,
                    interp_group=int(interp_group_final),
                    rp_center_mode=rp_center_mode_final,
                    rp_img_w=rp_img_w_final,
                    rp_img_h=rp_img_h_final,
                    min_valid_frac=min_valid_frac,
                )

                pred, pconf, p_fall = infer_one_window(
                    model=model,
                    window_feat=window_feat,
                    device=run_device,
                    use_half=use_half_temporal,
                    merge_fall_11_to_7=merge_fall_11_to_7,
                )
                most_recent_pred = (int(pred), float(pconf), p_fall)
                next_win_start_t += stride_duration_s

            # --------------------------------------------- trim old samples
            if sample_buf:
                cutoff_t = sample_buf[-1][0] - buffer_keep_s
                sample_buf = [s for s in sample_buf if s[0] >= cutoff_t]

            # ------------------------------------------- build and emit packet
            pred_id, pconf, p_fall = most_recent_pred
            label = class_names[pred_id] if 0 <= pred_id < len(class_names) else "..."
            fps_display = float(fps_ema) if fps_ema is not None else 0.0

            # Show "warming up" until the first window is ready.
            warming_up = pred_id < 0
            if warming_up:
                pose_line = f"pose: {standing_label} (warming up...)"
            else:
                pose_line = f"pose: {label} ({pconf:.2f})"

            hud: List[str] = [
                f"frame {display_frame_idx + 1}",
                f"fps: {fps_display:.1f}",
                pose_line,
                f"T={T_final} stride={stride_final} (live)",
            ]
            if p_fall is not None and not warming_up:
                hud.append(f"fall_prob: {p_fall:.2f}")

            if on_packet is not None:
                ok_jpg, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if ok_jpg:
                    frame_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
                    packet: Dict[str, Any] = {
                        "type": "frame",
                        "frame_index": display_frame_idx,
                        "frame_number": display_frame_idx + 1,
                        "frame_count": 0,  # unknown for live stream
                        "fps": fps_display,
                        "pred": {
                            "label": str(label),
                            "conf": float(pconf),
                            "class_id": int(pred_id),
                        },
                        "params": {
                            "T": int(T_final),
                            "stride": int(stride_final),
                            "k": 1,
                        },
                        "hud_lines": hud,
                        "pose": {
                            "format": "coco17",
                            "xy": np.asarray(xy, dtype=np.float32).tolist(),
                            "conf": np.asarray(cf, dtype=np.float32).tolist(),
                            "conf_thres": float(conf_thres),
                            "skeleton": [[int(a), int(b)] for a, b in SKELETON],
                        },
                        "frame_jpeg_b64": frame_b64,
                        "size": {
                            "w": int(frame.shape[1]),
                            "h": int(frame.shape[0]),
                        },
                        "overlay": {
                            "hud": {
                                "x": 10,
                                "y": 10,
                                "pad": 8,
                                "line_gap": 6,
                                "bg_alpha": 0.6,
                                "font_px": 20,
                            },
                            "pose": {
                                "keypoint_radius": 3,
                                "skeleton_width": 2,
                            },
                        },
                    }
                    if p_fall is not None and not warming_up:
                        packet["pred"]["fall_prob"] = float(p_fall)
                    on_packet(packet)

            if writer is not None:
                frame_to_write = draw_pose(frame.copy(), xy, cf, conf_thres=conf_thres)
                frame_to_write = draw_hud(frame_to_write, hud)
                frame_h, frame_w = frame_to_write.shape[:2]
                if frame_w != out_w or frame_h != out_h:
                    frame_to_write = cv2.resize(frame_to_write, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                writer.write(frame_to_write)

            # ------------------------------------------------- fps tracking
            total_ms = (time.perf_counter() - t_frame_start) * 1000.0
            inst_fps = 1000.0 / max(1e-6, total_ms)
            fps_ema = (
                inst_fps
                if fps_ema is None
                else (1.0 - ema_alpha) * fps_ema + ema_alpha * inst_fps
            )
            if (total_ms / 1000.0) > (capture_interval_s * 1.05):
                consecutive_overruns += 1
                if consecutive_overruns >= max_consecutive_overruns:
                    effective_fps = 1000.0 / max(1e-6, total_ms)
                    raise RuntimeError(
                        "Live inference cannot keep up with the selected FPS "
                        f"({training_fps:.0f}). Recent frame processed at "
                        f"{effective_fps:.1f} FPS. Select a lower FPS or a faster model."
                    )
            else:
                consecutive_overruns = 0
            display_frame_idx += 1

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

    return 0


def _decode_jpeg_frame(frame_bytes: bytes) -> Optional[np.ndarray]:
    if not frame_bytes:
        return None
    encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    return frame


def run_inference_live_client(
    *,
    frame_bytes_source: Callable[[], Optional[bytes]],
    stop_event: threading.Event,
    classification_model_path: Path,
    keypoint_model_path: Path,
    on_packet: Optional[Callable[[Dict[str, Any]], None]] = None,
    save_path: Optional[Path] = None,
    **kwargs: Any,
) -> int:
    def read_uploaded_frame() -> Optional[np.ndarray]:
        frame_bytes = frame_bytes_source()
        if frame_bytes is None:
            return None
        return _decode_jpeg_frame(frame_bytes)

    return run_inference_live(
        camera_index=-1,
        stop_event=stop_event,
        classification_model_path=classification_model_path,
        keypoint_model_path=keypoint_model_path,
        on_packet=on_packet,
        frame_source=read_uploaded_frame,
        source_label="client",
        save_path=save_path,
        **kwargs,
    )
