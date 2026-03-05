#!/usr/bin/env python3
"""
Offline video -> YOLOv11 pose -> MotionBERT ActionNet inference.

This script:
1) Runs YOLOv11 pose on a video
2) Builds a MotionBERT action .pkl (same schema as prepare_motionbert_dataset.py)
3) Loads a MotionBERT ActionNet checkpoint (*.bin)
4) Runs per-window predictions and saves a CSV

Display mode (--display)
- Runs YOLO pose + MotionBERT inference while the video is being displayed (similar to inference_on_video.py).
- If processing is slower than the source FPS, playback will automatically slow down (no frame skipping).
- Shows the effective FPS in an on-screen HUD.

Notes
- Preprocessing mirrors MotionBERT training (models/MotionBERT/lib/data/dataset_action.py):
  make_cam -> human_tracking -> coco2h36m -> concat conf -> resample -> crop_scale
- You can pass paths relative to:
    - your current working directory
    - this repo root (fall_models/Prototype)
    - MotionBERT root (fall_models/Prototype/models/MotionBERT)
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import platform
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# MotionBERT imports: add MotionBERT root (contains `lib/`) to sys.path
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]  # fall_models/Prototype

_MB_ROOT = _REPO_ROOT / "models" / "MotionBERT"
if not _MB_ROOT.exists():
    # Fallback: walk upwards until we find models/MotionBERT (helps if this file is moved).
    for parent in _THIS_FILE.parents:
        cand = parent / "models" / "MotionBERT"
        if cand.exists():
            _REPO_ROOT = parent
            _MB_ROOT = cand
            break

if not _MB_ROOT.exists():
    raise FileNotFoundError(f"MotionBERT root not found at: {_MB_ROOT.as_posix()}")

mb_root_str = str(_MB_ROOT)
if mb_root_str not in sys.path:
    sys.path.insert(0, mb_root_str)

# Match MotionBERT training/eval scripts (train_action_weighted_balanced.py) which import via top-level `lib.*`.
from lib.utils.tools import get_config  # noqa: E402
from lib.utils.learning import load_backbone  # noqa: E402
from lib.model.model_action import ActionNet  # noqa: E402
from lib.data.dataset_action import make_cam, coco2h36m, human_tracking  # noqa: E402
from lib.utils.utils_data import crop_scale, resample  # noqa: E402

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
CLASS_NAMES_DEFAULT = [
    "Falling forward using hands",  # 0
    "Falling forward using knees",  # 1
    "Falling backwards",  # 2
    "Falling sideward",  # 3
    "Falling sitting in an empty chair",  # 4
    "Walking",  # 5
    "Standing",  # 6
    "Sitting",  # 7
    "Picking up an object",  # 8
    "Jumping",  # 9
    "Laying",  # 10
]

CLASS_NAMES_MERGED_DEFAULT = [
    "Fall",  # 0 (all fall subclasses merged)
    "Walking",  # 1
    "Standing",  # 2
    "Sitting",  # 3
    "Picking up an object",  # 4
    "Jumping",  # 5
    "Laying",  # 6
]

FALL_CLASS_IDS_DEFAULT = [0, 1, 2, 3, 4]

# COCO keypoint order for Ultralytics pose models (17 joints)
K = 17

SKELETON = [
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
    (5, 6),  # shoulders
    (11, 12),  # hips
    (5, 11),
    (6, 12),  # torso sides
]


def pick_device(device: Optional[str]) -> str:
    if not device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    d = device.lower().strip()
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return float(default)
    return out


def _parse_first_number(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"[+-]?\d+(?:\.\d+)?", v)
        if m:
            return _safe_float(m.group(0))
    return float("nan")


def _is_finite(v: Any) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _avg_valid(vals: List[float]) -> float:
    good = [float(x) for x in vals if _is_finite(x)]
    if not good:
        return float("nan")
    return float(np.mean(good))


def _max_valid(vals: List[float]) -> float:
    good = [float(x) for x in vals if _is_finite(x)]
    if not good:
        return float("nan")
    return float(np.max(good))


def _median_valid(vals: List[float]) -> float:
    good = [float(x) for x in vals if _is_finite(x)]
    if not good:
        return float("nan")
    return float(np.median(good))


def _p95_valid(vals: List[float]) -> float:
    good = [float(x) for x in vals if _is_finite(x)]
    if not good:
        return float("nan")
    return float(np.percentile(good, 95))


def _to_csv_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else ""
    return v


def _json_safe_number(v: Any) -> Optional[float]:
    fv = _safe_float(v)
    if np.isfinite(fv):
        return float(fv)
    return None


def _fmt_live_metric(v: Any, unit: str = "", digits: int = 1) -> str:
    fv = _safe_float(v)
    if np.isfinite(fv):
        return f"{fv:.{digits}f}{unit}"
    return "NA"


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: _to_csv_cell(row.get(k, "")) for k in fieldnames}
            writer.writerow(out)


def _slugify_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name).strip())
    s = re.sub(r"-{2,}", "-", s).strip("-._")
    return s or "model"


def _pick_profile_out_dir(
    profile_out_arg: Optional[str],
    save_path: Optional[Path],
    ckpt_path: Path,
    run_tag: str = "motionbert",
) -> Path:
    base_root = Path(profile_out_arg).expanduser() if profile_out_arg else (save_path.parent if save_path is not None else Path("runs") / "profiling")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_tag = _slugify_name(Path(ckpt_path).stem)
    tag = _slugify_name(str(run_tag))
    run_name = f"{stamp}_{tag}_{model_tag}"
    out_dir = base_root / run_name

    # Keep unique even under very fast relaunches.
    suffix = 1
    while out_dir.exists():
        out_dir = base_root / f"{run_name}_{suffix:02d}"
        suffix += 1
    return out_dir


def assert_benchmark_device_ok(benchmark: bool, device: str) -> None:
    """
    Runtime guard for benchmark mode.
    Kept as a standalone helper so it is easy to test on CPU-only machines.
    """
    if not bool(benchmark):
        return
    device_str = str(device).strip()
    if not device_str.lower().startswith("cuda"):
        raise ValueError(
            f"--benchmark requires CUDA, but resolved runtime device is '{device_str}'. "
            "Use --device cuda on a CUDA-capable machine, or disable --benchmark."
        )


def get_benchmark_duration_s(default_s: float = 600.0) -> float:
    """
    Internal dev override for quick smoke benchmarks without changing CLI:
      BENCHMARK_DURATION_S=<seconds>
    """
    raw = os.getenv("BENCHMARK_DURATION_S")
    if raw is None or str(raw).strip() == "":
        return float(default_s)
    try:
        out = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid BENCHMARK_DURATION_S='{raw}'. Expected a positive number of seconds.") from e
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"BENCHMARK_DURATION_S must be a finite number > 0, got '{raw}'.")
    return float(out)


def _parse_tegrastats_line(line: str) -> Dict[str, float]:
    sample = {
        "ram_used_pct": float("nan"),
        "cpu_pct": float("nan"),
        "gpu_pct": float("nan"),
        "cpu_temp_c": float("nan"),
        "gpu_temp_c": float("nan"),
        "power_w": float("nan"),
    }

    m_ram = re.search(r"\bRAM\s+(\d+)/(\d+)MB\b", line, flags=re.IGNORECASE)
    if m_ram:
        used = _safe_float(m_ram.group(1))
        total = _safe_float(m_ram.group(2))
        if total > 0:
            sample["ram_used_pct"] = 100.0 * used / total

    m_cpu = re.search(r"\bCPU\s+\[([^\]]+)\]", line, flags=re.IGNORECASE)
    if m_cpu:
        loads: List[float] = []
        for tok in m_cpu.group(1).split(","):
            m_pct = re.search(r"([+-]?\d+(?:\.\d+)?)%", tok)
            if m_pct:
                loads.append(_safe_float(m_pct.group(1)))
        sample["cpu_pct"] = _avg_valid(loads)

    m_gpu = re.search(r"\bGR3D(?:_FREQ)?[:=]?\s*([+-]?\d+(?:\.\d+)?)%", line, flags=re.IGNORECASE)
    if m_gpu:
        sample["gpu_pct"] = _safe_float(m_gpu.group(1))

    m_cpu_t = re.search(r"\bCPU@([+-]?\d+(?:\.\d+)?)C\b", line, flags=re.IGNORECASE)
    if m_cpu_t:
        sample["cpu_temp_c"] = _safe_float(m_cpu_t.group(1))

    m_gpu_t = re.search(r"\bGPU@([+-]?\d+(?:\.\d+)?)C\b", line, flags=re.IGNORECASE)
    if m_gpu_t:
        sample["gpu_temp_c"] = _safe_float(m_gpu_t.group(1))

    power_patterns = [
        r"\bPOM_5V_IN\s+([+-]?\d+(?:\.\d+)?)(m?W)?(?:/([+-]?\d+(?:\.\d+)?)(m?W)?)?",
        r"\bVDD_IN\s+([+-]?\d+(?:\.\d+)?)(m?W)?(?:/([+-]?\d+(?:\.\d+)?)(m?W)?)?",
        r"\bPWR\s+([+-]?\d+(?:\.\d+)?)(m?W)?(?:/([+-]?\d+(?:\.\d+)?)(m?W)?)?",
    ]
    for pat in power_patterns:
        m = re.search(pat, line, flags=re.IGNORECASE)
        if not m:
            continue
        now_val = _safe_float(m.group(1))
        now_unit = (m.group(2) or "").lower()
        if now_unit == "mw":
            now_val /= 1000.0
        sample["power_w"] = now_val
        break

    return sample


def _extract_numeric_from_obj(v: Any) -> float:
    if isinstance(v, (int, float, np.number)):
        return _safe_float(v)
    if isinstance(v, str):
        return _parse_first_number(v)
    return float("nan")


def _collect_keyed_numeric(obj: Any, prefix: str = "") -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []

    def _rec(cur: Any, path: List[str]) -> None:
        if isinstance(cur, dict):
            for k, v in cur.items():
                key = str(k).lower()
                next_path = path + [key]
                if isinstance(v, (dict, list, tuple)):
                    _rec(v, next_path)
                else:
                    out.append(("_".join(next_path), _extract_numeric_from_obj(v)))
        elif hasattr(cur, "__dict__"):
            _rec(vars(cur), path)
        elif isinstance(cur, (list, tuple)):
            for i, v in enumerate(cur):
                next_path = path + [str(i)]
                if isinstance(v, (dict, list, tuple)):
                    _rec(v, next_path)
                else:
                    out.append(("_".join(next_path), _extract_numeric_from_obj(v)))

    start = [str(prefix).lower()] if prefix else []
    _rec(obj, start)
    return out


def _pick_pct_from_keyed(values: List[Tuple[str, float]], tokens_any: Tuple[str, ...]) -> float:
    cand: List[float] = []
    for key, val in values:
        if not np.isfinite(val):
            continue
        key_l = str(key).lower()
        if any(tok in key_l for tok in tokens_any) and 0.0 <= float(val) <= 100.0:
            cand.append(float(val))
    return _avg_valid(cand)


class HardwareSampler:
    def __init__(self, sample_hz: float) -> None:
        self.sample_hz = max(1e-3, float(sample_hz))
        self.interval_s = 1.0 / self.sample_hz
        self.backend = "none"
        self.samples: List[Dict[str, Any]] = []

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_t = 0.0

        self._jtop_obj = None
        self._tegrastats_proc: Optional[subprocess.Popen[str]] = None
        self._psutil_mod = None

    def start(self) -> str:
        self._start_t = time.perf_counter()
        self._stop_event.clear()

        # Preferred backend: jtop (jetson-stats)
        try:
            from jtop import jtop  # type: ignore
            from jtop.core import hardware as jtop_hardware  # type: ignore

            # Some old jetson-stats builds call platform.linux_distribution(), removed in Python 3.8+.
            try:
                src = inspect.getsource(jtop_hardware.get_platform_variables)  # type: ignore[name-defined]
                if ("linux_distribution" in src) and (not hasattr(platform, "linux_distribution")):
                    raise RuntimeError(
                        "Installed jetson-stats/jtop is incompatible with this Python version; falling back."
                    )
            except OSError:
                # Source may be unavailable in some installations; continue and try runtime start.
                pass

            self._jtop_obj = jtop()
            self._jtop_obj.start()
            self.backend = "jtop"
            self._thread = threading.Thread(target=self._run_jtop, name="hw-jtop", daemon=True)
            self._thread.start()
            return self.backend
        except Exception as e:
            self._jtop_obj = None
            print(f"[profile][WARN] jtop unavailable/incompatible: {e}")

        # Fallback backend: tegrastats
        if shutil.which("tegrastats") is not None:
            try:
                interval_ms = max(100, int(round(self.interval_s * 1000.0)))
                self._tegrastats_proc = subprocess.Popen(
                    ["tegrastats", "--interval", str(interval_ms)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                self.backend = "tegrastats"
                self._thread = threading.Thread(target=self._run_tegrastats, name="hw-tegrastats", daemon=True)
                self._thread.start()
                return self.backend
            except Exception:
                self._tegrastats_proc = None

        # Final fallback: psutil (RAM+CPU, best-effort temperatures).
        try:
            import psutil  # type: ignore

            self._psutil_mod = psutil
            self._psutil_mod.cpu_percent(interval=None)
            self.backend = "psutil"
            self._thread = threading.Thread(target=self._run_psutil, name="hw-psutil", daemon=True)
            self._thread.start()
            return self.backend
        except Exception:
            self._psutil_mod = None
            self.backend = "none"
            return self.backend

    def stop(self) -> None:
        self._stop_event.set()

        if self._tegrastats_proc is not None:
            try:
                self._tegrastats_proc.terminate()
            except Exception:
                pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._tegrastats_proc is not None:
            try:
                if self._tegrastats_proc.poll() is None:
                    self._tegrastats_proc.kill()
            except Exception:
                pass
            self._tegrastats_proc = None

        if self._jtop_obj is not None:
            try:
                self._jtop_obj.close()
            except Exception:
                pass
            self._jtop_obj = None

    def get_samples(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(x) for x in self.samples]

    def get_latest_sample(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self.samples:
                return None
            return dict(self.samples[-1])

    def _append_sample(self, sample: Dict[str, Any]) -> None:
        ram_pct = _safe_float(sample.get("ram_used_pct", float("nan")))
        cpu_pct = _safe_float(sample.get("cpu_pct", float("nan")))
        gpu_pct = _safe_float(sample.get("gpu_pct", float("nan")))
        cpu_temp = _safe_float(sample.get("cpu_temp_c", float("nan")))
        gpu_temp = _safe_float(sample.get("gpu_temp_c", float("nan")))
        power_w = _safe_float(sample.get("power_w", float("nan")))

        if not (0.0 <= ram_pct <= 100.0):
            ram_pct = float("nan")
        if not (0.0 <= cpu_pct <= 100.0):
            cpu_pct = float("nan")
        if not (0.0 <= gpu_pct <= 100.0):
            gpu_pct = float("nan")
        if not (-20.0 <= cpu_temp <= 150.0):
            cpu_temp = float("nan")
        if not (-20.0 <= gpu_temp <= 150.0):
            gpu_temp = float("nan")
        if not (0.0 <= power_w <= 200.0):
            power_w = float("nan")

        row = {
            "t_s": float(time.perf_counter() - self._start_t),
            "ram_used_pct": ram_pct,
            "cpu_pct": cpu_pct,
            "gpu_pct": gpu_pct,
            "cpu_temp_c": cpu_temp,
            "gpu_temp_c": gpu_temp,
            "power_w": power_w,
            "backend": self.backend,
        }
        with self._lock:
            self.samples.append(row)

    def _run_jtop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._jtop_obj is None:
                    break
                if hasattr(self._jtop_obj, "ok") and not self._jtop_obj.ok():
                    break
                stats = dict(getattr(self._jtop_obj, "stats", {}))
                gpu_obj = getattr(self._jtop_obj, "gpu", None)
                temp_obj = getattr(self._jtop_obj, "temperature", None)
                power_obj = getattr(self._jtop_obj, "power", None)
                memory_obj = getattr(self._jtop_obj, "memory", None)
                cpu_obj = getattr(self._jtop_obj, "cpu", None)
            except Exception:
                stats = {}
                gpu_obj = None
                temp_obj = None
                power_obj = None
                memory_obj = None
                cpu_obj = None

            sample = self._sample_from_jtop_stats(
                stats=stats,
                gpu_obj=gpu_obj,
                temp_obj=temp_obj,
                power_obj=power_obj,
                memory_obj=memory_obj,
                cpu_obj=cpu_obj,
            )
            self._append_sample(sample)

            if self._stop_event.wait(self.interval_s):
                break

    def _run_tegrastats(self) -> None:
        proc = self._tegrastats_proc
        if proc is None or proc.stdout is None:
            return
        while not self._stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            sample = _parse_tegrastats_line(line)
            self._append_sample(sample)

    def _run_psutil(self) -> None:
        psutil = self._psutil_mod
        if psutil is None:
            return
        while not self._stop_event.is_set():
            ram_pct = float("nan")
            cpu_pct = float("nan")
            cpu_temp = float("nan")

            try:
                ram_pct = float(psutil.virtual_memory().percent)
            except Exception:
                pass
            try:
                cpu_pct = float(psutil.cpu_percent(interval=None))
            except Exception:
                pass
            try:
                temps = psutil.sensors_temperatures(fahrenheit=False)
                cpu_candidates: List[float] = []
                for name, entries in (temps or {}).items():
                    name_l = str(name).lower()
                    for ent in entries:
                        cur = _safe_float(getattr(ent, "current", float("nan")))
                        lbl = str(getattr(ent, "label", "")).lower()
                        if not np.isfinite(cur):
                            continue
                        if any(tok in name_l for tok in ("cpu", "core", "package", "soc")) or any(
                            tok in lbl for tok in ("cpu", "core", "package", "soc")
                        ):
                            cpu_candidates.append(cur)
                cpu_temp = _avg_valid(cpu_candidates)
            except Exception:
                pass

            self._append_sample(
                {
                    "ram_used_pct": ram_pct,
                    "cpu_pct": cpu_pct,
                    "gpu_pct": float("nan"),
                    "cpu_temp_c": cpu_temp,
                    "gpu_temp_c": float("nan"),
                    "power_w": float("nan"),
                }
            )

            if self._stop_event.wait(self.interval_s):
                break

    def _sample_from_jtop_stats(
        self,
        *,
        stats: Dict[str, Any],
        gpu_obj: Any,
        temp_obj: Any,
        power_obj: Any,
        memory_obj: Any,
        cpu_obj: Any,
    ) -> Dict[str, float]:
        keyed = _collect_keyed_numeric(stats, prefix="stats")

        ram_pct = _pick_pct_from_keyed(keyed, ("ram", "mem"))
        cpu_pct = _pick_pct_from_keyed(keyed, ("cpu", "core"))
        gpu_pct = _pick_pct_from_keyed(keyed, ("gpu", "gr3d"))

        cpu_temp = float("nan")
        gpu_temp = float("nan")
        power_w = float("nan")

        for key, val in keyed:
            key_l = key.lower()
            if np.isfinite(val):
                if ("temp" in key_l or key_l.endswith("_c")) and ("cpu" in key_l or "soc" in key_l):
                    cpu_temp = val if not np.isfinite(cpu_temp) else _avg_valid([cpu_temp, val])
                if ("temp" in key_l or key_l.endswith("_c")) and ("gpu" in key_l or "gr3d" in key_l):
                    gpu_temp = val if not np.isfinite(gpu_temp) else _avg_valid([gpu_temp, val])
                if "power" in key_l and ("in" in key_l or "tot" in key_l or "sum" in key_l):
                    power_w = val if not np.isfinite(power_w) else _avg_valid([power_w, val])

        if np.isfinite(power_w) and power_w > 1000.0:
            power_w /= 1000.0

        # Fall back to object trees when stats doesn't expose enough.
        if memory_obj is not None:
            mem_keyed = _collect_keyed_numeric(memory_obj, prefix="memory")
            mem_pct = _pick_pct_from_keyed(mem_keyed, ("ram", "used", "percent", "util"))
            if np.isfinite(mem_pct):
                ram_pct = mem_pct
        if cpu_obj is not None:
            cpu_keyed = _collect_keyed_numeric(cpu_obj, prefix="cpu")
            cpu_pct2 = _pick_pct_from_keyed(cpu_keyed, ("util", "load", "percent"))
            if np.isfinite(cpu_pct2):
                cpu_pct = cpu_pct2
        if gpu_obj is not None:
            gpu_keyed = _collect_keyed_numeric(gpu_obj, prefix="gpu")
            gpu_pct2 = _pick_pct_from_keyed(gpu_keyed, ("util", "load", "percent"))
            if np.isfinite(gpu_pct2):
                gpu_pct = gpu_pct2
        if temp_obj is not None:
            temp_keyed = _collect_keyed_numeric(temp_obj, prefix="temp")
            cpu_t = _pick_pct_from_keyed(temp_keyed, ("cpu", "soc", "temp"))
            gpu_t = _pick_pct_from_keyed(temp_keyed, ("gpu", "gr3d", "temp"))
            if np.isfinite(cpu_t):
                cpu_temp = cpu_t
            if np.isfinite(gpu_t):
                gpu_temp = gpu_t
        if power_obj is not None:
            pwr_keyed = _collect_keyed_numeric(power_obj, prefix="power")
            pwr_vals = [v for k, v in pwr_keyed if np.isfinite(v) and any(tok in k for tok in ("tot", "in", "sum", "pwr", "watt"))]
            if pwr_vals:
                power_w = _avg_valid(pwr_vals)
                if np.isfinite(power_w) and power_w > 1000.0:
                    power_w /= 1000.0

        return {
            "ram_used_pct": ram_pct,
            "cpu_pct": cpu_pct,
            "gpu_pct": gpu_pct,
            "cpu_temp_c": cpu_temp,
            "gpu_temp_c": gpu_temp,
            "power_w": power_w,
        }


def _save_time_plots(
    profile_out: Path,
    time_rows: List[Dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    frame_idx = np.asarray([_safe_float(r.get("frame_idx", np.nan)) for r in time_rows], dtype=np.float64)
    fps = np.asarray([_safe_float(r.get("fps", np.nan)) for r in time_rows], dtype=np.float64)
    prep = np.asarray([_safe_float(r.get("preprocess_ms", np.nan)) for r in time_rows], dtype=np.float64)
    infer = np.asarray([_safe_float(r.get("inference_ms", np.nan)) for r in time_rows], dtype=np.float64)
    post = np.asarray([_safe_float(r.get("postprocess_ms", np.nan)) for r in time_rows], dtype=np.float64)
    vis = np.asarray([_safe_float(r.get("visualisation_ms", np.nan)) for r in time_rows], dtype=np.float64)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    if frame_idx.size > 0:
        ax0.plot(frame_idx, fps, color="tab:blue", linewidth=1.3, label="Total FPS")
        ax0.set_ylabel("FPS")
        ax0.legend(loc="upper right")
        ax0.grid(True, linestyle="--", alpha=0.35)
        fps_top = max(1.0, float(np.nanpercentile(fps, 99)) * 1.2) if np.isfinite(fps).any() else 1.0
        ax0.set_ylim(0.0, fps_top)

        ax1.plot(frame_idx, prep, label="Preprocess", linewidth=1.0)
        ax1.plot(frame_idx, infer, label="Inference", linewidth=1.0)
        ax1.plot(frame_idx, post, label="Postprocess", linewidth=1.0)
        ax1.plot(frame_idx, vis, label="Visualisation", linewidth=1.0)
        ax1.set_xlabel("Displayed Frame Index")
        ax1.set_ylabel("Time (ms)")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.35)
    else:
        ax0.text(0.5, 0.5, "No timing data", ha="center", va="center", transform=ax0.transAxes)
        ax1.text(0.5, 0.5, "No timing data", ha="center", va="center", transform=ax1.transAxes)
        ax0.set_ylabel("FPS")
        ax1.set_xlabel("Displayed Frame Index")
        ax1.set_ylabel("Time (ms)")

    fig.tight_layout()
    fig.savefig(profile_out / "fig_time_efficiency.png", dpi=180)
    plt.close(fig)


def _save_hw_plots(
    profile_out: Path,
    hw_rows: List[Dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    t_s = np.asarray([_safe_float(r.get("t_s", np.nan)) for r in hw_rows], dtype=np.float64)
    ram = np.asarray([_safe_float(r.get("ram_used_pct", np.nan)) for r in hw_rows], dtype=np.float64)
    cpu = np.asarray([_safe_float(r.get("cpu_pct", np.nan)) for r in hw_rows], dtype=np.float64)
    gpu = np.asarray([_safe_float(r.get("gpu_pct", np.nan)) for r in hw_rows], dtype=np.float64)
    cpu_t = np.asarray([_safe_float(r.get("cpu_temp_c", np.nan)) for r in hw_rows], dtype=np.float64)
    gpu_t = np.asarray([_safe_float(r.get("gpu_temp_c", np.nan)) for r in hw_rows], dtype=np.float64)
    pwr = np.asarray([_safe_float(r.get("power_w", np.nan)) for r in hw_rows], dtype=np.float64)

    fig1, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
    if t_s.size > 0:
        ax0.plot(t_s, ram, label="RAM", linewidth=1.2)
        ax0.plot(t_s, cpu, label="CPU", linewidth=1.2)
        ax0.plot(t_s, gpu, label="GPU", linewidth=1.2)
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("Utilisation (%)")
        ax0.set_ylim(0.0, 100.0)
        ax0.legend(loc="best")

        ax1.plot(t_s, ram, label="RAM", linewidth=1.2)
        ax1.plot(t_s, cpu, label="CPU", linewidth=1.2)
        ax1.plot(t_s, gpu, label="GPU", linewidth=1.2)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Utilisation (%)")
        ax1.set_ylim(0.0, 100.0)
        ax1.legend(loc="best")
        if np.isfinite(t_s).any():
            x0 = float(np.nanmax(t_s))
            x1 = max(1.0, x0 * 0.05)
            ax1.set_xlim(max(0.0, x0 - x1), x0)
    else:
        ax0.text(0.5, 0.5, "No hardware data", ha="center", va="center", transform=ax0.transAxes)
        ax1.text(0.5, 0.5, "No hardware data", ha="center", va="center", transform=ax1.transAxes)
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("Utilisation (%)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Utilisation (%)")
    fig1.tight_layout()
    fig1.savefig(profile_out / "fig_hw_usage.png", dpi=180)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    bx0, bx1 = axes2
    if t_s.size > 0:
        bx0.plot(t_s, cpu_t, label="CPU temp", linewidth=1.2)
        bx0.plot(t_s, gpu_t, label="GPU temp", linewidth=1.2)
        bx0.set_xlabel("Time (s)")
        bx0.set_ylabel("Temperature (C)")
        bx0.legend(loc="best")

        bx1.plot(t_s, pwr, label="Power draw", color="tab:red", linewidth=1.2)
        bx1.set_xlabel("Time (s)")
        bx1.set_ylabel("Power draw (W)")
        bx1.legend(loc="best")
    else:
        bx0.text(0.5, 0.5, "No hardware data", ha="center", va="center", transform=bx0.transAxes)
        bx1.text(0.5, 0.5, "No hardware data", ha="center", va="center", transform=bx1.transAxes)
        bx0.set_xlabel("Time (s)")
        bx0.set_ylabel("Temperature (C)")
        bx1.set_xlabel("Time (s)")
        bx1.set_ylabel("Power draw (W)")
    fig2.tight_layout()
    fig2.savefig(profile_out / "fig_temp_power.png", dpi=180)
    plt.close(fig2)


def _save_profile_artifacts(
    profile_out: Path,
    time_rows: List[Dict[str, Any]],
    hw_rows: List[Dict[str, Any]],
    extra_summary: Optional[Dict[str, Any]] = None,
) -> None:
    profile_out.mkdir(parents=True, exist_ok=True)

    time_fields = [
        "frame_idx",
        "t_s",
        "fps",
        "fps_ema",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "visualisation_ms",
        "total_ms",
        "cap_read_ms",
        "yolo_infer_ms",
        "window_feature_ms",
        "temporal_infer_ms",
        "label_select_ms",
        "draw_ms",
        "writer_ms",
        "display_wait_ms",
    ]
    _write_csv(profile_out / "time_metrics.csv", time_rows, time_fields)

    hw_fields = [
        "t_s",
        "ram_used_pct",
        "cpu_pct",
        "gpu_pct",
        "cpu_temp_c",
        "gpu_temp_c",
        "power_w",
    ]
    _write_csv(profile_out / "hw_metrics.csv", hw_rows, hw_fields)

    try:
        _save_time_plots(profile_out=profile_out, time_rows=time_rows)
    except Exception as e:
        print(f"[WARN] Could not save fig_time_efficiency.png: {e}")

    try:
        _save_hw_plots(profile_out=profile_out, hw_rows=hw_rows)
    except Exception as e:
        print(f"[WARN] Could not save hardware figures: {e}")

    fps_vals = [_safe_float(r.get("fps", np.nan)) for r in time_rows]
    preprocess_vals = [_safe_float(r.get("preprocess_ms", np.nan)) for r in time_rows]
    infer_vals = [_safe_float(r.get("inference_ms", np.nan)) for r in time_rows]
    post_vals = [_safe_float(r.get("postprocess_ms", np.nan)) for r in time_rows]
    vis_vals = [_safe_float(r.get("visualisation_ms", np.nan)) for r in time_rows]
    t_vals = [_safe_float(r.get("t_s", np.nan)) for r in time_rows]

    ram_vals = [_safe_float(r.get("ram_used_pct", np.nan)) for r in hw_rows]
    cpu_vals = [_safe_float(r.get("cpu_pct", np.nan)) for r in hw_rows]
    gpu_vals = [_safe_float(r.get("gpu_pct", np.nan)) for r in hw_rows]
    cpu_temp_vals = [_safe_float(r.get("cpu_temp_c", np.nan)) for r in hw_rows]
    gpu_temp_vals = [_safe_float(r.get("gpu_temp_c", np.nan)) for r in hw_rows]
    power_vals = [_safe_float(r.get("power_w", np.nan)) for r in hw_rows]

    duration_s = float("nan")
    if t_vals:
        finite_t = [x for x in t_vals if _is_finite(x)]
        if finite_t:
            duration_s = float(max(finite_t))

    summary: Dict[str, Any] = {
        "avg_fps": _json_safe_number(_avg_valid(fps_vals)),
        "median_fps": _json_safe_number(_median_valid(fps_vals)),
        "preprocess_ms": {
            "mean": _json_safe_number(_avg_valid(preprocess_vals)),
            "median": _json_safe_number(_median_valid(preprocess_vals)),
            "p95": _json_safe_number(_p95_valid(preprocess_vals)),
        },
        "inference_ms": {
            "mean": _json_safe_number(_avg_valid(infer_vals)),
            "median": _json_safe_number(_median_valid(infer_vals)),
            "p95": _json_safe_number(_p95_valid(infer_vals)),
        },
        "postprocess_ms": {
            "mean": _json_safe_number(_avg_valid(post_vals)),
            "median": _json_safe_number(_median_valid(post_vals)),
            "p95": _json_safe_number(_p95_valid(post_vals)),
        },
        "visualisation_ms": {
            "mean": _json_safe_number(_avg_valid(vis_vals)),
            "median": _json_safe_number(_median_valid(vis_vals)),
            "p95": _json_safe_number(_p95_valid(vis_vals)),
        },
        "avg_ram_pct": _json_safe_number(_avg_valid(ram_vals)),
        "avg_cpu_pct": _json_safe_number(_avg_valid(cpu_vals)),
        "avg_gpu_pct": _json_safe_number(_avg_valid(gpu_vals)),
        "avg_cpu_temp_c": _json_safe_number(_avg_valid(cpu_temp_vals)),
        "avg_gpu_temp_c": _json_safe_number(_avg_valid(gpu_temp_vals)),
        "max_ram_pct": _json_safe_number(_max_valid(ram_vals)),
        "max_cpu_pct": _json_safe_number(_max_valid(cpu_vals)),
        "max_gpu_pct": _json_safe_number(_max_valid(gpu_vals)),
        "avg_power_w": _json_safe_number(_avg_valid(power_vals)),
        "total_frames_processed": int(len(time_rows)),
        "duration_s": _json_safe_number(duration_s),
    }
    if extra_summary:
        summary.update(dict(extra_summary))

    with (profile_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def ceil_div_pos(a: int, b: int) -> int:
    a_i = int(a)
    b_i = int(b)
    if a_i <= 0:
        raise ValueError(f"Expected positive integer, got {a_i}.")
    if b_i <= 0:
        raise ValueError(f"Expected positive divisor, got {b_i}.")
    return (a_i + b_i - 1) // b_i


def resolve_path(path: str, *, desc: str) -> Path:
    """
    Resolve a user-provided path in a forgiving way:
      1) as given (relative to CWD)
      2) relative to this repo root (where this file lives)
      3) relative to MotionBERT root (repo_root/models/MotionBERT)
    """
    p = Path(path).expanduser()
    if p.exists():
        return p

    repo_rel = (_REPO_ROOT / path).expanduser()
    if repo_rel.exists():
        return repo_rel

    mb_rel = (_MB_ROOT / path).expanduser()
    if mb_rel.exists():
        return mb_rel

    raise FileNotFoundError(f"{desc} not found: {path}")


def resolve_checkpoint_path(ckpt: str) -> Path:
    """
    Accept either:
      - a checkpoint file (e.g. best_epoch.bin)
      - a checkpoint directory containing best_epoch.bin
    """
    p = resolve_path(ckpt, desc="Checkpoint")
    if p.is_file():
        return p

    best = p / "best_epoch.bin"
    if best.exists():
        return best
    latest = p / "latest_epoch.bin"
    if latest.exists():
        return latest

    bins = sorted(p.glob("**/*.bin"), key=lambda x: x.stat().st_mtime, reverse=True)
    if bins:
        return bins[0]

    raise FileNotFoundError(f"No *.bin checkpoints found under: {p.as_posix()}")


def infer_fall_indices(class_names: List[str]) -> List[int]:
    fall_idx = []
    for i, n in enumerate(class_names):
        s = n.lower()
        if s.startswith("fall") or "falling" in s:
            fall_idx.append(i)
    return fall_idx


def interpolate_missing_joints_inplace(
    kxy: np.ndarray,
    ksc: np.ndarray,
    missing_conf_thres: float = 0.0,
) -> None:
    """
    Interpolate missing joints over time within ONE window.
    Missing definition (per joint, per frame): score <= threshold OR non-finite coords.
    """
    T = kxy.shape[0]
    V = kxy.shape[1]
    if V != 17 or kxy.shape[2] != 2:
        raise ValueError(f"Expected kxy (T,17,2), got {kxy.shape}")
    if ksc.shape != (T, 17):
        raise ValueError(f"Expected ksc (T,17), got {ksc.shape}")

    t_idx = np.arange(T, dtype=np.float64)

    for j in range(V):
        finite_joint = np.isfinite(kxy[:, j, 0]) & np.isfinite(kxy[:, j, 1])
        valid_joint = (ksc[:, j] > missing_conf_thres) & finite_joint
        n_valid_joint = int(np.sum(valid_joint))

        if n_valid_joint == 0:
            kxy[:, j, :] = 0.0
            ksc[:, j] = 0.0
            continue

        for a in range(2):
            valid = (ksc[:, j] > missing_conf_thres) & np.isfinite(kxy[:, j, a])
            idx = np.where(valid)[0]
            if idx.size >= 2:
                vals = kxy[idx, j, a].astype(np.float64)
                interp_all = np.interp(t_idx, idx.astype(np.float64), vals)
                invalid = ~valid
                kxy[invalid, j, a] = interp_all[invalid].astype(np.float32)
            elif idx.size == 1:
                kxy[:, j, a] = float(kxy[idx[0], j, a])
            else:
                kxy[:, j, a] = 0.0


def draw_hud(
    frame,
    lines,
    org=(10, 10),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    thickness=2,
    pad=8,
    line_gap=6,
    bg_color=(0, 0, 0),
    bg_alpha=0.6,
    text_color=(255, 255, 255),
):
    if not lines:
        return frame

    x0, y0 = org
    sizes = [cv2.getTextSize(str(s), font, font_scale, thickness)[0] for s in lines]
    max_w = max(w for w, h in sizes)
    total_h = sum(h for w, h in sizes) + line_gap * (len(lines) - 1)

    box_w = max_w + 2 * pad
    box_h = total_h + 2 * pad

    h_img, w_img = frame.shape[:2]
    x1 = min(w_img - 1, x0 + box_w)
    y1 = min(h_img - 1, y0 + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    frame = cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0)

    y = y0 + pad
    for (w, h), s in zip(sizes, lines):
        y += h
        cv2.putText(frame, str(s), (x0 + pad, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_gap

    return frame


def draw_pose(frame, xy: np.ndarray, conf: np.ndarray, conf_thres: float = 0.2, draw_skeleton: bool = True):
    for i in range(K):
        if conf[i] > conf_thres:
            x, y = int(xy[i, 0]), int(xy[i, 1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    if draw_skeleton:
        for a, b in SKELETON:
            if conf[a] > conf_thres and conf[b] > conf_thres:
                ax, ay = int(xy[a, 0]), int(xy[a, 1])
                bx, by = int(xy[b, 0]), int(xy[b, 1])
                cv2.line(frame, (ax, ay), (bx, by), (0, 255, 255), 2)
    return frame


def open_video_writer(save_path: Path, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = save_path.suffix.lower()
    if suffix in {".avi"}:
        codecs = ["XVID", "MJPG", "mp4v"]
    elif suffix in {".mp4", ".m4v", ".mov"}:
        codecs = ["mp4v", "avc1", "H264", "MJPG"]
    else:
        codecs = ["mp4v", "MJPG"]

    w, h = int(frame_size[0]), int(frame_size[1])
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(save_path), fourcc, float(fps), (w, h))
        if writer.isOpened():
            print(f"[save] writing: {save_path.as_posix()} ({w}x{h} @{float(fps):.2f}fps, codec={codec})")
            return writer

    raise RuntimeError(f"Could not open VideoWriter for: {save_path} (tried codecs={codecs})")


def _clean_state_dict_for_model(state: dict, model: nn.Module) -> dict:
    """
    Flexibly handle DataParallel 'module.' prefixes.
    """
    if not isinstance(state, dict):
        return state
    state_keys = list(state.keys())
    has_module_prefix = any(k.startswith("module.") for k in state_keys)
    model_is_dp = isinstance(model, nn.DataParallel)

    if has_module_prefix and not model_is_dp:
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    if (not has_module_prefix) and model_is_dp:
        return {("module." + k): v for k, v in state.items()}
    return state


def build_windows(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    img_shape: Tuple[int, int],
    win_len: int,
    win_step: int,
    pad_tail: bool,
    missing_conf_thres: float,
    drop_empty_windows: bool,
    video_stem: str,
) -> Tuple[List[str], List[dict]]:
    """Return (split_list, annotations) with MotionBERT action pkl schema."""
    annotations: List[dict] = []
    split_list: List[str] = []

    T_total = int(kpts_xy.shape[0])
    if T_total <= 0:
        return split_list, annotations

    for start in range(0, T_total, win_step):
        end = start + win_len
        if end > T_total:
            if not pad_tail:
                break
            pad_n = end - T_total
            if pad_n >= win_len:
                break

        frame_dir = f"{video_stem}_s{start}_len{win_len}"

        raw_kxy = kpts_xy[start : min(end, T_total)].astype(np.float32)
        raw_ksc = kpts_conf[start : min(end, T_total)].astype(np.float32)

        if pad_tail and end > T_total:
            last_xy = raw_kxy[-1:, :, :]
            last_sc = raw_ksc[-1:, :]
            raw_kxy = np.concatenate([raw_kxy, np.repeat(last_xy, pad_n, axis=0)], axis=0)
            raw_ksc = np.concatenate([raw_ksc, np.repeat(last_sc, pad_n, axis=0)], axis=0)

        if raw_kxy.shape[0] != win_len or raw_ksc.shape[0] != win_len:
            continue

        kxy = raw_kxy.copy()
        ksc = raw_ksc.copy()

        nonfinite_xy = ~np.isfinite(kxy)
        nonfinite_sc = ~np.isfinite(ksc)
        if nonfinite_xy.any() or nonfinite_sc.any():
            kxy[nonfinite_xy] = 0.0
            ksc[nonfinite_sc] = 0.0
            nonfinite_joint = nonfinite_xy.any(axis=2) | nonfinite_sc
            ksc[nonfinite_joint] = 0.0

        if (ksc < 0).any() or (ksc > 1).any():
            ksc = np.clip(ksc, 0.0, 1.0)

        interpolate_missing_joints_inplace(kxy, ksc, missing_conf_thres=missing_conf_thres)

        if drop_empty_windows:
            if np.all(ksc <= missing_conf_thres):
                continue
            if (np.ptp(kxy[..., 0]) < 1e-6) and (np.ptp(kxy[..., 1]) < 1e-6):
                continue

        keypoint = kxy[None, ...].astype(np.float32)
        keypoint_score = ksc[None, ...].astype(np.float32)

        annotations.append(
            {
                "frame_dir": frame_dir,
                "total_frames": int(win_len),
                "img_shape": (int(img_shape[0]), int(img_shape[1])),
                "keypoint": keypoint,
                "keypoint_score": keypoint_score,
                "label": 0,
            }
        )
        split_list.append(frame_dir)

        if (not pad_tail) and end >= T_total:
            break

    return split_list, annotations


def build_motion_from_annotation(
    ann: dict,
    clip_len: int,
    scale_range: Optional[List[float]],
) -> np.ndarray:
    """
    Build MotionBERT ActionNet input exactly like MotionBERT's dataset_action.NTURGBD:
      resample -> make_cam -> human_tracking -> coco2h36m -> concat conf -> crop_scale
    """
    keypoint = ann["keypoint"]  # (1, T, 17, 2)
    keypoint_score = ann["keypoint_score"]  # (1, T, 17)
    img_shape = ann["img_shape"]

    resample_id = resample(ori_len=int(ann["total_frames"]), target_len=int(clip_len), randomness=False)
    motion_cam = make_cam(x=keypoint, img_shape=img_shape)
    motion_cam = human_tracking(motion_cam)
    motion_cam = coco2h36m(motion_cam)
    motion_conf = keypoint_score[..., None]
    motion = np.concatenate((motion_cam[:, resample_id], motion_conf[:, resample_id]), axis=-1)

    if motion.shape[0] == 1:
        fake = np.zeros(motion.shape, dtype=motion.dtype)
        motion = np.concatenate((motion, fake), axis=0)

    if scale_range:
        motion = crop_scale(motion, scale_range=scale_range)

    return motion.astype(np.float32)


def load_labels_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = resolve_path(path, desc="Labels file")
    names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return names or None


def pad_or_trim(names: List[str], num_classes: int) -> List[str]:
    if len(names) != int(num_classes):
        if len(names) > int(num_classes):
            names = names[: int(num_classes)]
        else:
            for i in range(len(names), int(num_classes)):
                names.append(f"class_{i}")
    return names


def window_start_end_from_frame_dir(frame_dir: str, total_frames: int, frame_step: int = 1) -> Tuple[int, int]:
    start_frame = 0
    try:
        parts = frame_dir.split("_s", 1)
        if len(parts) > 1:
            start_frame = int(parts[1].split("_len", 1)[0])
    except Exception:
        start_frame = 0

    step = max(1, int(frame_step))
    start_raw = int(start_frame) * step
    end_raw = int(start_raw) + (max(1, int(total_frames)) - 1) * step
    return int(start_raw), int(end_raw)


def predict_one_window(
    *,
    model: nn.Module,
    device: str,
    ann: dict,
    frame_dir: str,
    clip_len: int,
    scale_range: Optional[List[float]],
    merge_fall: bool,
    fall_idx: List[int],
    class_names_out: List[str],
    unmerged_len_expected: int,
    merged_len_expected: int,
    frame_step: int = 1,
) -> dict:
    motion = build_motion_from_annotation(ann, clip_len=clip_len, scale_range=scale_range)

    X = torch.from_numpy(motion).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(X)
        probs_t = torch.softmax(out, dim=1).squeeze(0).detach().cpu().numpy()

    if merge_fall:
        if probs_t.shape[0] == merged_len_expected:
            merged_probs = probs_t
            fall_prob = float(merged_probs[0]) if merged_probs.size else 0.0
        elif probs_t.shape[0] == unmerged_len_expected:
            fall_prob = float(np.sum(probs_t[FALL_CLASS_IDS_DEFAULT]))
            nonfall_probs = [probs_t[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
            merged_probs = np.array([fall_prob] + nonfall_probs, dtype=np.float32)
        else:
            merged_probs = probs_t
            fall_prob = float(np.sum(probs_t[fall_idx])) if fall_idx else 0.0
    else:
        merged_probs = probs_t
        fall_prob = float(np.sum(probs_t[fall_idx])) if fall_idx else 0.0

    pred_id = int(np.argmax(merged_probs))
    pred_conf = float(np.max(merged_probs))
    p_fall = float(fall_prob)

    start_frame, end_frame = window_start_end_from_frame_dir(
        frame_dir,
        int(ann["total_frames"]),
        frame_step=int(frame_step),
    )
    pred_name = class_names_out[pred_id] if 0 <= pred_id < len(class_names_out) else str(pred_id)

    return {
        "frame_dir": str(frame_dir),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "pred_id": int(pred_id),
        "pred_name": str(pred_name),
        "pred_conf": float(pred_conf),
        "p_fall": float(p_fall),
    }


def stream_infer_and_display(
    *,
    ckpt_path: Path,
    video_path: Path,
    cfg,
    yolo_path: Path,
    device: str,
    clip_len: int,
    win_step: int,
    frame_step: int,
    clip_len_raw: int,
    win_step_raw: int,
    save_path: Optional[Path],
    profile_enabled: bool,
    profile_out_dir: Optional[Path],
    profile_duration_s: float,
    hw_sample_hz: float,
    benchmark_enabled: bool,
    no_display: bool,
    args: argparse.Namespace,
) -> int:
    """
    One-pass streaming: read frames -> YOLO pose -> window inference -> imshow.

    Playback targets the source FPS (or --display-fps). If processing can't keep up, the display slows down.
    In benchmark mode, EOF rewinds/reopens the same video until duration expires.
    """

    win_step = max(1, int(win_step))
    frame_step = max(1, int(frame_step))
    missing_conf_thres = float(args.missing_conf_thres)
    drop_empty_windows = not bool(args.keep_empty_windows)
    pad_tail = bool(args.pad_tail)
    profile_enabled = bool(profile_enabled)
    benchmark_enabled = bool(benchmark_enabled)
    no_display = bool(no_display)
    profile_duration_s = max(0.0, float(profile_duration_s))
    hw_sample_hz = max(0.1, float(hw_sample_hz))

    # Models
    pose_model = YOLO(str(yolo_path))

    model_backbone = load_backbone(cfg)
    model = ActionNet(
        backbone=model_backbone,
        dim_rep=getattr(cfg, "dim_rep", 512),
        num_classes=int(getattr(cfg, "action_classes", 11)),
        dropout_ratio=getattr(cfg, "dropout_ratio", 0.0),
        version=getattr(cfg, "model_version", "class"),
        hidden_dim=getattr(cfg, "hidden_dim", 2048),
        num_joints=getattr(cfg, "num_joints", 17),
    )

    use_dp = device.startswith("cuda") and torch.cuda.device_count() > 1
    if use_dp:
        model = nn.DataParallel(model)
    model = model.to(device)
    print("Model device:", next(model.parameters()).device)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = _clean_state_dict_for_model(state, model)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Labels / inference config
    num_classes = int(getattr(cfg, "action_classes", 11))
    scale_range = getattr(cfg, "scale_range_test", None)

    labels_file_names = load_labels_file(args.labels_file)

    unmerged_len_expected = len(CLASS_NAMES_DEFAULT)
    merged_len_expected = len(CLASS_NAMES_MERGED_DEFAULT)

    merge_fall = (not bool(args.no_merge_fall)) and (num_classes == unmerged_len_expected)

    if labels_file_names is not None:
        if merge_fall and len(labels_file_names) == merged_len_expected:
            class_names_out = list(labels_file_names)
        else:
            base_names = pad_or_trim(list(labels_file_names), num_classes)
            if merge_fall:
                class_names_out = ["Fall"] + [base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
            else:
                class_names_out = base_names
    else:
        if num_classes == merged_len_expected:
            class_names_out = list(CLASS_NAMES_MERGED_DEFAULT)
        else:
            base_names = pad_or_trim(list(CLASS_NAMES_DEFAULT), num_classes)
            if merge_fall:
                class_names_out = ["Fall"] + [base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
            else:
                class_names_out = base_names

    fall_idx = infer_fall_indices(class_names_out)

    predict_kwargs = dict(
        model=model,
        device=device,
        clip_len=int(clip_len),
        scale_range=scale_range,
        merge_fall=merge_fall,
        fall_idx=fall_idx,
        class_names_out=class_names_out,
        unmerged_len_expected=unmerged_len_expected,
        merged_len_expected=merged_len_expected,
        frame_step=frame_step,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_pkl = Path(args.out_pkl)

    def write_header(writer: csv.writer) -> None:
        writer.writerow(
            [
                "frame_dir",
                "start_frame",
                "end_frame",
                "pred_id",
                "pred_name",
                "pred_conf",
                "p_fall",
            ]
        )

    def write_pred_row(writer: csv.writer, pred: dict) -> None:
        writer.writerow(
            [
                pred["frame_dir"],
                pred["start_frame"],
                pred["end_frame"],
                pred["pred_id"],
                pred["pred_name"],
                f"{float(pred['pred_conf']):.6f}",
                f"{float(pred['p_fall']):.6f}",
            ]
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path.as_posix()}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(src_fps) or src_fps <= 1e-3:
        src_fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    user_fps = float(args.display_fps) if args.display_fps is not None else None
    fps_target = float(user_fps) if (user_fps is not None and np.isfinite(user_fps) and user_fps > 0.0) else float(src_fps)
    if not np.isfinite(fps_target) or fps_target <= 1e-3:
        fps_target = 30.0
    frame_period_s = 1.0 / float(fps_target)

    window_name = "MotionBERT Inference"
    if not no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print(f"[Display] Target FPS={fps_target:.2f} (source={src_fps:.2f}). Press 'q' or Esc to quit.", flush=True)
    else:
        print(f"[Display] Headless mode enabled. Target FPS={fps_target:.2f} (source={src_fps:.2f}).", flush=True)

    video_writer: Optional[cv2.VideoWriter] = None
    out_w: Optional[int] = None
    out_h: Optional[int] = None

    frames_buf: "deque[np.ndarray]" = deque()
    xy_buf: "deque[np.ndarray]" = deque()
    cf_buf: "deque[np.ndarray]" = deque()
    prep_ms_buf: "deque[float]" = deque()
    yolo_ms_buf: "deque[float]" = deque()

    all_xy: List[np.ndarray] = []
    all_cf: List[np.ndarray] = []

    img_shape: Optional[Tuple[int, int]] = None
    processed_total = 0  # raw frames processed
    sampled_total = 0  # sampled frames used for MotionBERT windows
    display_idx = 0
    cap_done = False

    window_preds: dict[int, dict] = {}
    window_stage_ms: dict[int, Tuple[float, float]] = {}
    skipped_windows: set[int] = set()
    next_win_start = 0
    last_xy = np.zeros((17, 2), dtype=np.float32)
    last_cf = np.zeros((17,), dtype=np.float32)
    benchmark_loop_count = 0

    time_rows: List[Dict[str, Any]] = []
    hw_rows: List[Dict[str, Any]] = []
    hw_sampler: Optional[HardwareSampler] = None
    profile_run_t0 = time.perf_counter()
    last_hw_print_t = 0.0
    hw_print_interval_s = max(0.5, 1.0 / max(hw_sample_hz, 1e-3))

    def pose_on_frame(frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        results = pose_model.predict(
            source=frame_bgr,
            imgsz=int(args.imgsz),
            conf=float(args.conf_thres),
            verbose=False,
            device=device,
        )

        kpts_xy = np.zeros((17, 2), dtype=np.float32)
        kpts_conf = np.zeros((17,), dtype=np.float32)

        if results and len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints
            xy_all = kpts.xy.cpu().numpy() if hasattr(kpts.xy, "cpu") else np.array(kpts.xy)
            cf_all = kpts.conf.cpu().numpy() if hasattr(kpts.conf, "cpu") else np.array(kpts.conf)

            if xy_all.ndim == 3 and xy_all.shape[0] > 0:
                scores = cf_all.sum(axis=1) if (cf_all.ndim == 2 and cf_all.shape[0] == xy_all.shape[0]) else None
                best = int(np.argmax(scores)) if scores is not None else 0
                kpts_xy = xy_all[best].astype(np.float32)
                if cf_all.ndim == 2 and cf_all.shape[0] == xy_all.shape[0]:
                    kpts_conf = cf_all[best].astype(np.float32)
                else:
                    kpts_conf = np.ones((17,), dtype=np.float32)

        return kpts_xy, kpts_conf

    def stop_due_profile_duration() -> bool:
        if not profile_enabled:
            return False
        if profile_duration_s <= 0.0:
            return False
        return (time.perf_counter() - profile_run_t0) >= profile_duration_s

    def reset_tracking_state_for_new_video_loop() -> None:
        nonlocal last_xy, last_cf
        last_xy = np.zeros((17, 2), dtype=np.float32)
        last_cf = np.zeros((17,), dtype=np.float32)

    def rewind_or_reopen_capture_for_benchmark(force_reopen: bool = False) -> bool:
        nonlocal cap, benchmark_loop_count

        used_seek = False
        if not force_reopen:
            used_seek = bool(cap.set(cv2.CAP_PROP_POS_FRAMES, 0))
        if not used_seek:
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False

        benchmark_loop_count += 1
        reset_tracking_state_for_new_video_loop()
        method = "seek" if used_seek else "reopen"
        print(f"[benchmark] restarted video loop #{int(benchmark_loop_count)} via {method}")
        return True

    def process_next_frame() -> bool:
        nonlocal processed_total, sampled_total, cap_done, img_shape, last_xy, last_cf

        if cap_done:
            return False

        eof_recovery_attempts = 0
        while True:
            cap_read_ms = 0.0
            if profile_enabled:
                t_cap0 = time.perf_counter()
            ok, frame = cap.read()
            if profile_enabled:
                cap_read_ms = (time.perf_counter() - t_cap0) * 1000.0
            if ok:
                break
            if benchmark_enabled and (not stop_due_profile_duration()) and eof_recovery_attempts < 2:
                force_reopen = eof_recovery_attempts > 0
                if rewind_or_reopen_capture_for_benchmark(force_reopen=force_reopen):
                    eof_recovery_attempts += 1
                    continue
                print("[benchmark][WARN] Could not rewind/reopen video stream; stopping benchmark.")
            cap_done = True
            return False

        if img_shape is None:
            h, w = frame.shape[:2]
            img_shape = (int(h), int(w))

        raw_idx = int(processed_total)
        do_pose = (int(raw_idx) % int(frame_step)) == 0

        if do_pose:
            yolo_infer_ms = 0.0
            if profile_enabled:
                t_yolo0 = time.perf_counter()
            xy, cf = pose_on_frame(frame)
            if profile_enabled:
                yolo_infer_ms = (time.perf_counter() - t_yolo0) * 1000.0
            all_xy.append(xy)
            all_cf.append(cf)
            sampled_total += 1
            last_xy = xy
            last_cf = cf
        else:
            yolo_infer_ms = 0.0
            xy = last_xy
            cf = last_cf

        frames_buf.append(frame)
        xy_buf.append(xy)
        cf_buf.append(cf)
        if profile_enabled:
            prep_ms_buf.append(float(cap_read_ms))
            yolo_ms_buf.append(float(yolo_infer_ms))
        processed_total += 1

        if (not benchmark_enabled) and args.limit_frames is not None and processed_total >= int(args.limit_frames):
            cap_done = True

        return True

    def make_window_annotation(start: int) -> Optional[Tuple[str, dict]]:
        if img_shape is None:
            return None

        end = int(start) + int(clip_len)
        if end <= int(sampled_total):
            raw_kxy = np.stack(all_xy[start:end], axis=0).astype(np.float32)
            raw_ksc = np.stack(all_cf[start:end], axis=0).astype(np.float32)
        else:
            if not cap_done or (not pad_tail):
                return None
            if start >= int(sampled_total):
                return None
            pad_n = int(end - int(sampled_total))
            if pad_n >= int(clip_len):
                return None
            raw_kxy = np.stack(all_xy[start:sampled_total], axis=0).astype(np.float32)
            raw_ksc = np.stack(all_cf[start:sampled_total], axis=0).astype(np.float32)
            last_xy = raw_kxy[-1:, :, :]
            last_sc = raw_ksc[-1:, :]
            raw_kxy = np.concatenate([raw_kxy, np.repeat(last_xy, pad_n, axis=0)], axis=0)
            raw_ksc = np.concatenate([raw_ksc, np.repeat(last_sc, pad_n, axis=0)], axis=0)

        if raw_kxy.shape != (int(clip_len), 17, 2) or raw_ksc.shape != (int(clip_len), 17):
            return None

        kxy = raw_kxy.copy()
        ksc = raw_ksc.copy()

        nonfinite_xy = ~np.isfinite(kxy)
        nonfinite_sc = ~np.isfinite(ksc)
        if nonfinite_xy.any() or nonfinite_sc.any():
            kxy[nonfinite_xy] = 0.0
            ksc[nonfinite_sc] = 0.0
            nonfinite_joint = nonfinite_xy.any(axis=2) | nonfinite_sc
            ksc[nonfinite_joint] = 0.0

        if (ksc < 0).any() or (ksc > 1).any():
            ksc = np.clip(ksc, 0.0, 1.0)

        interpolate_missing_joints_inplace(kxy, ksc, missing_conf_thres=missing_conf_thres)

        if drop_empty_windows:
            if np.all(ksc <= missing_conf_thres):
                return None
            if (np.ptp(kxy[..., 0]) < 1e-6) and (np.ptp(kxy[..., 1]) < 1e-6):
                return None

        frame_dir = f"{video_path.stem}_s{start}_len{clip_len}"
        ann = {
            "frame_dir": frame_dir,
            "total_frames": int(clip_len),
            "img_shape": (int(img_shape[0]), int(img_shape[1])),
            "keypoint": kxy[None, ...].astype(np.float32),
            "keypoint_score": ksc[None, ...].astype(np.float32),
            "label": 0,
        }
        return frame_dir, ann

    def compute_window_pred(start: int, *, writer: csv.writer) -> Optional[dict]:
        if int(start) in window_preds:
            return window_preds[int(start)]
        if int(start) in skipped_windows:
            return None

        window = make_window_annotation(int(start))
        if window is None:
            # Window is either not ready yet, or it was dropped (empty / too short)
            if cap_done and (not pad_tail) and int(sampled_total) < int(start) + int(clip_len):
                skipped_windows.add(int(start))
            if drop_empty_windows and int(sampled_total) >= int(start) + int(clip_len):
                skipped_windows.add(int(start))
            return None

        frame_dir, ann = window
        temporal_infer_ms = 0.0
        if profile_enabled:
            t_pred0 = time.perf_counter()
        pred = predict_one_window(ann=ann, frame_dir=frame_dir, **predict_kwargs)
        if profile_enabled:
            temporal_infer_ms = (time.perf_counter() - t_pred0) * 1000.0
        window_preds[int(start)] = pred
        if profile_enabled:
            # MotionBERT window building + model inference are bundled inside predict_one_window.
            window_stage_ms[int(start)] = (0.0, float(temporal_infer_ms))
        write_pred_row(writer, pred)
        return pred

    def compute_ready_windows(*, writer: csv.writer) -> None:
        nonlocal next_win_start

        while True:
            if int(next_win_start) in window_preds or int(next_win_start) in skipped_windows:
                next_win_start = int(next_win_start) + int(win_step)
                continue

            if not cap_done:
                if int(sampled_total) >= int(next_win_start) + int(clip_len):
                    compute_window_pred(int(next_win_start), writer=writer)
                    next_win_start = int(next_win_start) + int(win_step)
                    continue
                break

            # cap_done
            if int(sampled_total) >= int(next_win_start) + int(clip_len):
                compute_window_pred(int(next_win_start), writer=writer)
                next_win_start = int(next_win_start) + int(win_step)
                continue
            if pad_tail and int(next_win_start) < int(sampled_total):
                compute_window_pred(int(next_win_start), writer=writer)
                next_win_start = int(next_win_start) + int(win_step)
                continue
            break

    def get_pred_for_frame(frame_idx: int) -> Optional[dict]:
        if frame_idx < 0:
            return None
        sample_idx = int(frame_idx) // int(frame_step)
        ws = (int(sample_idx) // int(win_step)) * int(win_step)
        pred = window_preds.get(int(ws))
        if pred is not None:
            return pred

        # If the exact window was dropped (empty), fall back to the most recent available window covering frame_idx.
        s = int(ws) - int(win_step)
        while s >= 0:
            p = window_preds.get(int(s))
            if p is not None and int(frame_idx) <= int(p["end_frame"]):
                return p
            s -= int(win_step)
        return None

    try:
        if profile_enabled:
            if profile_out_dir is None:
                raise RuntimeError("Profile mode enabled but profile output directory is unset.")
            profile_out_dir.mkdir(parents=True, exist_ok=True)
            hw_sampler = HardwareSampler(sample_hz=hw_sample_hz)
            hw_backend = hw_sampler.start()
            print(f"[profile] enabled -> {profile_out_dir.as_posix()}")
            print(f"[profile] hw backend: {hw_backend}")

        with out_csv.open("w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            write_header(writer)

            # Warm up: read enough frames to make the first window prediction.
            while int(sampled_total) < int(clip_len) and not cap_done and not stop_due_profile_duration():
                process_next_frame()
            if int(processed_total) <= 0:
                raise RuntimeError("Video had 0 frames.")

            compute_window_pred(0, writer=writer)
            next_win_start = int(win_step)

            if save_path is not None and frames_buf:
                out_h, out_w = frames_buf[0].shape[:2]
                video_writer = open_video_writer(
                    save_path=save_path,
                    fps=float(src_fps),
                    frame_size=(int(out_w), int(out_h)),
                )

            fps_ema: Optional[float] = None
            ema_alpha = 0.1
            user_exit = False

            while True:
                if stop_due_profile_duration():
                    break
                if not frames_buf and cap_done:
                    break

                t_frame0 = time.perf_counter()
                display_sample_idx = int(display_idx) // int(frame_step)

                target_sampled = int(display_sample_idx) + int(clip_len) + 1
                while (not cap_done) and int(sampled_total) < int(target_sampled) and (not stop_due_profile_duration()):
                    process_next_frame()

                compute_ready_windows(writer=writer)

                if profile_enabled and hw_sampler is not None:
                    now_t = time.perf_counter()
                    if (now_t - last_hw_print_t) >= hw_print_interval_s:
                        latest = hw_sampler.get_latest_sample()
                        if latest is not None:
                            print(
                                "[hw] "
                                f"t={_fmt_live_metric(latest.get('t_s', np.nan), 's')} "
                                f"gpu={_fmt_live_metric(latest.get('gpu_pct', np.nan), '%')} "
                                f"cpu={_fmt_live_metric(latest.get('cpu_pct', np.nan), '%')} "
                                f"ram={_fmt_live_metric(latest.get('ram_used_pct', np.nan), '%')} "
                                f"cpu_t={_fmt_live_metric(latest.get('cpu_temp_c', np.nan), 'C')} "
                                f"gpu_t={_fmt_live_metric(latest.get('gpu_temp_c', np.nan), 'C')} "
                                f"pwr={_fmt_live_metric(latest.get('power_w', np.nan), 'W')}"
                            )
                        else:
                            print("[hw] waiting for first hardware sample...")
                        last_hw_print_t = now_t

                if not frames_buf:
                    continue

                post_extra_ms = 0.0
                if profile_enabled:
                    t_post_extra0 = time.perf_counter()

                win_start = (int(display_sample_idx) // int(win_step)) * int(win_step)
                if win_start not in window_preds and win_start not in skipped_windows:
                    while (
                        (not cap_done)
                        and int(sampled_total) < int(win_start) + int(clip_len)
                        and (not stop_due_profile_duration())
                    ):
                        process_next_frame()
                        compute_ready_windows(writer=writer)
                    compute_window_pred(int(win_start), writer=writer)

                pred = get_pred_for_frame(int(display_idx))

                frame = frames_buf[0].copy()
                xy = xy_buf[0]
                cf = cf_buf[0]

                preprocess_ms = float(prep_ms_buf[0]) if profile_enabled and prep_ms_buf else 0.0
                yolo_infer_ms = float(yolo_ms_buf[0]) if profile_enabled and yolo_ms_buf else 0.0
                win_feature_ms, temporal_infer_ms = window_stage_ms.get(int(win_start), (0.0, 0.0))

                if profile_enabled:
                    post_extra_ms = (time.perf_counter() - t_post_extra0) * 1000.0

                draw_ms = 0.0
                writer_ms = 0.0
                display_wait_ms = 0.0

                t_vis0 = time.perf_counter()
                t_draw0 = time.perf_counter()
                frame = draw_pose(
                    frame,
                    xy,
                    cf,
                    conf_thres=float(args.display_conf_thres),
                    draw_skeleton=True,
                )

                frame_info = f"frame {int(display_idx) + 1}"
                if int(frame_count) > 0:
                    frame_info += f"/{int(frame_count)}"

                win_id = int(win_start) // max(1, int(win_step))
                win_start_raw = int(win_start) * int(frame_step)
                fps_for_hud = float(fps_ema) if fps_ema is not None else float(fps_target)
                hud = [
                    frame_info,
                    f"fps: {float(fps_for_hud):.1f} (target {float(fps_target):.1f})",
                    f"window {win_id} (sample_start={win_start}, raw_start={win_start_raw})",
                ]
                if pred is not None:
                    hud.append(f"pred: {pred['pred_name']} ({float(pred['pred_conf']):.2f})")
                    hud.append(f"fall_prob: {float(pred['p_fall']):.2f}")
                    hud.append(f"win: {int(pred['start_frame'])}-{int(pred['end_frame'])}")
                else:
                    hud.append("pred: ... (warming up)")
                hud.append(
                    f"T={int(clip_len)} stride={int(win_step)} sampled "
                    f"(raw T/stride={int(clip_len_raw)}/{int(win_step_raw)}, k={int(frame_step)})"
                )

                frame = draw_hud(frame, hud)
                draw_ms = (time.perf_counter() - t_draw0) * 1000.0

                if video_writer is not None and out_w is not None and out_h is not None:
                    t_writer0 = time.perf_counter()
                    frame_h, frame_w = frame.shape[:2]
                    frame_to_write = frame
                    if frame_h != int(out_h) or frame_w != int(out_w):
                        frame_to_write = cv2.resize(frame, (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR)
                    video_writer.write(frame_to_write)
                    writer_ms = (time.perf_counter() - t_writer0) * 1000.0

                key = -1
                if not no_display:
                    t_display0 = time.perf_counter()
                    cv2.imshow(window_name, frame)

                    elapsed = float(time.perf_counter() - t_frame0)
                    wait_s = float(frame_period_s) - elapsed
                    wait_ms = max(1, int(round(1000.0 * wait_s))) if wait_s > 0.0 else 1
                    key = cv2.waitKey(int(wait_ms)) & 0xFF
                    display_wait_ms = (time.perf_counter() - t_display0) * 1000.0

                visualisation_ms = (time.perf_counter() - t_vis0) * 1000.0
                total_ms = (time.perf_counter() - t_frame0) * 1000.0
                inst_fps = 1000.0 / max(1e-6, total_ms)
                fps_ema = inst_fps if fps_ema is None else (1.0 - ema_alpha) * float(fps_ema) + ema_alpha * inst_fps

                if profile_enabled:
                    inference_ms = float(yolo_infer_ms) + float(temporal_infer_ms)
                    postprocess_ms = float(win_feature_ms) + float(post_extra_ms)
                    time_rows.append(
                        {
                            "frame_idx": int(display_idx),
                            "t_s": float(time.perf_counter() - profile_run_t0),
                            "fps": float(inst_fps),
                            "fps_ema": float(fps_ema),
                            "preprocess_ms": float(preprocess_ms),
                            "inference_ms": float(inference_ms),
                            "postprocess_ms": float(postprocess_ms),
                            "visualisation_ms": float(visualisation_ms),
                            "total_ms": float(total_ms),
                            "cap_read_ms": float(preprocess_ms),
                            "yolo_infer_ms": float(yolo_infer_ms),
                            "window_feature_ms": float(win_feature_ms),
                            "temporal_infer_ms": float(temporal_infer_ms),
                            "label_select_ms": float(post_extra_ms),
                            "draw_ms": float(draw_ms),
                            "writer_ms": float(writer_ms),
                            "display_wait_ms": float(display_wait_ms),
                        }
                    )

                should_quit = (key in (ord("q"), 27))

                frames_buf.popleft()
                xy_buf.popleft()
                cf_buf.popleft()
                if profile_enabled:
                    if prep_ms_buf:
                        prep_ms_buf.popleft()
                    if yolo_ms_buf:
                        yolo_ms_buf.popleft()
                display_idx += 1

                if should_quit:
                    user_exit = True
                    break

            if (not user_exit) and cap_done and (not stop_due_profile_duration()):
                while True:
                    before = int(next_win_start)
                    compute_ready_windows(writer=writer)
                    if int(next_win_start) == before:
                        break
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not no_display:
            cv2.destroyAllWindows()
        if hw_sampler is not None:
            hw_sampler.stop()
            hw_rows = hw_sampler.get_samples()
        if profile_enabled and profile_out_dir is not None:
            try:
                _save_profile_artifacts(
                    profile_out=profile_out_dir,
                    time_rows=time_rows,
                    hw_rows=hw_rows,
                    extra_summary={
                        "benchmark_enabled": bool(benchmark_enabled),
                        "benchmark_duration_s": float(profile_duration_s) if benchmark_enabled else None,
                        "benchmark_duration_source": "BENCHMARK_DURATION_S" if os.getenv("BENCHMARK_DURATION_S") else "default",
                    },
                )
                print(f"[profile] wrote outputs to: {profile_out_dir.as_posix()}")
            except Exception as e:
                print(f"[WARN] Failed to save profiling outputs: {e}")

    if img_shape is None:
        raise RuntimeError("No frames read from video.")

    if not all_xy or not all_cf:
        raise RuntimeError(
            "No sampled frames were generated. Reduce --k/--frame-step or ensure the video has readable frames."
        )

    # Build and save MotionBERT action pkl from sampled frames only.
    kpts_xy = np.stack(all_xy, axis=0)  # (T_sampled,17,2)
    kpts_conf = np.stack(all_cf, axis=0)  # (T_sampled,17)
    split_list, annotations = build_windows(
        kpts_xy=kpts_xy,
        kpts_conf=kpts_conf,
        img_shape=img_shape,
        win_len=int(clip_len),
        win_step=int(win_step),
        pad_tail=pad_tail,
        missing_conf_thres=missing_conf_thres,
        drop_empty_windows=drop_empty_windows,
        video_stem=video_path.stem,
    )

    dataset = {"split": {"xsub_train": [], "xsub_val": split_list}, "annotations": annotations}
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved pkl: {out_pkl.as_posix()}")
    print(f"Saved predictions: {out_csv.as_posix()}")
    print(f"Windows: {len(split_list)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to MotionBERT checkpoint (*.bin) OR checkpoint directory (contains best_epoch.bin)",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="configs/action/MB_ft_UPFall_xsub_LITE.yaml",
        help="MotionBERT config yaml (can be relative to models/MotionBERT/)",
    )
    ap.add_argument("--video", type=str, required=True, help="Path to input mp4")
    ap.add_argument("--yolo-weights", type=str, default="pose_models/ultralytics/yolo11l-pose.pt")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument(
        "--win-len",
        type=int,
        default=None,
        help="Raw-frame window length (defaults to config.clip_len, then scaled by --k/--frame-step).",
    )
    ap.add_argument(
        "--win-step",
        type=int,
        default=16,
        help="Raw-frame window stride (scaled by --k/--frame-step for sampled inference).",
    )
    ap.add_argument(
        "--frame-step",
        "--k",
        type=int,
        default=1,
        help=(
            "Run YOLO pose every k raw frames (k>=1). "
            "Window length/stride are defined in raw frames and scaled to sampled frames using ceil division."
        ),
    )
    ap.add_argument("--pad-tail", action="store_true")
    ap.add_argument("--missing-conf-thres", type=float, default=0.0)
    ap.add_argument("--keep-empty-windows", action="store_true", default=False)
    ap.add_argument("--out-pkl", type=str, default="outputs/motionbert_video.pkl")
    ap.add_argument("--out-csv", type=str, default="outputs/motionbert_video_preds.csv")
    ap.add_argument("--labels-file", type=str, default=None)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--display", action="store_true", help="Display video with pose + streaming window prediction (FPS HUD)")
    ap.add_argument("--display-conf-thres", type=float, default=0.2, help="Keypoint conf threshold for drawing")
    ap.add_argument("--display-fps", type=float, default=None, help="Playback FPS for display (default: video FPS)")
    ap.add_argument("--profile", type=int, default=0, help="Enable profiling outputs (0/1).")
    ap.add_argument(
        "--profile-out",
        type=str,
        default=None,
        help="Base directory for profiling outputs. A unique per-run subdirectory is created (timestamp + model).",
    )
    ap.add_argument("--profile-duration-s", type=float, default=0.0, help="0 => full run, else stop after N seconds.")
    ap.add_argument("--benchmark", type=int, default=0, help="Loop inference on the same video for benchmark duration (0/1). Requires CUDA.")
    ap.add_argument("--hw-sample-hz", type=float, default=1.0, help="Hardware metrics sample rate (Hz).")
    ap.add_argument("--no-display", type=int, default=0, help="Run headless: skip imshow/waitKey (0/1).")
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save annotated output video (e.g. out.mp4). If a directory, writes <video_stem>_annotated.mp4 inside.",
    )
    ap.add_argument("--no-merge-fall", action="store_true", help="Disable merging the first five fall labels into one class")
    args = ap.parse_args()

    benchmark_enabled = bool(int(args.benchmark))
    profile_enabled = bool(int(args.profile)) or benchmark_enabled
    no_display = bool(int(args.no_display))
    if benchmark_enabled:
        try:
            profile_duration_s = get_benchmark_duration_s(default_s=600.0)
        except ValueError as e:
            print(f"[benchmark][ERROR] {e}", file=sys.stderr)
            return 2
    else:
        profile_duration_s = max(0.0, float(args.profile_duration_s))
    hw_sample_hz = max(0.1, float(args.hw_sample_hz))

    device = pick_device(args.device)
    try:
        assert_benchmark_device_ok(benchmark=benchmark_enabled, device=device)
    except ValueError as e:
        print(f"[benchmark][ERROR] {e}", file=sys.stderr)
        return 2
    print(
        f"[runtime] device={device} "
        f"(requested={args.device if args.device else 'auto'}, cuda_available={torch.cuda.is_available()})"
    )
    if benchmark_enabled:
        duration_src = "BENCHMARK_DURATION_S" if os.getenv("BENCHMARK_DURATION_S") else "default"
        print(f"[benchmark] enabled: duration_s={float(profile_duration_s):.3f} (source={duration_src}), profile=1")

    ckpt_path = resolve_checkpoint_path(args.model)
    video_path = resolve_path(args.video, desc="Video")
    cfg_path = resolve_path(args.config, desc="Config")
    yolo_path = resolve_path(args.yolo_weights, desc="YOLO weights")

    cfg = get_config(str(cfg_path))
    clip_len_raw = int(args.win_len) if args.win_len is not None else int(getattr(cfg, "clip_len", 64))
    win_step_raw = max(1, int(args.win_step))
    frame_step = int(args.frame_step)
    if int(frame_step) <= 0:
        raise ValueError("--frame-step/--k must be >= 1.")
    if int(clip_len_raw) <= 0:
        raise ValueError(f"Invalid window length: {clip_len_raw}.")

    clip_len = max(1, int(ceil_div_pos(int(clip_len_raw), int(frame_step))))
    win_step = max(1, int(ceil_div_pos(int(win_step_raw), int(frame_step))))
    if int(frame_step) > 1 and (
        (int(clip_len_raw) % int(frame_step)) != 0 or (int(win_step_raw) % int(frame_step)) != 0
    ):
        print(
            f"[window][WARN] raw clip_len/win_step ({int(clip_len_raw)}/{int(win_step_raw)}) "
            f"are not divisible by frame_step={int(frame_step)}; using ceil division for sampled windows."
        )
    print(
        f"[window] raw clip_len/win_step={int(clip_len_raw)}/{int(win_step_raw)} "
        f"-> sampled clip_len/win_step={int(clip_len)}/{int(win_step)} (k={int(frame_step)})"
    )

    save_path: Optional[Path] = None
    if args.save:
        save_arg = Path(args.save).expanduser()
        if str(args.save).endswith(("/", "\\")) or (save_arg.exists() and save_arg.is_dir()):
            save_path = save_arg / f"{video_path.stem}_annotated.mp4"
        else:
            save_path = save_arg
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".mp4")
        if not args.display:
            print("[WARN] --save provided; enabling --display for annotated video output.")
            args.display = True

    if benchmark_enabled and not args.display:
        print("[benchmark] enabling streaming path (--display) for timed looping.")
        args.display = True
    if profile_enabled and not args.display:
        print("[profile] enabling streaming path (--display) to collect profiling artifacts.")
        args.display = True
    if benchmark_enabled and args.limit_frames is not None:
        print("[benchmark][WARN] --limit-frames is ignored in benchmark mode (duration controls stop).")

    profile_out_dir: Optional[Path] = None
    if profile_enabled:
        profile_out_dir = _pick_profile_out_dir(
            profile_out_arg=args.profile_out,
            save_path=save_path,
            ckpt_path=ckpt_path,
            run_tag="motionbert",
        )

    if args.display:
        return stream_infer_and_display(
            ckpt_path=ckpt_path,
            video_path=video_path,
            cfg=cfg,
            yolo_path=yolo_path,
            device=device,
            clip_len=clip_len,
            win_step=win_step,
            frame_step=frame_step,
            clip_len_raw=clip_len_raw,
            win_step_raw=win_step_raw,
            save_path=save_path,
            profile_enabled=profile_enabled,
            profile_out_dir=profile_out_dir,
            profile_duration_s=profile_duration_s,
            hw_sample_hz=hw_sample_hz,
            benchmark_enabled=benchmark_enabled,
            no_display=no_display,
            args=args,
        )

    # ------------------------------------------------------------------
    # 1) YOLOv11 pose extraction
    # ------------------------------------------------------------------
    pose_model = YOLO(str(yolo_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path.as_posix()}")

    frames_xy: List[np.ndarray] = []
    frames_cf: List[np.ndarray] = []
    img_shape = None
    frame_idx = 0
    sampled_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if img_shape is None:
            h, w = frame.shape[:2]
            img_shape = (int(h), int(w))

        do_pose = (int(frame_idx) % int(frame_step)) == 0
        if do_pose:
            results = pose_model.predict(
                source=frame,
                imgsz=int(args.imgsz),
                conf=float(args.conf_thres),
                verbose=False,
                device=device,
            )

            kpts_xy = np.zeros((17, 2), dtype=np.float32)
            kpts_conf = np.zeros((17,), dtype=np.float32)

            if results and len(results) > 0 and results[0].keypoints is not None:
                kpts = results[0].keypoints
                xy_all = kpts.xy.cpu().numpy() if hasattr(kpts.xy, "cpu") else np.array(kpts.xy)
                cf_all = kpts.conf.cpu().numpy() if hasattr(kpts.conf, "cpu") else np.array(kpts.conf)

                if xy_all.ndim == 3 and xy_all.shape[0] > 0:
                    scores = cf_all.sum(axis=1) if (cf_all.ndim == 2 and cf_all.shape[0] == xy_all.shape[0]) else None
                    best = int(np.argmax(scores)) if scores is not None else 0
                    kpts_xy = xy_all[best].astype(np.float32)
                    if cf_all.ndim == 2 and cf_all.shape[0] == xy_all.shape[0]:
                        kpts_conf = cf_all[best].astype(np.float32)
                    else:
                        kpts_conf = np.ones((17,), dtype=np.float32)

            frames_xy.append(kpts_xy)
            frames_cf.append(kpts_conf)
            sampled_idx += 1

        frame_idx += 1
        if args.limit_frames is not None and frame_idx >= int(args.limit_frames):
            break

    cap.release()

    if img_shape is None:
        raise RuntimeError("No frames read from video.")
    if sampled_idx <= 0:
        raise RuntimeError("No sampled frames were generated. Reduce --k/--frame-step or remove --limit-frames.")

    kpts_xy = np.stack(frames_xy, axis=0)  # (T_sampled,17,2)
    kpts_conf = np.stack(frames_cf, axis=0)  # (T_sampled,17)

    # ------------------------------------------------------------------
    # 2) Build and save MotionBERT action pkl
    # ------------------------------------------------------------------
    split_list, annotations = build_windows(
        kpts_xy=kpts_xy,
        kpts_conf=kpts_conf,
        img_shape=img_shape,
        win_len=int(clip_len),
        win_step=int(win_step),
        pad_tail=bool(args.pad_tail),
        missing_conf_thres=float(args.missing_conf_thres),
        drop_empty_windows=not bool(args.keep_empty_windows),
        video_stem=video_path.stem,
    )

    dataset = {"split": {"xsub_train": [], "xsub_val": split_list}, "annotations": annotations}

    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    if not annotations:
        print("No windows were generated. Try --pad-tail or smaller --win-len.")
        return 1

    # ------------------------------------------------------------------
    # 3) Load MotionBERT ActionNet
    # ------------------------------------------------------------------
    model_backbone = load_backbone(cfg)
    model = ActionNet(
        backbone=model_backbone,
        dim_rep=getattr(cfg, "dim_rep", 512),
        num_classes=int(getattr(cfg, "action_classes", 11)),
        dropout_ratio=getattr(cfg, "dropout_ratio", 0.0),
        version=getattr(cfg, "model_version", "class"),
        hidden_dim=getattr(cfg, "hidden_dim", 2048),
        num_joints=getattr(cfg, "num_joints", 17),
    )

    use_dp = device.startswith("cuda") and torch.cuda.device_count() > 1
    if use_dp:
        model = nn.DataParallel(model)
    model = model.to(device)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = _clean_state_dict_for_model(state, model)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ------------------------------------------------------------------
    # 4) Run per-window inference
    # ------------------------------------------------------------------
    num_classes = int(getattr(cfg, "action_classes", 11))
    scale_range = getattr(cfg, "scale_range_test", None)

    labels_file_names = load_labels_file(args.labels_file)

    unmerged_len_expected = len(CLASS_NAMES_DEFAULT)
    merged_len_expected = len(CLASS_NAMES_MERGED_DEFAULT)

    # Only merge at inference time when the model is trained with separate fall subclasses (11-class).
    merge_fall = (not bool(args.no_merge_fall)) and (num_classes == unmerged_len_expected)

    if labels_file_names is not None:
        if merge_fall and len(labels_file_names) == merged_len_expected:
            # User provided already-merged display names (7).
            class_names_out = list(labels_file_names)
        else:
            # User provided names matching the model output space.
            base_names = pad_or_trim(list(labels_file_names), num_classes)
            if merge_fall:
                class_names_out = ["Fall"] + [base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
            else:
                class_names_out = base_names
    else:
        # No labels file: choose a sane default taxonomy for the configured model output space.
        if num_classes == merged_len_expected:
            class_names_out = list(CLASS_NAMES_MERGED_DEFAULT)
        else:
            base_names = pad_or_trim(list(CLASS_NAMES_DEFAULT), num_classes)
            if merge_fall:
                class_names_out = ["Fall"] + [base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
            else:
                class_names_out = base_names

    fall_idx = infer_fall_indices(class_names_out)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ann_map = {ann["frame_dir"]: ann for ann in annotations}

    def write_header(writer: csv.writer) -> None:
        writer.writerow(
            [
                "frame_dir",
                "start_frame",
                "end_frame",
                "pred_id",
                "pred_name",
                "pred_conf",
                "p_fall",
            ]
        )

    def write_pred_row(writer: csv.writer, pred: dict) -> None:
        writer.writerow(
            [
                pred["frame_dir"],
                pred["start_frame"],
                pred["end_frame"],
                pred["pred_id"],
                pred["pred_name"],
                f"{float(pred['pred_conf']):.6f}",
                f"{float(pred['p_fall']):.6f}",
            ]
        )

    # ------------------------------------------------------------------
    # 4) Predict windows (offline)
    # ------------------------------------------------------------------
    predict_kwargs = dict(
        model=model,
        device=device,
        clip_len=clip_len,
        scale_range=scale_range,
        merge_fall=merge_fall,
        fall_idx=fall_idx,
        class_names_out=class_names_out,
        unmerged_len_expected=unmerged_len_expected,
        merged_len_expected=merged_len_expected,
        frame_step=frame_step,
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        write_header(writer)
        for frame_dir in split_list:
            ann = ann_map[frame_dir]
            pred = predict_one_window(ann=ann, frame_dir=frame_dir, **predict_kwargs)
            write_pred_row(writer, pred)

    print(f"Saved pkl: {out_pkl.as_posix()}")
    print(f"Saved predictions: {out_csv.as_posix()}")
    print(f"Windows: {len(split_list)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
