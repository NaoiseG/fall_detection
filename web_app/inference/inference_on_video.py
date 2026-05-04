#!/usr/bin/env python3
"""
MP4 -> YOLO pose (configurable keypoint model) -> temporal model inference -> popup display.

Preprocessing mirrors training:
  - dataset_helpers/dataset.py: fill/interp missing joints, optional normalize/vel/acc/global + mask channel
  - training/train_models.py: temporal model architectures + checkpoint metadata

Usage:
  python -m inference.inference_on_video --video path\\to\\clip.mp4 --model models\\tcn\\<run>\\tcn_best.pt
  python -m inference.inference_on_video --video path\\to\\clip.mp4 --model models\\tcn --save out.mp4
  
  python inference/inference_on_video.py \
  --video /path/to/myclip.mp4 \
  --model /path/to/mymodel.pt \
  --arch tcn \
  --save runs/annotated/out.mp4 \
  --profile 1 \
  --profile-out runs/nano_test \
  --profile-duration-s 60 \
  --hw-sample-hz 1.0 \
  --no-display 1
"""

from __future__ import annotations

import argparse
import base64
import csv
import inspect
import json
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn

# Allow running as a script from any working directory (mirrors `python -m ...`).
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import inference.helpers.dataset as ds
from inference.helpers.keypoint_runtime import KeypointRuntime

from models.classification.tcn.simple_tcn import TCNBaseline
from models.classification.stgcn.paper_stgcn import PaperSTGCNClassifier
from models.classification.cnnlstm.cnn_lstm import CNNLSTMTwoHead

try:
    from models.classification.gru.simple_gru import GRUBaseline
except ModuleNotFoundError:
    GRUBaseline = None  # type: ignore[assignment,misc]

K = 17  # COCO-17 joints for Ultralytics pose models

SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12),
]

KNOWN_ARCHES = ["tcn", "lstm", "gru", "gcn", "mlp", "stgcn", "cnnlstm", "rf"]

# Updated fall-merged label names (7 classes)
# 0: Fall (all fall subclasses merged)
# 1..6: ADLs
FALL_MERGED_CLASS_NAMES = [
    "Fall",
    "Walking",
    "Standing",
    "Sitting",
    "Picking up an object",
    "Jumping",
    "Laying",
]


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


def _maybe_cuda_sync(sync_cuda: bool) -> None:
    if sync_cuda:
        torch.cuda.synchronize()


def _slugify_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name).strip())
    s = re.sub(r"-{2,}", "-", s).strip("-._")
    return s or "model"


def _pick_profile_out_dir(
    profile_out_arg: Optional[str],
    save_path: Optional[Path],
    ckpt_path: Path,
    arch: str,
) -> Path:
    base_root = Path(profile_out_arg).expanduser() if profile_out_arg else (save_path.parent if save_path is not None else Path("runs") / "profiling")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_tag = _slugify_name(Path(ckpt_path).stem)
    arch_tag = _slugify_name(str(arch).lower())
    run_name = f"{stamp}_{arch_tag}_{model_tag}"
    out_dir = base_root / run_name

    # Very unlikely with microseconds, but keep guaranteed uniqueness.
    suffix = 1
    while out_dir.exists():
        out_dir = base_root / f"{run_name}_{suffix:02d}"
        suffix += 1
    return out_dir


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
        r"\bVDD_IN\s+([+-]?\d+(?:\.\d+)?)(m?W|W)?(?:/([+-]?\d+(?:\.\d+)?)(m?W|W)?)?",
        r"\bPOM_5V_SYS\s+([+-]?\d+(?:\.\d+)?)(m?W)?(?:/([+-]?\d+(?:\.\d+)?)(m?W)?)?",
    ]
    for pat in power_patterns:
        m_pow = re.search(pat, line, flags=re.IGNORECASE)
        if not m_pow:
            continue
        raw = _safe_float(m_pow.group(1))
        unit = (m_pow.group(2) or "").lower()
        if not np.isfinite(raw):
            continue
        if unit == "w":
            sample["power_w"] = raw
        elif unit == "mw" or raw > 100.0:
            sample["power_w"] = raw / 1000.0
        else:
            sample["power_w"] = raw
        break

    return sample


def _extract_numeric_from_obj(obj: Any) -> float:
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, str):
        return _parse_first_number(obj)
    if isinstance(obj, dict):
        # Prefer common scalar keys if present.
        for key in ("value", "val", "avg", "cur", "current", "usage", "percent", "perc"):
            if key in obj and _is_finite(obj[key]):
                return float(obj[key])
        vals = [_extract_numeric_from_obj(v) for v in obj.values()]
        return _avg_valid(vals)
    if isinstance(obj, (list, tuple)):
        vals = [_extract_numeric_from_obj(v) for v in obj]
        return _avg_valid(vals)
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
                src = inspect.getsource(jtop_hardware.get_platform_variables)
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

        # Final fallback: psutil (RAM+CPU, best effort temperatures).
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
                        if not np.isfinite(cur):
                            continue
                        if "cpu" in name_l or "core" in name_l or "soc" in name_l:
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
        stats: Dict[str, Any],
        gpu_obj: Any = None,
        temp_obj: Any = None,
        power_obj: Any = None,
        memory_obj: Any = None,
        cpu_obj: Any = None,
    ) -> Dict[str, float]:
        out = {
            "ram_used_pct": float("nan"),
            "cpu_pct": float("nan"),
            "gpu_pct": float("nan"),
            "cpu_temp_c": float("nan"),
            "gpu_temp_c": float("nan"),
            "power_w": float("nan"),
        }
        if not stats and gpu_obj is None and temp_obj is None and power_obj is None and memory_obj is None and cpu_obj is None:
            return out

        if stats:
            # RAM (%)
            ram_v = stats.get("RAM", None)
            if isinstance(ram_v, dict):
                used = _extract_numeric_from_obj(ram_v.get("used", ram_v.get("use", None)))
                total = _extract_numeric_from_obj(ram_v.get("tot", ram_v.get("total", None)))
                if np.isfinite(used) and np.isfinite(total) and total > 0:
                    out["ram_used_pct"] = 100.0 * used / total
                else:
                    out["ram_used_pct"] = _extract_numeric_from_obj(ram_v)
            elif isinstance(ram_v, (list, tuple)) and len(ram_v) >= 2:
                used = _extract_numeric_from_obj(ram_v[0])
                total = _extract_numeric_from_obj(ram_v[1])
                if np.isfinite(used) and np.isfinite(total) and total > 0:
                    out["ram_used_pct"] = 100.0 * used / total
            else:
                out["ram_used_pct"] = _extract_numeric_from_obj(ram_v)
            if np.isfinite(out["ram_used_pct"]) and out["ram_used_pct"] <= 1.0:
                out["ram_used_pct"] *= 100.0

            # CPU (%): prefer explicit "CPU", then per-core CPUx keys.
            out["cpu_pct"] = _extract_numeric_from_obj(stats.get("CPU", None))
            if not np.isfinite(out["cpu_pct"]):
                cpu_keys = [k for k in stats.keys() if re.fullmatch(r"cpu\d+", str(k).lower())]
                cpu_vals = [_extract_numeric_from_obj(stats[k]) for k in cpu_keys]
                out["cpu_pct"] = _avg_valid(cpu_vals)

            # GPU (%): only accept percentage-like values (0..100) to avoid MHz clocks.
            gpu_candidates: List[float] = []
            for k, v in stats.items():
                k_l = str(k).lower()
                if "gpu" in k_l or "gr3d" in k_l:
                    keyed = _collect_keyed_numeric(v, prefix=k_l)
                    pct_from_keys = _pick_pct_from_keyed(
                        keyed,
                        tokens_any=("load", "usage", "util", "percent", "perc", "gr3d", "gpu"),
                    )
                    if np.isfinite(pct_from_keys):
                        gpu_candidates.append(float(pct_from_keys))
                    scalar = _extract_numeric_from_obj(v)
                    if np.isfinite(scalar) and 0.0 <= float(scalar) <= 100.0:
                        gpu_candidates.append(float(scalar))
            out["gpu_pct"] = _avg_valid(gpu_candidates)

            # Temperatures.
            cpu_t = []
            gpu_t = []
            for k, v in stats.items():
                k_l = str(k).lower().replace(" ", "_")
                if "temp" in k_l and "cpu" in k_l:
                    cpu_t.append(_extract_numeric_from_obj(v))
                if "temp" in k_l and "gpu" in k_l:
                    gpu_t.append(_extract_numeric_from_obj(v))
            out["cpu_temp_c"] = _avg_valid(cpu_t)
            out["gpu_temp_c"] = _avg_valid(gpu_t)

            # Power: prefer total/input rails if available.
            power_candidates: List[float] = []
            power_pref: List[float] = []
            for k, v in stats.items():
                k_l = str(k).lower().replace(" ", "_")
                if "power" in k_l or "pom_" in k_l or "vdd_in" in k_l:
                    val = _extract_numeric_from_obj(v)
                    if np.isfinite(val):
                        power_candidates.append(val)
                        if "5v_in" in k_l or "vdd_in" in k_l or "tot" in k_l:
                            power_pref.append(val)
            raw_power = _avg_valid(power_pref) if power_pref else _avg_valid(power_candidates)
            if np.isfinite(raw_power):
                out["power_w"] = raw_power / 1000.0 if raw_power > 100.0 else raw_power

        # Fallback/augmentation from direct jtop objects.
        if memory_obj is not None and not np.isfinite(out["ram_used_pct"]):
            mem_keyed = _collect_keyed_numeric(memory_obj, prefix="memory")
            used_vals = [v for k, v in mem_keyed if np.isfinite(v) and any(tok in k for tok in ("used", "use", "util"))]
            total_vals = [v for k, v in mem_keyed if np.isfinite(v) and any(tok in k for tok in ("total", "tot", "size"))]
            used = _avg_valid(used_vals)
            total = _avg_valid(total_vals)
            if np.isfinite(used) and np.isfinite(total) and total > 0:
                out["ram_used_pct"] = 100.0 * used / total
            else:
                out["ram_used_pct"] = _pick_pct_from_keyed(mem_keyed, tokens_any=("percent", "perc", "usage", "util"))

        if cpu_obj is not None and not np.isfinite(out["cpu_pct"]):
            cpu_keyed = _collect_keyed_numeric(cpu_obj, prefix="cpu")
            out["cpu_pct"] = _pick_pct_from_keyed(cpu_keyed, tokens_any=("load", "usage", "util", "percent", "perc"))
            if not np.isfinite(out["cpu_pct"]):
                # Last fallback: average all plausible percentages.
                all_pct = [v for _, v in cpu_keyed if np.isfinite(v) and 0.0 <= float(v) <= 100.0]
                out["cpu_pct"] = _avg_valid(all_pct)

        if gpu_obj is not None and not np.isfinite(out["gpu_pct"]):
            gpu_keyed = _collect_keyed_numeric(gpu_obj, prefix="gpu")
            out["gpu_pct"] = _pick_pct_from_keyed(gpu_keyed, tokens_any=("load", "usage", "util", "gr3d", "gpu", "percent", "perc"))
            if not np.isfinite(out["gpu_pct"]):
                all_pct = [v for _, v in gpu_keyed if np.isfinite(v) and 0.0 <= float(v) <= 100.0]
                out["gpu_pct"] = _avg_valid(all_pct)

        if temp_obj is not None and (not np.isfinite(out["cpu_temp_c"]) or not np.isfinite(out["gpu_temp_c"])):
            temp_keyed = _collect_keyed_numeric(temp_obj, prefix="temp")
            if not np.isfinite(out["cpu_temp_c"]):
                cpu_vals = [v for k, v in temp_keyed if np.isfinite(v) and ("cpu" in k or "soc" in k) and -20.0 <= float(v) <= 150.0]
                out["cpu_temp_c"] = _avg_valid(cpu_vals)
            if not np.isfinite(out["gpu_temp_c"]):
                gpu_vals = [v for k, v in temp_keyed if np.isfinite(v) and "gpu" in k and -20.0 <= float(v) <= 150.0]
                out["gpu_temp_c"] = _avg_valid(gpu_vals)

        if power_obj is not None and not np.isfinite(out["power_w"]):
            power_keyed = _collect_keyed_numeric(power_obj, prefix="power")
            pref = [v for k, v in power_keyed if np.isfinite(v) and any(tok in k for tok in ("5v_in", "vdd_in", "in", "tot", "total"))]
            vals = pref if pref else [v for _, v in power_keyed if np.isfinite(v)]
            raw_power = _avg_valid(vals)
            if np.isfinite(raw_power):
                out["power_w"] = raw_power / 1000.0 if raw_power > 100.0 else raw_power

        # Last-resort GPU parse from full keyed stats.
        if not np.isfinite(out["gpu_pct"]) and stats:
            keyed = _collect_keyed_numeric(stats, prefix="stats")
            out["gpu_pct"] = _pick_pct_from_keyed(keyed, tokens_any=("gr3d", "gpu"))

        # Guardrail: utilisation cannot be outside 0..100%.
        if np.isfinite(out["gpu_pct"]) and not (0.0 <= float(out["gpu_pct"]) <= 100.0):
            out["gpu_pct"] = float("nan")

        return out


def _save_time_plots(
    profile_out: Path,
    time_rows: List[Dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_idx = np.asarray([_safe_float(r.get("frame_idx", np.nan)) for r in time_rows], dtype=np.float64)
    fps = np.asarray([_safe_float(r.get("fps", np.nan)) for r in time_rows], dtype=np.float64)
    preprocess = np.asarray([_safe_float(r.get("preprocess_ms", np.nan)) for r in time_rows], dtype=np.float64)
    infer = np.asarray([_safe_float(r.get("inference_ms", np.nan)) for r in time_rows], dtype=np.float64)
    post = np.asarray([_safe_float(r.get("postprocess_ms", np.nan)) for r in time_rows], dtype=np.float64)
    vis = np.asarray([_safe_float(r.get("visualisation_ms", np.nan)) for r in time_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax0, ax1 = axes

    if frame_idx.size > 0:
        ax0.plot(frame_idx, fps, color="tab:blue", linewidth=1.3, label="Total FPS")
        ax0.set_xlabel("Frame Number")
        ax0.set_ylabel("FPS")
        ax0.legend(loc="best")
        fps_top = max(1.0, float(np.nanpercentile(fps, 99)) * 1.2) if np.isfinite(fps).any() else 1.0
        ax0.set_ylim(0.0, fps_top)
    else:
        ax0.text(0.5, 0.5, "No timing data", ha="center", va="center", transform=ax0.transAxes)
        ax0.set_xlabel("Frame Number")
        ax0.set_ylabel("FPS")

    if frame_idx.size > 0:
        ax1.plot(frame_idx, preprocess, label="Pre-processing", linewidth=1.2)
        ax1.plot(frame_idx, infer, label="Inference", linewidth=1.2)
        ax1.plot(frame_idx, post, label="Statistic calculation / post-processing", linewidth=1.2)
        ax1.plot(frame_idx, vis, label="Visualisation", linewidth=1.2)
        ax1.set_xlabel("Frame Number")
        ax1.set_ylabel("Latency (ms)")
        lat_all = np.concatenate([preprocess, infer, post, vis], axis=0)
        lat_top = max(1.0, float(np.nanpercentile(lat_all, 99)) * 1.2) if np.isfinite(lat_all).any() else 1.0
        ax1.set_ylim(0.0, lat_top)
        ax1.legend(loc="best", fontsize=9)
    else:
        ax1.text(0.5, 0.5, "No timing data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xlabel("Frame Number")
        ax1.set_ylabel("Latency (ms)")

    fig.tight_layout()
    fig.savefig(profile_out / "fig_time_efficiency.png", dpi=180)
    plt.close(fig)


def _save_hw_plots(
    profile_out: Path,
    hw_rows: List[Dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_s = np.asarray([_safe_float(r.get("t_s", np.nan)) for r in hw_rows], dtype=np.float64)
    ram = np.asarray([_safe_float(r.get("ram_used_pct", np.nan)) for r in hw_rows], dtype=np.float64)
    cpu = np.asarray([_safe_float(r.get("cpu_pct", np.nan)) for r in hw_rows], dtype=np.float64)
    gpu = np.asarray([_safe_float(r.get("gpu_pct", np.nan)) for r in hw_rows], dtype=np.float64)
    cpu_t = np.asarray([_safe_float(r.get("cpu_temp_c", np.nan)) for r in hw_rows], dtype=np.float64)
    gpu_t = np.asarray([_safe_float(r.get("gpu_temp_c", np.nan)) for r in hw_rows], dtype=np.float64)
    pwr = np.asarray([_safe_float(r.get("power_w", np.nan)) for r in hw_rows], dtype=np.float64)

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    ax0, ax1 = axes1
    if t_s.size > 0:
        ax0.plot(t_s, ram, label="RAM usage", color="tab:green", linewidth=1.2)
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("RAM usage (%)")
        ax0.set_ylim(0.0, 100.0)
        ax0.legend(loc="best")

        ax1.plot(t_s, cpu, label="CPU", linewidth=1.2)
        ax1.plot(t_s, gpu, label="GPU", linewidth=1.2)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Utilisation (%)")
        ax1.set_ylim(0.0, 100.0)
        ax1.legend(loc="best")
    else:
        ax0.text(0.5, 0.5, "No hardware data", ha="center", va="center", transform=ax0.transAxes)
        ax1.text(0.5, 0.5, "No hardware data", ha="center", va="center", transform=ax1.transAxes)
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("RAM usage (%)")
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
    power_vals = [_safe_float(r.get("power_w", np.nan)) for r in hw_rows]

    duration_s = float("nan")
    if t_vals:
        finite_t = [x for x in t_vals if _is_finite(x)]
        if finite_t:
            duration_s = float(max(finite_t))

    summary = {
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
        "max_ram_pct": _json_safe_number(_max_valid(ram_vals)),
        "max_cpu_pct": _json_safe_number(_max_valid(cpu_vals)),
        "max_gpu_pct": _json_safe_number(_max_valid(gpu_vals)),
        "avg_power_w": _json_safe_number(_avg_valid(power_vals)),
        "total_frames_processed": int(len(time_rows)),
        "duration_s": _json_safe_number(duration_s),
    }

    with (profile_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def pick_device(device: Optional[str]) -> str:
    if not device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    d = device.lower().strip()
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def infer_arch_from_path(p: Path) -> Optional[str]:
    tokens = [p.name.lower(), p.stem.lower()] + [x.lower() for x in p.parts]
    for arch in sorted(KNOWN_ARCHES, key=len, reverse=True):
        if any(tok == arch for tok in tokens):
            return arch
        if any(tok.startswith(arch + "_") for tok in tokens):
            return arch
        if arch == "rf":
            # Avoid substring matches like "pe[r f]ormance" -> "rf".
            if any(tok.startswith("rf") for tok in tokens):
                return arch
            continue
        if any(arch in tok for tok in tokens):
            return arch
    return None


def resolve_ckpt_and_arch(model_arg: str, arch_arg: Optional[str]) -> Tuple[Path, str]:
    """
    --model can be:
      - a checkpoint file (*.pt or *.pkl)
      - a model folder containing checkpoints (picks newest *best*.pt / *best*.pkl)
      - a model python file under models/<arch>/...py (picks newest *best*.pt / *best*.pkl under that folder)
    """
    p = Path(model_arg).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"--model not found: {p}")

    arch = (arch_arg or "").lower().strip() or infer_arch_from_path(p)

    if p.is_file():
        suf = p.suffix.lower()
        if suf in {".pt", ".pth", ".bin"}:
            if not arch:
                arch = infer_arch_from_path(p)
            if not arch:
                raise ValueError("Could not infer --arch from checkpoint path. Pass --arch explicitly.")
            if arch == "rf":
                raise ValueError(
                    "Inferred/selected --arch rf for a .pt checkpoint. RF checkpoints are expected to be .pkl/.pickle.\n"
                    "Pass the correct --arch (tcn/lstm/...) or provide an RF .pkl checkpoint."
                )
            return p, arch

        if suf in {".pkl", ".pickle"}:
            if not arch:
                arch = infer_arch_from_path(p) or "rf"
            return p, arch

        if suf == ".py":
            if not arch:
                raise ValueError("Could not infer --arch from model .py path. Pass --arch explicitly.")
            model_dir = p.parent
            if arch == "rf":
                ckpts = sorted(model_dir.glob("**/*best*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
                if not ckpts:
                    ckpts = sorted(model_dir.glob("**/*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
            else:
                ckpts = sorted(model_dir.glob("**/*best*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
                if not ckpts:
                    ckpts = sorted(model_dir.glob("**/*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found under: {model_dir}")
            return ckpts[0], arch

        raise ValueError(f"Unsupported --model file type: {p.suffix}")

    if arch == "rf":
        ckpts = sorted(p.glob("**/*best*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not ckpts:
            ckpts = sorted(p.glob("**/*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    else:
        ckpts = sorted(p.glob("**/*best*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not ckpts:
            # Allow pointing to a folder containing an RF run folder without passing --arch rf.
            ckpts = sorted(p.glob("**/*best*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not ckpts:
            ckpts = sorted(p.glob("**/*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not ckpts:
            ckpts = sorted(p.glob("**/*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint *.pt/*.pkl files found under: {p}")

    ckpt = ckpts[0]
    if not arch:
        arch = infer_arch_from_path(ckpt)
    if not arch:
        if ckpt.suffix.lower() in {".pkl", ".pickle"}:
            arch = "rf"
        else:
            raise ValueError("Could not infer --arch from checkpoint path. Pass --arch explicitly.")
    return ckpt, arch


def load_checkpoint(ckpt_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    # PyTorch >=2.6 defaults `weights_only=True`, which can fail on our training checkpoints
    # because they include non-tensor metadata (e.g., NumPy scalars). We trained these
    # checkpoints ourselves, so we opt into the legacy behavior for compatibility.
    try:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"], ckpt_obj
    if isinstance(ckpt_obj, dict):
        return ckpt_obj, {}
    raise TypeError("Unsupported checkpoint format (expected dict or dict with 'state_dict').")


def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state.keys())
    if any(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_rf_checkpoint(ckpt_path: Path) -> Dict[str, object]:
    """
    Loads a Random Forest checkpoint saved by `models/rf/train_rf.py`.
    """
    try:
        with Path(ckpt_path).open("rb") as f:
            obj = pickle.load(f)
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Failed to load RF checkpoint. You likely need scikit-learn installed.\n"
            "Install it with: pip install scikit-learn\n"
            f"Import error: {e}"
        )

    if not isinstance(obj, dict) or "model" not in obj:
        raise TypeError(f"Unsupported RF checkpoint format: expected a dict with key 'model'. Got: {type(obj)}")
    return obj


def load_class_names(num_classes: int, meta: Dict[str, object], labels_file: Optional[str]) -> List[str]:
    names: List[str] = []
    if labels_file:
        p = Path(labels_file).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"--labels-file not found: {p}")
        names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # Default to updated fall-merged names when displaying 7-class outputs.
    if not names and int(num_classes) == 7:
        return list(FALL_MERGED_CLASS_NAMES)

    if not names:
        for key in ("new_label_names", "class_names", "classes", "labels"):
            v = meta.get(key, None)
            if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                names = list(v)
                break

    if not names:
        names = [f"class_{i}" for i in range(int(num_classes))]

    if len(names) != int(num_classes):
        names = names[: int(num_classes)] + [f"class_{i}" for i in range(len(names), int(num_classes))]
    return names


def build_temporal_model(
    arch: str,
    in_features: int,
    num_classes: int,
    device: str,
    T_used: int,
    node_features: Optional[int],
) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "tcn":
        model = TCNBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            kernel_size=3,
            dropout=0.1,
        )
    elif arch == "gru":
        if GRUBaseline is None:
            raise RuntimeError("GRUBaseline import failed (models/classification/gru not found).")
        model = GRUBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
            pool="last",
        )
    elif arch == "stgcn":
        if node_features is None:
            raise ValueError("STGCN requires node_features (in_features must be divisible by 17).")
        model = PaperSTGCNClassifier(
            num_nodes=17,
            node_features=int(node_features),
            num_classes=num_classes,
            channels=(64, 64, 64, 64, 128, 128, 128, 256, 256, 256),
            t_kernel=9,
            dropout=0.2,
        )
    elif arch == "cnnlstm":
        if CNNLSTMTwoHead is None:
            raise RuntimeError("CNNLSTMTwoHead import failed (models/cnnlstm).")
        model = CNNLSTMTwoHead(
            in_features=in_features,
            num_classes=num_classes,
            embed_dim=128,
            hidden_size=128,
            lstm_layers=1,
            dropout=0.2,
            num_keypoints=17 if node_features is not None else None,
            kp_channels=node_features,
            pool="last",
        )
    else:
        raise ValueError(f"Unknown --arch: {arch} (expected one of {KNOWN_ARCHES})")
    model = model.to(device)
    print("Model device:", next(model.parameters()).device)
    return model


def draw_hud(frame: np.ndarray, lines: List[str], org: Tuple[int, int] = (10, 10)) -> np.ndarray:
    if not lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    pad = 8
    line_gap = 6
    bg_alpha = 0.6

    x0, y0 = org
    sizes = [cv2.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
    max_w = max(w for (w, h) in sizes)
    total_h = sum(h for (w, h) in sizes) + line_gap * (len(lines) - 1)

    box_w = max_w + 2 * pad
    box_h = total_h + 2 * pad
    h_img, w_img = frame.shape[:2]
    x1 = min(w_img - 1, x0 + box_w)
    y1 = min(h_img - 1, y0 + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0)

    y = y0 + pad
    for (w, h), s in zip(sizes, lines):
        y += h
        cv2.putText(frame, s, (x0 + pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_gap
    return frame


def draw_pose(frame: np.ndarray, xy: np.ndarray, conf: np.ndarray, conf_thres: float) -> np.ndarray:
    for i in range(K):
        if float(conf[i]) > float(conf_thres):
            x, y = int(xy[i, 0]), int(xy[i, 1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    for a, b in SKELETON:
        if float(conf[a]) > float(conf_thres) and float(conf[b]) > float(conf_thres):
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


def feature_layout(use_conf: bool, add_vel: bool, add_acc: bool) -> Dict[str, Optional[object]]:
    idx = 2  # xy
    conf_idx = None
    if use_conf:
        conf_idx = idx
        idx += 1
    vel_slice = None
    if add_vel:
        vel_slice = slice(idx, idx + 2)
        idx += 2
    acc_slice = None
    if add_acc:
        acc_slice = slice(idx, idx + 2)
        idx += 2
    return {"conf_idx": conf_idx, "vel_slice": vel_slice, "acc_slice": acc_slice}


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
            high_conf = np.where(np.isfinite(box_conf[:num_people]) & (box_conf[:num_people] >= float(conf_min)))[0]
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
    if not np.isfinite(best_dist) or best_dist > float(max_jump_px):
        return None, prev_center
    return best_idx, box_centers[best_idx].astype(np.float32, copy=True)


def expected_in_features(
    use_conf: bool,
    add_vel: bool,
    add_acc: bool,
    add_global: bool,
    add_mask: bool,
) -> int:
    if add_acc and not add_vel:
        raise ValueError("add_acc=True requires add_vel=True (acc is computed from vel).")
    c = 2
    if use_conf:
        c += 1
    if add_vel:
        c += 2
    if add_acc:
        c += 2
    if add_global:
        c += 4
    if add_mask:
        c += 1
    return int(K * c)


def pose_on_frame(
    keypoint_runtime: KeypointRuntime,
    frame_bgr: np.ndarray,
    imgsz: int,
    yolo_conf: float,
    max_people: int,
    use_half: bool,
    prev_center: Optional[np.ndarray],
    target_center: np.ndarray,
    conf_min: float,
    max_jump_px: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
    xy_zeros = np.zeros((K, 2), dtype=np.float32)
    cf_zeros = np.zeros((K,), dtype=np.float32)

    detections = keypoint_runtime.predict(
        frame_bgr=frame_bgr,
        imgsz=int(imgsz),
        conf=float(yolo_conf),
        max_people=max(1, int(max_people)),
        use_half=bool(use_half),
    )
    if detections.xy.ndim != 3 or detections.xy.shape[0] == 0:
        return xy_zeros, cf_zeros, prev_center, False

    xy_all = detections.xy
    cf_all = detections.conf
    box_centers = detections.box_centers
    box_conf = detections.box_conf

    if xy_all.shape[1] != K:
        raise ValueError(f"Expected {K} keypoints, got {xy_all.shape[1]}")

    num_candidates = int(xy_all.shape[0])
    if box_centers.shape[0] > 0:
        num_candidates = min(num_candidates, int(box_centers.shape[0]))
    else:
        box_centers = np.mean(xy_all, axis=1).astype(np.float32, copy=False)

    if cf_all is not None and cf_all.ndim == 2:
        num_candidates = min(num_candidates, int(cf_all.shape[0]))
    else:
        cf_all = None

    if box_conf is not None:
        num_candidates = min(num_candidates, int(box_conf.shape[0]))

    if num_candidates <= 0:
        return xy_zeros, cf_zeros, prev_center, False

    xy_all = xy_all[:num_candidates]
    box_centers = box_centers[:num_candidates]
    if cf_all is not None:
        cf_all = cf_all[:num_candidates]
    if box_conf is not None:
        box_conf = box_conf[:num_candidates]

    idx, new_center = select_person_idx(
        box_centers=box_centers,
        box_conf=box_conf,
        prev_center=prev_center,
        target_center=target_center,
        conf_min=float(conf_min),
        max_jump_px=float(max_jump_px),
    )
    if idx is None:
        return xy_zeros, cf_zeros, prev_center, False

    xy = xy_all[idx].astype(np.float32, copy=False)
    if cf_all is not None and idx < cf_all.shape[0]:
        cf = cf_all[idx].astype(np.float32, copy=False)
        if cf.shape[0] != K:
            cf = np.ones((K,), dtype=np.float32)
    else:
        # No confidences available: treat as all-ones (model will likely ignore if use_conf=False)
        cf = np.ones((K,), dtype=np.float32)

    return xy, cf, new_center, True


def make_window_features(
    xy_seq: np.ndarray,      # (L,K,2) in pixel coords
    conf_seq: np.ndarray,    # (L,K)
    T: int,
    use_conf: bool,
    normalize: bool,
    normalize_mode: str,
    add_vel: bool,
    add_acc: bool,
    add_global: bool,
    add_mask: bool,
    conf_thres: float,
    max_interp_gap: int,
    missing_mode: str,
    interp_mode: str,
    interp_group: int,
    rp_center_mode: str,
    rp_img_w: Optional[int],
    rp_img_h: Optional[int],
    min_valid_frac: float,
) -> np.ndarray:
    """
    Build a single window feature tensor (T,F) matching dataset_helpers/dataset.py + training/train_models.py.
    """
    T = int(T)
    L = int(xy_seq.shape[0])
    if L <= 0:
        feat_dim = expected_in_features(use_conf, add_vel, add_acc, add_global, add_mask)
        return np.zeros((T, feat_dim), dtype=np.float32)

    missing_mode = str(missing_mode).lower().strip()
    interp_mode = str(interp_mode).lower().strip()
    if missing_mode == "conf_thres" and interp_mode == "short_gap_hold":
        xy_filled, conf_filled = ds._fill_and_mask_kpts(
            xy_seq.astype(np.float32, copy=False),
            conf_seq.astype(np.float32, copy=False),
            conf_thres=float(conf_thres),
            max_interp_gap=int(max_interp_gap),
        )
    else:
        xy_filled, conf_filled = ds._fill_and_mask_kpts_paper(
            xy_seq.astype(np.float32, copy=False),
            conf_seq.astype(np.float32, copy=False),
            conf_thres=float(conf_thres),
            missing_mode=str(missing_mode),
            interp_mode=str(interp_mode),
            max_interp_gap=int(max_interp_gap),
            interp_group=int(interp_group),
        )

    if not bool(normalize):
        xy_used = xy_filled.astype(np.float32, copy=False)
    else:
        nm = str(normalize_mode).lower().strip()
        if nm == "center_scale":
            xy_used = ds._normalize_xy(xy_filled, conf_filled)
        elif nm == "paper_rp":
            center = ds._compute_image_center(
                xy=xy_filled,
                rp_center_mode=str(rp_center_mode),
                rp_img_w=rp_img_w,
                rp_img_h=rp_img_h,
            )
            xy_used = ds._normalize_xy_paper_rp(xy_filled, conf_filled, center=center)
        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

    parts = [xy_used]
    if use_conf:
        parts.append(conf_filled[..., None])

    vel = None
    if add_vel:
        vel = ds._add_velocity_channels(xy_used)
        parts.append(vel)
    if add_acc:
        if vel is None:
            vel = ds._add_velocity_channels(xy_used)
        parts.append(ds._add_acceleration_channels(vel))
    if add_global:
        g = ds._global_features(xy_used, conf_filled)  # (L,4)
        parts.append(np.repeat(g[:, None, :], repeats=K, axis=1))

    Xf = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)  # (L,K,C)

    frac_valid = (conf_filled > float(conf_thres)).mean(axis=1)
    valid = frac_valid >= float(min_valid_frac)  # (L,)

    layout = feature_layout(use_conf=use_conf, add_vel=add_vel, add_acc=add_acc)
    seq = Xf

    if L < T:
        pad = np.repeat(seq[-1:, :, :], repeats=(T - L), axis=0) if L > 0 else np.zeros((T, K, Xf.shape[2]), np.float32)
        if layout["conf_idx"] is not None:
            pad[:, :, int(layout["conf_idx"])] = 0.0
        if layout["vel_slice"] is not None:
            pad[:, :, layout["vel_slice"]] = 0.0
        if layout["acc_slice"] is not None:
            pad[:, :, layout["acc_slice"]] = 0.0

        seq = np.concatenate([seq, pad], axis=0)
        valid = np.concatenate([valid, np.zeros((T - L,), dtype=bool)], axis=0)

    seq = seq.copy()
    seq[~valid] = 0.0
    if add_mask:
        m = np.repeat(valid.astype(np.float32)[:, None, None], repeats=K, axis=1)
        seq = np.concatenate([seq, m], axis=-1)

    # Flatten to (T, F)
    return seq.reshape(T, int(seq.shape[1]) * int(seq.shape[2])).astype(np.float32, copy=False)


@torch.no_grad()
def infer_one_window(
    model: nn.Module,
    window_feat: np.ndarray,  # (T,F)
    device: str,
    use_half: bool,
    merge_fall_11_to_7: bool,
) -> Tuple[int, float, Optional[float]]:
    model.eval()

    xb = torch.from_numpy(window_feat[None, ...]).to(device)
    xb = xb.half() if use_half else xb.float()

    out = model(xb)
    fall_logit = None
    if isinstance(out, (tuple, list)) and len(out) == 2:
        logits, fall_logit = out[0], out[1]
    else:
        logits = out

    if logits.ndim == 3:
        logits = logits[:, -1, :]

    prob = torch.softmax(logits, dim=-1)
    if merge_fall_11_to_7:
        if int(prob.shape[-1]) != 11:
            raise ValueError(f"merge_fall_11_to_7=True expects 11 classes, got {int(prob.shape[-1])}")
        prob = torch.cat([prob[:, :5].sum(dim=1, keepdim=True), prob[:, 5:]], dim=1)  # (1,7)

    pconf, pred = torch.max(prob, dim=-1)

    p_fall = None
    if fall_logit is not None:
        p_fall = float(torch.sigmoid(fall_logit.view(-1))[0].item())

    return int(pred.item()), float(pconf.item()), p_fall


def run_inference_stream_packets(
    *,
    video_path: Path,
    classification_model_path: Path,
    keypoint_model_path: Path,
    on_packet: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
    arch: Optional[str] = None,
    labels_file: Optional[str] = None,
    save_path: Optional[Path] = None,
    no_display: bool = True,
    realtime: bool = True,
    display_fps: float = 0.0,
    device: Optional[str] = None,
    keypoint_backend: Optional[str] = None,
    half: int = 0,
    imgsz: int = 640,
    yolo_conf: float = 0.25,
    max_people: int = 10,
    track_conf_min: float = 0.75,
    track_max_jump_px: float = 0.0,
    track_max_jump_diag_frac: float = 0.25,
    track_max_lost: int = 10,
    track_target_x_frac: float = 0.5,
    track_target_y_frac: float = 0.5,
    T: int = 0,
    stride: int = 0,
    frame_step: int = 1,
    normalize_mode: Optional[str] = None,
    missing_mode: Optional[str] = None,
    interp_mode: Optional[str] = None,
    interp_group: int = 0,
    rp_center_mode: Optional[str] = None,
    rp_img_w: int = 0,
    rp_img_h: int = 0,
    jpeg_quality: int = 80,
    **_unused_options: Any,
) -> int:
    frame_step = int(frame_step)
    if frame_step <= 0:
        raise ValueError("--frame-step must be >= 1.")
    if int(max_people) <= 0:
        raise ValueError("--max-people must be >= 1.")
    if not np.isfinite(float(track_conf_min)) or float(track_conf_min) < 0.0:
        raise ValueError("--track-conf-min must be a finite float >= 0.")
    if not np.isfinite(float(track_max_jump_px)):
        raise ValueError("--track-max-jump-px must be finite.")
    if not np.isfinite(float(track_max_jump_diag_frac)) or float(track_max_jump_diag_frac) <= 0.0:
        raise ValueError("--track-max-jump-diag-frac must be a finite float > 0.")
    if int(track_max_lost) < 0:
        raise ValueError("--track-max-lost must be >= 0.")
    if not np.isfinite(float(track_target_x_frac)) or not np.isfinite(float(track_target_y_frac)):
        raise ValueError("--track-target-x-frac and --track-target-y-frac must be finite.")

    resolved_video_path = Path(video_path).expanduser()
    if not resolved_video_path.exists():
        raise FileNotFoundError(f"--video not found: {resolved_video_path}")

    resolved_save_path: Optional[Path] = None
    if save_path is not None:
        candidate = Path(save_path).expanduser()
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".mp4")
        resolved_save_path = candidate

    run_device = pick_device(device)
    use_half_requested = bool(int(half)) and run_device.startswith("cuda")
    use_half_keypoint = bool(use_half_requested)
    use_half_temporal = bool(use_half_requested)
    print(
        f"[runtime] device={run_device} "
        f"(requested={device if device else 'auto'}, cuda_available={torch.cuda.is_available()}, half={int(use_half_requested)})"
    )
    if int(max_people) <= 1:
        print("[track][WARN] --max-people=1 limits disambiguation when multiple people are present.")

    ckpt_path, resolved_arch = resolve_ckpt_and_arch(str(classification_model_path), arch)
    print(f"[model] arch={resolved_arch} ckpt={ckpt_path.as_posix()}")

    is_rf = str(resolved_arch).lower().strip() == "rf" or ckpt_path.suffix.lower() in {".pkl", ".pickle"}
    if is_rf:
        raise ValueError("RF checkpoints are not supported in run_inference_stream_packets.")

    state, meta = load_checkpoint(ckpt_path)
    state = clean_state_dict(state)

    T_raw = int(T) if int(T) > 0 else int(meta.get("T", meta.get("T_used", 64)) or 64)
    stride_raw = int(stride) if int(stride) > 0 else int(meta.get("stride", 16) or 16)
    T_raw = max(1, int(T_raw))
    stride_raw = max(1, int(stride_raw))

    T_final = max(1, int((int(T_raw) + int(frame_step) - 1) // int(frame_step)))
    stride_final = max(1, int((int(stride_raw) + int(frame_step) - 1) // int(frame_step)))
    if int(frame_step) > 1 and ((int(T_raw) % int(frame_step)) != 0 or (int(stride_raw) % int(frame_step)) != 0):
        print(
            f"[window][WARN] raw T/stride ({int(T_raw)}/{int(stride_raw)}) are not divisible by frame_step={int(frame_step)}; "
            "using ceil division for sampled windows."
        )
    print(
        f"[window] raw T/stride={int(T_raw)}/{int(stride_raw)} "
        f"-> sampled T/stride={int(T_final)}/{int(stride_final)} (frame_step={int(frame_step)})"
    )

    use_conf = bool(meta.get("use_conf", True))
    normalize = bool(meta.get("normalize", True))
    normalize_mode_final = str(normalize_mode) if normalize_mode else str(meta.get("normalize_mode") or "center_scale")
    add_vel = bool(meta.get("add_vel", True))
    add_acc = bool(meta.get("add_acc", True))
    add_global = bool(meta.get("add_global", True))
    add_mask = bool(meta.get("add_mask_channel", True))
    conf_thres = float(meta.get("conf_thres", 0.2))
    max_interp_gap = int(meta.get("max_interp_gap", 5))
    missing_mode_final = str(missing_mode) if missing_mode else str(meta.get("missing_mode") or "conf_thres")
    interp_mode_final = str(interp_mode) if interp_mode else str(meta.get("interp_mode") or "short_gap_hold")
    interp_group_final = int(interp_group) if int(interp_group) > 0 else int(meta.get("interp_group", 100) or 100)
    rp_center_mode_final = str(rp_center_mode) if rp_center_mode else str(meta.get("rp_center_mode") or "auto")
    rp_img_w_final: Optional[int] = None
    rp_img_h_final: Optional[int] = None
    if int(rp_img_w) > 0:
        rp_img_w_final = int(rp_img_w)
    elif meta.get("rp_img_w", None) is not None:
        rp_img_w_final = int(meta.get("rp_img_w"))  # type: ignore[arg-type]
    if int(rp_img_h) > 0:
        rp_img_h_final = int(rp_img_h)
    elif meta.get("rp_img_h", None) is not None:
        rp_img_h_final = int(meta.get("rp_img_h"))  # type: ignore[arg-type]
    min_valid_frac = float(meta.get("min_valid_frac", 0.3))

    num_classes = int(meta.get("num_classes", 0) or 0)
    in_features_meta = int(meta.get("in_features", 0) or 0)
    if num_classes <= 0:
        raise ValueError("Checkpoint missing num_classes. Use a checkpoint from training/train_models.py.")

    merge_fall_11_to_7 = int(num_classes) == 11
    display_num_classes = 7 if merge_fall_11_to_7 else int(num_classes)
    class_names = load_class_names(num_classes=display_num_classes, meta=meta, labels_file=labels_file)
    in_features = expected_in_features(
        use_conf=use_conf,
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        add_mask=add_mask,
    )
    if in_features_meta > 0 and int(in_features) != int(in_features_meta):
        raise ValueError(f"Feature mismatch: expected in_features={in_features}, ckpt expects {in_features_meta}")

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
        print("[WARN] missing keys:", missing_keys[:8], "..." if len(missing_keys) > 8 else "")
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    model.eval()
    if use_half_temporal:
        try:
            model.half()
        except Exception as error:
            use_half_temporal = False
            model.float()
            print(f"[runtime][WARN] Failed to enable FP16 for temporal model: {error}. Falling back to FP32.")

    keypoint_runtime = KeypointRuntime(
        model_path=Path(keypoint_model_path).expanduser(),
        device=run_device,
        backend=keypoint_backend,
    )
    print(
        f"[pose] backend={keypoint_runtime.backend} "
        f"model={Path(keypoint_model_path).expanduser()}"
    )

    cap = cv2.VideoCapture(str(resolved_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {resolved_video_path}")

    writer: Optional[cv2.VideoWriter] = None
    base_w = 0
    base_h = 0
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(src_fps) or src_fps <= 1e-3:
            src_fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        print(
            "[track] "
            f"conf_min={float(track_conf_min):.2f} "
            f"max_lost={int(track_max_lost)} "
            f"target=({float(track_target_x_frac):.2f},{float(track_target_y_frac):.2f}) "
            f"jump_px={'auto' if float(track_max_jump_px) <= 0.0 else f'{float(track_max_jump_px):.1f}'} "
            f"jump_diag_frac={float(track_max_jump_diag_frac):.3f}"
        )

        is_packet_stream = on_packet is not None and bool(no_display)
        fps_play = float(display_fps) if float(display_fps) > 1e-3 else float(src_fps)
        if is_packet_stream and float(display_fps) <= 1e-3:
            fps_play = min(float(src_fps), 18.0)
        frame_period_s = 1.0 / max(1e-6, float(fps_play))

        frames_buf: deque[np.ndarray] = deque()
        draw_xy_buf: deque[np.ndarray] = deque()
        draw_cf_buf: deque[np.ndarray] = deque()
        sample_xy_seq: List[np.ndarray] = []
        sample_cf_seq: List[np.ndarray] = []

        display_idx = 0
        processed_total = 0
        sampled_total = 0
        cap_done = False
        last_xy = np.zeros((K, 2), dtype=np.float32)
        last_cf = np.zeros((K,), dtype=np.float32)
        track_prev_center: Optional[np.ndarray] = None
        track_target_center: Optional[np.ndarray] = None
        track_max_jump_px_final: Optional[float] = None
        track_lost_count = 0

        window_preds: Dict[int, Tuple[int, float, Optional[float]]] = {}
        next_win_start = 0
        t_pose0 = time.perf_counter()

        def process_next_frame() -> bool:
            nonlocal processed_total, sampled_total, cap_done, last_xy, last_cf
            nonlocal track_prev_center, track_target_center, track_max_jump_px_final, track_lost_count

            ok, frame = cap.read()
            if not ok:
                cap_done = True
                return False

            raw_idx = int(processed_total)
            do_pose = (int(raw_idx) % int(frame_step)) == 0
            xy = last_xy
            cf = last_cf
            if do_pose:
                if track_target_center is None or track_max_jump_px_final is None:
                    h_img, w_img = frame.shape[:2]
                    track_target_center = np.array(
                        [float(w_img) * float(track_target_x_frac), float(h_img) * float(track_target_y_frac)],
                        dtype=np.float32,
                    )
                    frame_diag = float(np.hypot(float(w_img), float(h_img)))
                    if float(track_max_jump_px) > 0.0:
                        track_max_jump_px_final = float(track_max_jump_px)
                    else:
                        track_max_jump_px_final = float(track_max_jump_diag_frac) * float(frame_diag)

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
                    if int(track_lost_count) > int(track_max_lost):
                        track_prev_center = None

                sample_xy_seq.append(xy)
                sample_cf_seq.append(cf)
                sampled_total += 1
                last_xy = xy
                last_cf = cf

            frames_buf.append(frame)
            draw_xy_buf.append(xy)
            draw_cf_buf.append(cf)
            processed_total += 1

            if processed_total % 200 == 0:
                dt = time.perf_counter() - t_pose0
                if frame_count > 0:
                    pct = 100.0 * float(processed_total) / float(frame_count)
                    print(f"[pose] raw={processed_total}/{frame_count} ({pct:.1f}%) sampled={sampled_total} | {dt:.1f}s")
                else:
                    print(f"[pose] raw={processed_total} sampled={sampled_total} | {dt:.1f}s")

            return True

        def compute_window_pred(start: int) -> Tuple[int, float, Optional[float]]:
            if start in window_preds:
                return window_preds[start]
            if start >= sampled_total:
                raise RuntimeError(f"Cannot compute window {start}: sampled frame not processed yet (sampled_total={sampled_total}).")

            avail = int(sampled_total - start)
            L = int(min(int(T_final), max(0, avail)))
            if L <= 0:
                raise RuntimeError(f"Window start {start} has no available sampled frames (sampled_total={sampled_total}).")

            xy_seq = np.stack(sample_xy_seq[int(start): int(start) + int(L)], axis=0)
            conf_seq = np.stack(sample_cf_seq[int(start): int(start) + int(L)], axis=0)
            window_feat = make_window_features(
                xy_seq=xy_seq,
                conf_seq=conf_seq,
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
            window_preds[start] = (pred, pconf, p_fall)
            return window_preds[start]

        def compute_ready_windows() -> None:
            nonlocal next_win_start
            while True:
                if next_win_start in window_preds:
                    next_win_start += int(stride_final)
                    continue
                if cap_done:
                    if next_win_start >= sampled_total:
                        break
                    compute_window_pred(int(next_win_start))
                    next_win_start += int(stride_final)
                    continue
                if sampled_total >= int(next_win_start) + int(T_final):
                    compute_window_pred(int(next_win_start))
                    next_win_start += int(stride_final)
                    continue
                break

        while sampled_total < int(T_final) and not cap_done:
            process_next_frame()
        if processed_total <= 0:
            raise RuntimeError("Video had 0 frames.")

        if str(normalize_mode_final).lower().strip() == "paper_rp" and frames_buf:
            h_img, w_img = frames_buf[0].shape[:2]
            if rp_img_w_final is None:
                rp_img_w_final = int(w_img)
            if rp_img_h_final is None:
                rp_img_h_final = int(h_img)

        compute_window_pred(0)
        next_win_start = int(stride_final)

        if resolved_save_path is not None:
            base_h, base_w = frames_buf[0].shape[:2]
            writer = open_video_writer(save_path=resolved_save_path, fps=float(src_fps), frame_size=(base_w, base_h))

        window_name = "inference_on_video"
        fps_ema: Optional[float] = None
        stream_pace_fps_ema: Optional[float] = None
        ema_alpha = 0.1
        while True:
            if not frames_buf and cap_done:
                break

            t_frame_start = time.perf_counter()
            cap_done_at_frame_start = bool(cap_done)
            display_sample_idx = int(display_idx // max(1, int(frame_step)))

            target_sampled = int(display_sample_idx) + int(T_final) + 1
            while not cap_done and sampled_total < target_sampled:
                process_next_frame()

            compute_ready_windows()
            if not frames_buf:
                continue

            win_start = (int(display_sample_idx) // int(stride_final)) * int(stride_final)
            if win_start not in window_preds:
                while not cap_done and sampled_total < int(win_start) + int(T_final):
                    process_next_frame()
                    compute_ready_windows()
                if win_start not in window_preds:
                    compute_window_pred(int(win_start))

            pred, pconf, p_fall = window_preds.get(int(win_start), (-1, 0.0, None))
            label = class_names[pred] if 0 <= int(pred) < len(class_names) else "..."

            frame_original = frames_buf[0]
            xy = draw_xy_buf[0]
            cf = draw_cf_buf[0]

            frame_info = f"frame {int(display_idx) + 1}"
            if frame_count > 0:
                frame_info += f"/{frame_count}"

            fps_for_hud = float(fps_ema) if fps_ema is not None else float(fps_play)
            hud = [
                frame_info,
                f"fps: {float(fps_for_hud):.1f}",
                f"pose: {label} ({float(pconf):.2f})" if int(pred) >= 0 else "pose: ...",
                f"T={int(T_final)} stride={int(stride_final)} sampled (k={int(frame_step)})",
            ]
            if p_fall is not None:
                hud.append(f"fall_prob: {float(p_fall):.2f}")
            hud_to_draw = list(hud)

            if on_packet is not None:
                ok_jpg, encoded = cv2.imencode(
                    ".jpg",
                    frame_original,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not ok_jpg:
                    raise RuntimeError(f"Failed to encode frame {int(display_idx)} to JPEG.")
                frame_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
                packet: Dict[str, Any] = {
                    "type": "frame",
                    "frame_index": int(display_idx),
                    "frame_number": int(display_idx) + 1,
                    "frame_count": int(frame_count),
                    "fps": float(fps_for_hud),
                    "pred": {
                        "label": str(label),
                        "conf": float(pconf),
                        "class_id": int(pred),
                    },
                    "params": {
                        "T": int(T_final),
                        "stride": int(stride_final),
                        "k": int(frame_step),
                    },
                    "hud_lines": list(hud_to_draw),
                    "pose": {
                        "format": "coco17",
                        "xy": np.asarray(xy, dtype=np.float32).tolist(),
                        "conf": np.asarray(cf, dtype=np.float32).tolist(),
                        "conf_thres": float(conf_thres),
                        "skeleton": [[int(a), int(b)] for a, b in SKELETON],
                    },
                    "frame_jpeg_b64": frame_b64,
                    "size": {"w": int(frame_original.shape[1]), "h": int(frame_original.shape[0])},
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
                if p_fall is not None:
                    packet["pred"]["fall_prob"] = float(p_fall)
                on_packet(packet)

            key = -1
            needs_rendered_frame = (on_frame is not None) or (writer is not None) or (not no_display)
            if needs_rendered_frame:
                frame_to_render = draw_pose(frame_original.copy(), xy, cf, conf_thres=conf_thres)
                frame_to_render = draw_hud(frame_to_render, hud_to_draw)
                if on_frame is not None:
                    on_frame(frame_to_render)

                if writer is not None:
                    frame_h, frame_w = frame_to_render.shape[:2]
                    frame_to_write = frame_to_render
                    if frame_h != base_h or frame_w != base_w:
                        frame_to_write = cv2.resize(frame_to_render, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
                    writer.write(frame_to_write)

                if not no_display:
                    cv2.imshow(window_name, frame_to_render)
                    if bool(realtime):
                        elapsed_s = time.perf_counter() - t_frame_start
                        remaining_s = frame_period_s - elapsed_s
                        wait_ms = int(max(1, remaining_s * 1000.0)) if remaining_s > 0 else 1
                    else:
                        wait_ms = 1
                    key = cv2.waitKey(wait_ms) & 0xFF

            if bool(realtime) and is_packet_stream:
                pace_fps = float(fps_play)
                if cap_done_at_frame_start and stream_pace_fps_ema is not None and float(stream_pace_fps_ema) > 1e-6:
                    pace_fps = min(float(fps_play), float(stream_pace_fps_ema))
                elapsed_s = time.perf_counter() - t_frame_start
                remaining_s = (1.0 / max(1e-6, float(pace_fps))) - elapsed_s
                if remaining_s > 0.0:
                    time.sleep(remaining_s)

            total_ms = (time.perf_counter() - t_frame_start) * 1000.0
            inst_fps = 1000.0 / max(1e-6, total_ms)
            fps_ema = inst_fps if fps_ema is None else (1.0 - ema_alpha) * float(fps_ema) + ema_alpha * inst_fps
            if is_packet_stream and not cap_done_at_frame_start:
                stream_pace_fps_ema = (
                    float(fps_ema)
                    if stream_pace_fps_ema is None
                    else (1.0 - ema_alpha) * float(stream_pace_fps_ema) + ema_alpha * float(fps_ema)
                )

            should_quit = (key in (ord("q"), 27))

            frames_buf.popleft()
            draw_xy_buf.popleft()
            draw_cf_buf.popleft()
            display_idx += 1
            if should_quit:
                break

        return 0
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()


def run_inference_stream(
    *,
    video_path: Path,
    classification_model_path: Path,
    keypoint_model_path: Path,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
    save_path: Optional[Path] = None,
    no_display: bool = True,
    realtime: bool = True,
    display_fps: float = 0.0,
    **inference_options: Any,
) -> int:
    return run_inference_stream_packets(
        video_path=video_path,
        classification_model_path=classification_model_path,
        keypoint_model_path=keypoint_model_path,
        on_packet=None,
        on_frame=on_frame,
        save_path=save_path,
        no_display=bool(no_display),
        realtime=bool(realtime),
        display_fps=float(display_fps),
        **dict(inference_options),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream windowed pose inference on an MP4 with YOLO pose overlay.")
    ap.add_argument("--video", type=str, required=True, help="Path to input .mp4")
    ap.add_argument("--model", type=str, required=True, help="Checkpoint *.pt/*.pkl OR model folder OR model .py")
    ap.add_argument("--arch", type=str, default=None, choices=KNOWN_ARCHES, help="Override model architecture if needed")
    ap.add_argument("--keypoint-model", type=str, default="models/keypoint/ultralytics/yolo11l-pose.pt")
    ap.add_argument(
        "--keypoint-backend",
        type=str,
        default=None,
        choices=["yolo", "alphapose", "vitpose"],
        help="Override keypoint backend (auto-detected from --keypoint-model when omitted).",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument(
        "--max-people",
        type=int,
        default=10,
        help="Maximum YOLO pose candidates per frame to consider for center-based target tracking.",
    )
    ap.add_argument(
        "--track-conf-min",
        type=float,
        default=0.75,
        help="During acquisition (no prior track), prefer boxes with conf >= this threshold.",
    )
    ap.add_argument(
        "--track-max-jump-px",
        type=float,
        default=0.0,
        help="Max tracked-center jump in pixels. <=0 => use frame diagonal * --track-max-jump-diag-frac.",
    )
    ap.add_argument(
        "--track-max-jump-diag-frac",
        type=float,
        default=0.25,
        help="Fallback jump gate as a fraction of frame diagonal when --track-max-jump-px <= 0.",
    )
    ap.add_argument("--track-max-lost", type=int, default=10, help="Reset tracking after this many consecutive misses.")
    ap.add_argument("--track-target-x-frac", type=float, default=0.5, help="Target x location as fraction of frame width.")
    ap.add_argument("--track-target-y-frac", type=float, default=0.5, help="Target y location as fraction of frame height.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--half", type=int, default=0, help="Use FP16 on CUDA for YOLO+temporal model (0/1)")
    ap.add_argument("--T", type=int, default=0, help="0 => use ckpt T_used/T, else override")
    ap.add_argument("--stride", type=int, default=0, help="0 => use ckpt stride, else override")
    ap.add_argument(
        "--frame-step", "--k",
        type=int,
        default=1,
        help="Run YOLO pose every k raw frames (k>=1). Window T/stride are defined in raw frames and scaled to sampled frames.",
    )
    ap.add_argument(
        "--normalize-mode",
        type=str,
        default=None,
        choices=["center_scale", "paper_rp"],
        help="Override checkpoint normalize_mode when --normalize 1 (center_scale or paper_rp).",
    )
    ap.add_argument(
        "--missing-mode",
        type=str,
        default=None,
        choices=["conf_thres", "zeros_only", "conf_or_zeros"],
        help="Override checkpoint missing_mode (conf_thres, zeros_only, conf_or_zeros).",
    )
    ap.add_argument(
        "--interp-mode",
        type=str,
        default=None,
        choices=["short_gap_hold", "paper_group_linear"],
        help="Override checkpoint interp_mode (short_gap_hold or paper_group_linear).",
    )
    ap.add_argument(
        "--interp-group",
        type=int,
        default=0,
        help="Override checkpoint interp_group (>0). Only used for --interp-mode paper_group_linear.",
    )
    ap.add_argument(
        "--rp-center-mode",
        type=str,
        default=None,
        choices=["auto", "normalized_01", "pixel"],
        help="Override checkpoint rp_center_mode for --normalize-mode paper_rp.",
    )
    ap.add_argument("--rp-img-w", type=int, default=0, help="Image width W for paper_rp when using pixel coordinates (0 => auto from video).")
    ap.add_argument("--rp-img-h", type=int, default=0, help="Image height H for paper_rp when using pixel coordinates (0 => auto from video).")
    ap.add_argument("--labels-file", type=str, default=None)
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save annotated output video (e.g. out.mp4). If a directory, writes <video_stem>_annotated.mp4 inside.",
    )
    ap.add_argument("--display-fps", type=float, default=0.0, help="0 => source fps")
    ap.add_argument("--profile", type=int, default=0, help="Enable profiling outputs (0/1).")
    ap.add_argument(
        "--profile-out",
        type=str,
        default=None,
        help="Base directory for profiling outputs. A unique per-run subdirectory is created (timestamp + model).",
    )
    ap.add_argument("--profile-duration-s", type=float, default=0.0, help="0 => full run, else stop after N seconds.")
    ap.add_argument("--hw-sample-hz", type=float, default=1.0, help="Hardware metrics sample rate (Hz).")
    ap.add_argument("--no-display", type=int, default=0, help="Run headless: skip imshow/waitKey (0/1).")
    args = ap.parse_args()
    frame_step = int(args.frame_step)
    if int(frame_step) <= 0:
        raise ValueError("--frame-step must be >= 1.")
    if int(args.max_people) <= 0:
        raise ValueError("--max-people must be >= 1.")

    track_conf_min = float(args.track_conf_min)
    if not np.isfinite(track_conf_min) or track_conf_min < 0.0:
        raise ValueError("--track-conf-min must be a finite float >= 0.")
    track_max_jump_px_arg = float(args.track_max_jump_px)
    if not np.isfinite(track_max_jump_px_arg):
        raise ValueError("--track-max-jump-px must be finite.")
    track_max_jump_diag_frac = float(args.track_max_jump_diag_frac)
    if not np.isfinite(track_max_jump_diag_frac) or track_max_jump_diag_frac <= 0.0:
        raise ValueError("--track-max-jump-diag-frac must be a finite float > 0.")
    track_max_lost = int(args.track_max_lost)
    if track_max_lost < 0:
        raise ValueError("--track-max-lost must be >= 0.")
    track_target_x_frac = float(args.track_target_x_frac)
    track_target_y_frac = float(args.track_target_y_frac)
    if not np.isfinite(track_target_x_frac) or not np.isfinite(track_target_y_frac):
        raise ValueError("--track-target-x-frac and --track-target-y-frac must be finite.")

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"--video not found: {video_path}")

    save_path: Optional[Path] = None
    if args.save:
        save_arg = Path(args.save).expanduser()
        if str(args.save).endswith(("/", "\\")) or (save_arg.exists() and save_arg.is_dir()):
            save_path = save_arg / f"{video_path.stem}_annotated.mp4"
        else:
            save_path = save_arg
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".mp4")

    profile_enabled = bool(int(args.profile))
    no_display = bool(int(args.no_display))
    profile_duration_s = max(0.0, float(args.profile_duration_s))
    hw_sample_hz = max(0.1, float(args.hw_sample_hz))
    profile_out_dir: Optional[Path] = None

    device = pick_device(args.device)
    use_half = bool(int(args.half)) and device.startswith("cuda")
    sync_cuda_timing = bool(device.startswith("cuda") and torch.cuda.is_available())
    print(
        f"[runtime] device={device} "
        f"(requested={args.device if args.device else 'auto'}, cuda_available={torch.cuda.is_available()}, half={int(use_half)})"
    )
    if int(args.max_people) <= 1:
        print("[track][WARN] --max-people=1 limits disambiguation when multiple people are present.")

    ckpt_path, arch = resolve_ckpt_and_arch(args.model, args.arch)
    print(f"[model] arch={arch} ckpt={ckpt_path.as_posix()}")
    if profile_enabled:
        profile_out_dir = _pick_profile_out_dir(
            profile_out_arg=args.profile_out,
            save_path=save_path,
            ckpt_path=ckpt_path,
            arch=arch,
        )

    is_rf = str(arch).lower().strip() == "rf" or ckpt_path.suffix.lower() in {".pkl", ".pickle"}
    rf_model = None

    if is_rf:
        meta = load_rf_checkpoint(ckpt_path)
        rf_model = meta.get("model", None)
    else:
        state, meta = load_checkpoint(ckpt_path)
        state = clean_state_dict(state)

    # Preproc config (prefer checkpoint meta)
    T_raw = int(args.T) if int(args.T) > 0 else int(meta.get("T", meta.get("T_used", 64)) or 64)
    stride_raw = int(args.stride) if int(args.stride) > 0 else int(meta.get("stride", 16) or 16)
    T_raw = max(1, int(T_raw))
    stride_raw = max(1, int(stride_raw))

    # Keep temporal coverage roughly constant when only every k-th raw frame is sampled.
    T_final = max(1, int((int(T_raw) + int(frame_step) - 1) // int(frame_step)))
    stride_final = max(1, int((int(stride_raw) + int(frame_step) - 1) // int(frame_step)))
    if int(frame_step) > 1 and ((int(T_raw) % int(frame_step)) != 0 or (int(stride_raw) % int(frame_step)) != 0):
        print(
            f"[window][WARN] raw T/stride ({int(T_raw)}/{int(stride_raw)}) are not divisible by frame_step={int(frame_step)}; "
            "using ceil division for sampled windows."
        )
    print(
        f"[window] raw T/stride={int(T_raw)}/{int(stride_raw)} "
        f"-> sampled T/stride={int(T_final)}/{int(stride_final)} (frame_step={int(frame_step)})"
    )

    use_conf = bool(meta.get("use_conf", True))
    normalize = bool(meta.get("normalize", True))
    normalize_mode = str(args.normalize_mode) if args.normalize_mode else str(meta.get("normalize_mode") or "center_scale")
    add_vel = bool(meta.get("add_vel", True))
    add_acc = bool(meta.get("add_acc", True))
    add_global = bool(meta.get("add_global", True))
    add_mask = bool(meta.get("add_mask_channel", True))
    conf_thres = float(meta.get("conf_thres", 0.2))
    max_interp_gap = int(meta.get("max_interp_gap", 5))
    missing_mode = str(args.missing_mode) if args.missing_mode else str(meta.get("missing_mode") or "conf_thres")
    interp_mode = str(args.interp_mode) if args.interp_mode else str(meta.get("interp_mode") or "short_gap_hold")
    interp_group = int(args.interp_group) if int(args.interp_group) > 0 else int(meta.get("interp_group", 100) or 100)
    rp_center_mode = str(args.rp_center_mode) if args.rp_center_mode else str(meta.get("rp_center_mode") or "auto")
    rp_img_w: Optional[int] = None
    rp_img_h: Optional[int] = None
    if int(args.rp_img_w) > 0:
        rp_img_w = int(args.rp_img_w)
    elif meta.get("rp_img_w", None) is not None:
        rp_img_w = int(meta.get("rp_img_w"))  # type: ignore[arg-type]
    if int(args.rp_img_h) > 0:
        rp_img_h = int(args.rp_img_h)
    elif meta.get("rp_img_h", None) is not None:
        rp_img_h = int(meta.get("rp_img_h"))  # type: ignore[arg-type]
    min_valid_frac = float(meta.get("min_valid_frac", 0.3))

    num_classes = int(meta.get("num_classes", 0) or 0)
    in_features_meta = int(meta.get("in_features", 0) or 0)
    if num_classes <= 0:
        if is_rf:
            nln = meta.get("new_label_names", None)
            if isinstance(nln, (list, tuple)):
                num_classes = int(len(nln))
            if int(num_classes) <= 0:
                num_classes = 7
        else:
            raise ValueError("Checkpoint missing num_classes. Use a checkpoint from training/train_models.py.")

    merge_fall_11_to_7 = int(num_classes) == 11
    display_num_classes = 7 if merge_fall_11_to_7 else int(num_classes)
    class_names = load_class_names(num_classes=display_num_classes, meta=meta, labels_file=args.labels_file)
    in_features = expected_in_features(
        use_conf=use_conf,
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        add_mask=add_mask,
    )
    if in_features_meta > 0 and int(in_features) != int(in_features_meta):
        raise ValueError(f"Feature mismatch: expected in_features={in_features}, ckpt expects {in_features_meta}")

    if is_rf:
        if rf_model is None:
            raise ValueError("RF checkpoint missing 'model'.")
    else:
        node_features_meta = meta.get("node_features", None)
        if node_features_meta is None:
            nf = int(in_features // K)
            node_features_meta = nf if nf * K == int(in_features) else None

        model = build_temporal_model(
            arch=arch,
            in_features=int(in_features),
            num_classes=int(num_classes),
            device=device,
            T_used=int(T_final),
            node_features=int(node_features_meta) if node_features_meta is not None else None,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print("[WARN] missing keys:", missing[:8], "..." if len(missing) > 8 else "")
        if unexpected:
            print("[WARN] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
        model.eval()

    keypoint_runtime = KeypointRuntime(
        model_path=Path(args.keypoint_model).expanduser(),
        device=device,
        backend=args.keypoint_backend,
    )
    print(f"[pose] backend={keypoint_runtime.backend} model={Path(args.keypoint_model).expanduser()}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer: Optional[cv2.VideoWriter] = None
    h0 = 0
    w0 = 0
    time_rows: List[Dict[str, Any]] = []
    hw_rows: List[Dict[str, Any]] = []
    hw_sampler: Optional[HardwareSampler] = None
    profile_run_t0 = time.perf_counter()
    metrics_cutoff_active = False
    metrics_cutoff_t_s: Optional[float] = None
    last_hw_print_t = 0.0
    hw_print_interval_s = max(0.5, 1.0 / max(hw_sample_hz, 1e-3))

    try:
        if profile_enabled and profile_out_dir is not None:
            profile_out_dir.mkdir(parents=True, exist_ok=True)
            hw_sampler = HardwareSampler(sample_hz=hw_sample_hz)
            hw_backend = hw_sampler.start()
            print(f"[profile] enabled -> {profile_out_dir.as_posix()}")
            print(f"[profile] hw backend: {hw_backend}")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(src_fps) or src_fps <= 1e-3:
            src_fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        print(
            "[track] "
            f"conf_min={float(track_conf_min):.2f} "
            f"max_lost={int(track_max_lost)} "
            f"target=({float(track_target_x_frac):.2f},{float(track_target_y_frac):.2f}) "
            f"jump_px={'auto' if float(track_max_jump_px_arg) <= 0.0 else f'{float(track_max_jump_px_arg):.1f}'} "
            f"jump_diag_frac={float(track_max_jump_diag_frac):.3f}"
        )

        fps_play = float(args.display_fps) if float(args.display_fps) > 1e-3 else float(src_fps)
        frame_period_s = 1.0 / max(1e-6, float(fps_play))

        frames_buf: deque[np.ndarray] = deque()
        draw_xy_buf: deque[np.ndarray] = deque()
        draw_cf_buf: deque[np.ndarray] = deque()
        prep_ms_buf: deque[float] = deque()
        yolo_ms_buf: deque[float] = deque()
        sample_xy_seq: List[np.ndarray] = []
        sample_cf_seq: List[np.ndarray] = []

        display_idx = 0      # absolute raw frame index being displayed
        processed_total = 0  # absolute raw frame count read from video
        sampled_total = 0    # absolute sampled-frame count used by temporal model
        cap_done = False
        last_xy = np.zeros((K, 2), dtype=np.float32)
        last_cf = np.zeros((K,), dtype=np.float32)
        track_prev_center: Optional[np.ndarray] = None
        track_target_center: Optional[np.ndarray] = None
        track_max_jump_px: Optional[float] = None
        track_lost_count = 0

        window_preds: Dict[int, Tuple[int, float, Optional[float]]] = {}
        window_stage_ms: Dict[int, Tuple[float, float]] = {}
        next_win_start = 0

        t_pose0 = time.perf_counter()

        def stop_due_profile_duration() -> bool:
            if not profile_enabled:
                return False
            if profile_duration_s <= 0.0:
                return False
            return (time.perf_counter() - profile_run_t0) >= profile_duration_s

        def process_next_frame() -> bool:
            nonlocal processed_total, sampled_total, cap_done, last_xy, last_cf
            nonlocal track_prev_center, track_target_center, track_max_jump_px, track_lost_count

            cap_read_ms = 0.0
            if profile_enabled:
                t_cap0 = time.perf_counter()
            ok, frame = cap.read()
            if profile_enabled:
                cap_read_ms = (time.perf_counter() - t_cap0) * 1000.0
            if not ok:
                cap_done = True
                return False

            raw_idx = int(processed_total)
            do_pose = (int(raw_idx) % int(frame_step)) == 0
            yolo_infer_ms = 0.0
            xy = last_xy
            cf = last_cf
            if do_pose:
                if track_target_center is None or track_max_jump_px is None:
                    h_img, w_img = frame.shape[:2]
                    track_target_center = np.array(
                        [float(w_img) * float(track_target_x_frac), float(h_img) * float(track_target_y_frac)],
                        dtype=np.float32,
                    )
                    frame_diag = float(np.hypot(float(w_img), float(h_img)))
                    if float(track_max_jump_px_arg) > 0.0:
                        track_max_jump_px = float(track_max_jump_px_arg)
                    else:
                        track_max_jump_px = float(track_max_jump_diag_frac) * float(frame_diag)

                if profile_enabled:
                    _maybe_cuda_sync(sync_cuda_timing)
                    t_yolo0 = time.perf_counter()
                xy, cf, new_center, found = pose_on_frame(
                    keypoint_runtime=keypoint_runtime,
                    frame_bgr=frame,
                    imgsz=int(args.imgsz),
                    yolo_conf=float(args.yolo_conf),
                    max_people=int(args.max_people),
                    use_half=use_half,
                    prev_center=track_prev_center,
                    target_center=track_target_center,
                    conf_min=float(track_conf_min),
                    max_jump_px=float(track_max_jump_px),
                )
                if found:
                    track_prev_center = new_center
                    track_lost_count = 0
                else:
                    track_lost_count += 1
                    if int(track_lost_count) > int(track_max_lost):
                        track_prev_center = None
                if profile_enabled:
                    _maybe_cuda_sync(sync_cuda_timing)
                    yolo_infer_ms = (time.perf_counter() - t_yolo0) * 1000.0
                sample_xy_seq.append(xy)
                sample_cf_seq.append(cf)
                sampled_total += 1
                last_xy = xy
                last_cf = cf

            frames_buf.append(frame)
            draw_xy_buf.append(xy)
            draw_cf_buf.append(cf)
            if profile_enabled:
                prep_ms_buf.append(cap_read_ms)
                yolo_ms_buf.append(yolo_infer_ms)
            processed_total += 1

            if processed_total % 200 == 0:
                dt = time.perf_counter() - t_pose0
                if frame_count > 0:
                    pct = 100.0 * float(processed_total) / float(frame_count)
                    print(f"[pose] raw={processed_total}/{frame_count} ({pct:.1f}%) sampled={sampled_total} | {dt:.1f}s")
                else:
                    print(f"[pose] raw={processed_total} sampled={sampled_total} | {dt:.1f}s")

            return True

        def compute_window_pred(start: int) -> Tuple[int, float, Optional[float]]:
            if start in window_preds:
                return window_preds[start]
            if start >= sampled_total:
                raise RuntimeError(f"Cannot compute window {start}: sampled frame not processed yet (sampled_total={sampled_total}).")

            avail = int(sampled_total - start)
            L = int(min(int(T_final), max(0, avail)))
            if L <= 0:
                raise RuntimeError(f"Window start {start} has no available sampled frames (sampled_total={sampled_total}).")

            xy_seq = np.stack(sample_xy_seq[int(start): int(start) + int(L)], axis=0)
            conf_seq = np.stack(sample_cf_seq[int(start): int(start) + int(L)], axis=0)

            win_feat_ms = 0.0
            if profile_enabled:
                t_feat0 = time.perf_counter()
            window_feat = make_window_features(
                xy_seq=xy_seq,
                conf_seq=conf_seq,
                T=int(T_final),
                use_conf=use_conf,
                normalize=normalize,
                normalize_mode=normalize_mode,
                add_vel=add_vel,
                add_acc=add_acc,
                add_global=add_global,
                add_mask=add_mask,
                conf_thres=conf_thres,
                max_interp_gap=max_interp_gap,
                missing_mode=missing_mode,
                interp_mode=interp_mode,
                interp_group=int(interp_group),
                rp_center_mode=rp_center_mode,
                rp_img_w=rp_img_w,
                rp_img_h=rp_img_h,
                min_valid_frac=min_valid_frac,
            )
            if profile_enabled:
                win_feat_ms = (time.perf_counter() - t_feat0) * 1000.0

            temporal_infer_ms = 0.0
            if profile_enabled:
                _maybe_cuda_sync(sync_cuda_timing)
                t_temp0 = time.perf_counter()
            pred, pconf, p_fall = infer_one_window(
                model=model,
                window_feat=window_feat,
                device=device,
                use_half=use_half,
                merge_fall_11_to_7=merge_fall_11_to_7,
            )
            if profile_enabled:
                _maybe_cuda_sync(sync_cuda_timing)
                temporal_infer_ms = (time.perf_counter() - t_temp0) * 1000.0

            window_preds[start] = (pred, pconf, p_fall)
            if profile_enabled:
                window_stage_ms[start] = (float(win_feat_ms), float(temporal_infer_ms))
            return window_preds[start]

        def compute_ready_windows() -> None:
            nonlocal next_win_start

            while True:
                if next_win_start in window_preds:
                    next_win_start += int(stride_final)
                    continue

                if cap_done:
                    if next_win_start >= sampled_total:
                        break
                    compute_window_pred(int(next_win_start))
                    next_win_start += int(stride_final)
                    continue

                if sampled_total >= int(next_win_start) + int(T_final):
                    compute_window_pred(int(next_win_start))
                    next_win_start += int(stride_final)
                    continue

                break

        # Warm up: read enough frames to make the first window prediction, then start display.
        while sampled_total < int(T_final) and not cap_done and not stop_due_profile_duration():
            process_next_frame()
        if processed_total <= 0:
            raise RuntimeError("Video had 0 frames.")

        # For paper_rp normalisation, default image dims to the video frame size.
        if str(normalize_mode).lower().strip() == "paper_rp" and frames_buf:
            h_img, w_img = frames_buf[0].shape[:2]
            if rp_img_w is None:
                rp_img_w = int(w_img)
            if rp_img_h is None:
                rp_img_h = int(h_img)

        compute_window_pred(0)
        next_win_start = int(stride_final)

        if save_path is not None:
            h0, w0 = frames_buf[0].shape[:2]
            writer = open_video_writer(save_path=save_path, fps=float(src_fps), frame_size=(w0, h0))

        # Main loop: keep a small lookahead so each next-window prediction is ready in time.
        window_name = "inference_on_video"
        fps_ema: Optional[float] = None
        ema_alpha = 0.1
        while True:
            if stop_due_profile_duration():
                break
            if not frames_buf and cap_done:
                break

            t_frame_start = time.perf_counter()
            display_sample_idx = int(display_idx // max(1, int(frame_step)))

            # Keep a lead of ~T frames (plus 1) so we can predict the next window before it is displayed.
            target_sampled = int(display_sample_idx) + int(T_final) + 1
            while not cap_done and sampled_total < target_sampled and not stop_due_profile_duration():
                process_next_frame()

            compute_ready_windows()

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

            if (
                profile_enabled
                and not metrics_cutoff_active
                and cap_done
                and next_win_start >= sampled_total
                and (int(sampled_total) - int(display_sample_idx)) <= int(T_final)
            ):
                metrics_cutoff_active = True
                metrics_cutoff_t_s = float(time.perf_counter() - profile_run_t0)
                cutoff_frame_idx = int(display_idx)
                print(
                    f"[profile] timing metrics cutoff at frame {cutoff_frame_idx} "
                    "(tail drain phase after final window inference)."
                )

            if not frames_buf:
                continue

            post_extra_ms = 0.0
            if profile_enabled:
                t_post_extra0 = time.perf_counter()

            win_start = (int(display_sample_idx) // int(stride_final)) * int(stride_final)
            if win_start not in window_preds:
                # If we're behind (slow device), wait until we can compute it, then continue.
                while not cap_done and sampled_total < int(win_start) + int(T_final) and not stop_due_profile_duration():
                    process_next_frame()
                    compute_ready_windows()
                if win_start not in window_preds:
                    compute_window_pred(int(win_start))

            pred, pconf, p_fall = window_preds.get(int(win_start), (-1, 0.0, None))
            label = class_names[pred] if 0 <= int(pred) < len(class_names) else "..."

            frame = frames_buf[0].copy()
            xy = draw_xy_buf[0]
            cf = draw_cf_buf[0]

            preprocess_ms = float(prep_ms_buf[0]) if profile_enabled and prep_ms_buf else 0.0
            yolo_infer_ms = float(yolo_ms_buf[0]) if profile_enabled and yolo_ms_buf else 0.0
            win_feature_ms, temporal_infer_ms = window_stage_ms.get(int(win_start), (0.0, 0.0))

            frame_info = f"frame {int(display_idx) + 1}"
            if frame_count > 0:
                frame_info += f"/{frame_count}"

            if profile_enabled:
                post_extra_ms = (time.perf_counter() - t_post_extra0) * 1000.0

            draw_ms = 0.0
            writer_ms = 0.0
            display_wait_ms = 0.0

            t_vis0 = time.perf_counter()
            t_draw0 = time.perf_counter()
            frame = draw_pose(frame, xy, cf, conf_thres=conf_thres)

            fps_for_hud = float(fps_ema) if fps_ema is not None else float(fps_play)
            win_id = int(win_start) // max(1, int(stride_final))
            win_start_raw = int(win_start) * int(frame_step)
            hud = [
                frame_info,
                f"fps: {float(fps_for_hud):.1f}",
                f"pose: {label} ({float(pconf):.2f})" if int(pred) >= 0 else "pose: ...",
                f"T={int(T_final)} stride={int(stride_final)} sampled (k={int(frame_step)})",
            ]
            if p_fall is not None:
                hud.append(f"fall_prob: {float(p_fall):.2f}")
            hud_to_draw = list(hud)
            frame = draw_hud(frame, hud_to_draw)
            draw_ms = (time.perf_counter() - t_draw0) * 1000.0

            if writer is not None:
                t_writer0 = time.perf_counter()
                frame_h, frame_w = frame.shape[:2]
                frame_to_write = frame
                if frame_h != h0 or frame_w != w0:
                    frame_to_write = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_LINEAR)
                writer.write(frame_to_write)
                writer_ms = (time.perf_counter() - t_writer0) * 1000.0

            key = -1
            if not no_display:
                t_display0 = time.perf_counter()
                cv2.imshow(window_name, frame)
                # Wait just the remaining time to hit target display FPS (accounting for processing).
                elapsed_s = time.perf_counter() - t_frame_start
                remaining_s = frame_period_s - elapsed_s
                wait_ms = int(max(1, remaining_s * 1000.0)) if remaining_s > 0 else 1
                key = cv2.waitKey(wait_ms) & 0xFF
                display_wait_ms = (time.perf_counter() - t_display0) * 1000.0

            visualisation_ms = (time.perf_counter() - t_vis0) * 1000.0
            total_ms = (time.perf_counter() - t_frame_start) * 1000.0
            inst_fps = 1000.0 / max(1e-6, total_ms)
            fps_ema = inst_fps if fps_ema is None else (1.0 - ema_alpha) * float(fps_ema) + ema_alpha * inst_fps

            if profile_enabled and not metrics_cutoff_active:
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
            draw_xy_buf.popleft()
            draw_cf_buf.popleft()
            if profile_enabled:
                if prep_ms_buf:
                    prep_ms_buf.popleft()
                if yolo_ms_buf:
                    yolo_ms_buf.popleft()
            display_idx += 1

            if should_quit:
                break

        return 0
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()
        if hw_sampler is not None:
            hw_sampler.stop()
            hw_rows = hw_sampler.get_samples()
        if metrics_cutoff_t_s is not None:
            hw_rows = [r for r in hw_rows if _safe_float(r.get("t_s", np.nan)) <= float(metrics_cutoff_t_s)]
        if profile_enabled and profile_out_dir is not None:
            try:
                _save_profile_artifacts(profile_out=profile_out_dir, time_rows=time_rows, hw_rows=hw_rows)
                print(f"[profile] wrote outputs to: {profile_out_dir.as_posix()}")
            except Exception as e:
                print(f"[WARN] Failed to save profiling outputs: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
