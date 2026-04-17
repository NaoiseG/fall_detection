from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from inference.classifier_adapters import Prediction, TemporalClassifierAdapter, WindowData
from inference.pose_pipeline import K, SKELETON, PosePipeline


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return float(default)
    return out


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


def _parse_first_number(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"[+-]?\d+(?:\.\d+)?", v)
        if m:
            return _safe_float(m.group(0))
    return float("nan")


def _percentile_valid(vals: List[float], q: float) -> float:
    good = [float(x) for x in vals if _is_finite(x)]
    if not good:
        return float("nan")
    return float(np.percentile(good, q))


def _to_csv_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else ""
    return v


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: _to_csv_cell(row.get(k, "")) for k in fieldnames}
            writer.writerow(out)


def _json_safe_number(v: Any) -> Optional[float]:
    fv = _safe_float(v)
    if np.isfinite(fv):
        return float(fv)
    return None


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
        if total > 0.0:
            sample["ram_used_pct"] = 100.0 * used / total

    m_cpu = re.search(r"\bCPU\s+\[([^\]]+)\]", line, flags=re.IGNORECASE)
    if m_cpu:
        loads: List[float] = []
        for tok in m_cpu.group(1).split(","):
            m_pct = re.search(r"([+-]?\d+(?:\.\d+)?)%", tok)
            if m_pct:
                loads.append(_safe_float(m_pct.group(1)))
        sample["cpu_pct"] = _avg_valid(loads)

    m_gpu = re.search(r"\bGR3D(?:_FREQ)?\s+([+-]?\d+(?:\.\d+)?)%", line, flags=re.IGNORECASE)
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
        for key in ("value", "val", "avg", "cur", "current", "usage", "percent", "perc"):
            if key in obj and _is_finite(obj[key]):
                return float(obj[key])
        vals = [_extract_numeric_from_obj(v) for v in obj.values()]
        return _avg_valid(vals)
    if isinstance(obj, (list, tuple)):
        vals = [_extract_numeric_from_obj(v) for v in obj]
        return _avg_valid(vals)
    return float("nan")


def _normalize_stat_key(name: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def _extract_percent_from_obj(obj: Any) -> float:
    """
    Extract utilization-style percentages without accidentally treating
    temperatures or raw frequencies as percentages.

    This is used for jtop GPU parsing where values may look like:
      - 95
      - "95%"
      - "95%@1109"
      - {"val": 95, "frq": 1109}
    """
    if isinstance(obj, (int, float)):
        val = float(obj)
        if np.isfinite(val) and 0.0 <= val <= 100.0:
            return val
        return float("nan")

    if isinstance(obj, str):
        matches = [
            _safe_float(m.group(1))
            for m in re.finditer(r"([+-]?\d+(?:\.\d+)?)\s*%", obj)
        ]
        return _avg_valid(matches)

    if isinstance(obj, dict):
        preferred_keys = (
            "load",
            "usage",
            "percent",
            "perc",
            "value",
            "val",
            "avg",
            "cur",
            "current",
        )

        preferred_vals = [_extract_percent_from_obj(obj[key]) for key in preferred_keys if key in obj]
        preferred_pct = _avg_valid(preferred_vals)
        if np.isfinite(preferred_pct):
            return preferred_pct

        vals = []
        for key, value in obj.items():
            key_l = _normalize_stat_key(key)
            if any(token in key_l for token in ("temp", "freq", "mhz", "clock", "power", "volt", "fan")):
                continue
            vals.append(_extract_percent_from_obj(value))
        return _avg_valid(vals)

    if isinstance(obj, (list, tuple)):
        vals = [_extract_percent_from_obj(v) for v in obj]
        return _avg_valid(vals)

    return float("nan")


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

        try:
            from jtop import jtop  # type: ignore

            self._jtop_obj = jtop()
            self._jtop_obj.start()
            self.backend = "jtop"
            self._thread = threading.Thread(target=self._run_jtop, name="hw-jtop", daemon=True)
            self._thread.start()
            return self.backend
        except Exception:
            self._jtop_obj = None

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
            jtop_obj = self._jtop_obj
            self._jtop_obj = None
            try:
                # jtop.close() can block indefinitely if the jtop service is
                # unresponsive (e.g. after a previous long-running connection).
                # Run it in a daemon thread so it cannot hang the process.
                close_thread = threading.Thread(
                    target=jtop_obj.close, daemon=True
                )
                close_thread.start()
                close_thread.join(timeout=5.0)
            except Exception:
                pass

    def get_samples(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(x) for x in self.samples]

    def _append_sample(self, sample: Dict[str, Any]) -> None:
        row = {
            "t_s": float(time.perf_counter() - self._start_t),
            "ram_used_pct": sample.get("ram_used_pct", float("nan")),
            "cpu_pct": sample.get("cpu_pct", float("nan")),
            "gpu_pct": sample.get("gpu_pct", float("nan")),
            "cpu_temp_c": sample.get("cpu_temp_c", float("nan")),
            "gpu_temp_c": sample.get("gpu_temp_c", float("nan")),
            "power_w": sample.get("power_w", float("nan")),
            "backend": self.backend,
        }
        with self._lock:
            self.samples.append(row)

    def _run_jtop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._jtop_obj is None:
                    break
                ok_method = getattr(self._jtop_obj, "ok", None)
                if callable(ok_method) and not ok_method():
                    break
                stats = dict(getattr(self._jtop_obj, "stats", {}))
            except Exception:
                stats = {}

            self._append_sample(self._sample_from_jtop_stats(stats))
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
            self._append_sample(_parse_tegrastats_line(line))

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

    def _sample_from_jtop_stats(self, stats: Dict[str, Any]) -> Dict[str, float]:
        out = {
            "ram_used_pct": float("nan"),
            "cpu_pct": float("nan"),
            "gpu_pct": float("nan"),
            "cpu_temp_c": float("nan"),
            "gpu_temp_c": float("nan"),
            "power_w": float("nan"),
        }
        if not stats:
            return out

        ram_v = stats.get("RAM", None)
        if isinstance(ram_v, dict):
            used = _extract_numeric_from_obj(ram_v.get("used", ram_v.get("use", None)))
            total = _extract_numeric_from_obj(ram_v.get("tot", ram_v.get("total", None)))
            if np.isfinite(used) and np.isfinite(total) and total > 0.0:
                out["ram_used_pct"] = 100.0 * used / total
            else:
                out["ram_used_pct"] = _extract_numeric_from_obj(ram_v)
        elif isinstance(ram_v, (list, tuple)) and len(ram_v) >= 2:
            used = _extract_numeric_from_obj(ram_v[0])
            total = _extract_numeric_from_obj(ram_v[1])
            if np.isfinite(used) and np.isfinite(total) and total > 0.0:
                out["ram_used_pct"] = 100.0 * used / total
        else:
            out["ram_used_pct"] = _extract_numeric_from_obj(ram_v)
        if np.isfinite(out["ram_used_pct"]) and out["ram_used_pct"] <= 1.0:
            out["ram_used_pct"] *= 100.0

        out["cpu_pct"] = _extract_numeric_from_obj(stats.get("CPU", None))
        if not np.isfinite(out["cpu_pct"]):
            cpu_keys = [k for k in stats.keys() if re.fullmatch(r"cpu\d+", str(k).lower())]
            cpu_vals = [_extract_numeric_from_obj(stats[k]) for k in cpu_keys]
            out["cpu_pct"] = _avg_valid(cpu_vals)

        gpu_candidates: List[float] = []
        for k, v in stats.items():
            k_l = _normalize_stat_key(k)
            if "gpu" not in k_l and "gr3d" not in k_l:
                continue
            if any(token in k_l for token in ("temp", "power", "volt", "fan")):
                continue

            gpu_pct = _extract_percent_from_obj(v)
            if np.isfinite(gpu_pct):
                gpu_candidates.append(gpu_pct)
        out["gpu_pct"] = _avg_valid(gpu_candidates)

        cpu_t: List[float] = []
        gpu_t: List[float] = []
        for k, v in stats.items():
            k_l = _normalize_stat_key(k)
            if "temp" in k_l and "cpu" in k_l:
                cpu_t.append(_extract_numeric_from_obj(v))
            if "temp" in k_l and "gpu" in k_l:
                gpu_t.append(_extract_numeric_from_obj(v))
        out["cpu_temp_c"] = _avg_valid(cpu_t)
        out["gpu_temp_c"] = _avg_valid(gpu_t)

        power_candidates: List[float] = []
        power_pref: List[float] = []
        for k, v in stats.items():
            k_l = _normalize_stat_key(k)
            if "power" in k_l or "pom_" in k_l or "vdd_in" in k_l:
                val = _extract_numeric_from_obj(v)
                if np.isfinite(val):
                    power_candidates.append(val)
                    if "5v_in" in k_l or "vdd_in" in k_l or "tot" in k_l:
                        power_pref.append(val)
        raw_power = _avg_valid(power_pref) if power_pref else _avg_valid(power_candidates)
        if np.isfinite(raw_power):
            out["power_w"] = raw_power / 1000.0 if raw_power > 100.0 else raw_power

        return out


def _slugify_name(name: str) -> str:
    import re

    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name).strip())
    s = re.sub(r"-{2,}", "-", s).strip("-._")
    return s or "model"


def pick_profile_out_dir(
    profile_out_arg: Optional[str],
    save_path: Optional[Path],
    model_tag: str,
    yolo_weights_path: Path,
    run_tag: Optional[str] = None,
) -> Path:
    base_root = Path(profile_out_arg).expanduser() if profile_out_arg else (save_path.parent if save_path is not None else Path("runs") / "profiling")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    model_slug = _slugify_name(model_tag)
    yolo_slug = _slugify_name(Path(yolo_weights_path).name)
    run_name = f"{ts}__model_{model_slug}__kpts_{yolo_slug}"
    if run_tag is not None and str(run_tag).strip() != "":
        run_name = f"{run_name}__{_slugify_name(run_tag)}"
    out_dir = base_root / run_name

    suffix = 1
    while out_dir.exists():
        out_dir = base_root / f"{run_name}_{suffix:02d}"
        suffix += 1
    return out_dir


def get_benchmark_duration_s(default_s: float = 600.0) -> float:
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


def assert_benchmark_device_ok(benchmark: bool, device: str) -> None:
    if not bool(benchmark):
        return
    device_str = str(device).strip()
    if not device_str.lower().startswith("cuda"):
        raise ValueError(
            f"Benchmark looping requires CUDA, but resolved runtime device is '{device_str}'. "
            "Use --device cuda on a CUDA-capable machine, or disable --benchmark."
        )


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
    sizes = [cv2.getTextSize(str(s), font, font_scale, thickness)[0] for s in lines]
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
        cv2.putText(frame, str(s), (x0 + pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_gap
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


@dataclass
class BenchmarkRunConfig:
    video_path: Path
    benchmark_name: str

    profile_enabled: bool
    profile_out_dir: Optional[Path]
    profile_duration_s: float

    benchmark_mode: bool
    benchmark_loop_video: bool

    no_display: bool
    display_fps: float
    save_path: Optional[Path]
    draw_conf_thres: float

    warmup_frames: int
    warmup_windows: int

    limit_frames: Optional[int]
    pad_tail: bool

    retain_window_payloads: bool = False
    hw_sample_hz: float = 1.0


@dataclass
class WindowEvalRecord:
    window_id: int
    sampled_start_idx: int
    sampled_end_idx: int
    window_start_raw: int
    window_end_raw: int
    raw_window_stride: int
    sampled_window_stride: int
    window_assembly_ms: float
    temporal_prep_ms: float
    temporal_forward_ms: float
    temporal_total_ms: float
    label_post_ms: float
    prediction: Prediction
    frame_dir: str
    xy_seq: Optional[np.ndarray]
    conf_seq: Optional[np.ndarray]
    image_shape: Tuple[int, int]


@dataclass
class BenchmarkRunResult:
    summary: Dict[str, Any]
    profile_out_dir: Optional[Path]
    per_frame_csv: Optional[Path]
    per_window_csv: Optional[Path]
    summary_json: Optional[Path]
    frame_rows: List[Dict[str, Any]]
    window_rows: List[Dict[str, Any]]
    window_records: List[WindowEvalRecord]
    image_shape: Optional[Tuple[int, int]]


def _resolve_active_window(
    window_by_start: Dict[int, WindowEvalRecord],
    sample_idx: int,
    sampled_stride: int,
) -> Optional[WindowEvalRecord]:
    stride = max(1, int(sampled_stride))
    s = (int(sample_idx) // stride) * stride
    while s >= 0:
        rec = window_by_start.get(int(s))
        if rec is not None:
            return rec
        s -= stride
    return None


def _build_summary(
    frame_rows: List[Dict[str, Any]],
    window_rows: List[Dict[str, Any]],
    hw_rows: List[Dict[str, Any]],
    warmup_frames: int,
    warmup_windows: int,
) -> Dict[str, Any]:
    wup_frames = max(0, int(warmup_frames))
    wup_windows = max(0, int(warmup_windows))

    kept_frames = frame_rows[wup_frames:] if wup_frames < len(frame_rows) else []
    kept_windows = window_rows[wup_windows:] if wup_windows < len(window_rows) else []

    pose_vals = [_safe_float(r.get("pose_infer_ms", np.nan)) for r in kept_frames]
    track_vals = [_safe_float(r.get("track_ms", np.nan)) for r in kept_frames]
    render_vals = [_safe_float(r.get("render_ms", np.nan)) for r in kept_frames]
    writer_vals = [_safe_float(r.get("writer_ms", np.nan)) for r in kept_frames]
    loop_vals = [_safe_float(r.get("total_loop_ms", np.nan)) for r in kept_frames]
    temp_eff_vals = [_safe_float(r.get("temporal_effective_ms", np.nan)) for r in kept_frames]

    prep_vals = [_safe_float(r.get("temporal_prep_ms", np.nan)) for r in kept_windows]
    forward_vals = [_safe_float(r.get("temporal_forward_ms", np.nan)) for r in kept_windows]
    total_vals = [_safe_float(r.get("temporal_total_ms", np.nan)) for r in kept_windows]

    avg_total_loop = _avg_valid(loop_vals)
    avg_fps = float(1000.0 / avg_total_loop) if np.isfinite(avg_total_loop) and avg_total_loop > 0.0 else float("nan")

    summary: Dict[str, Any] = {
        "avg_fps": _json_safe_number(avg_fps),
        "avg_pose_infer_ms_per_frame": _json_safe_number(_avg_valid(pose_vals)),
        "avg_track_ms_per_frame": _json_safe_number(_avg_valid(track_vals)),
        "avg_render_ms_per_frame": _json_safe_number(_avg_valid(render_vals)),
        "avg_writer_ms_per_frame": _json_safe_number(_avg_valid(writer_vals)),
        "avg_total_loop_ms_per_frame": _json_safe_number(avg_total_loop),
        "avg_temporal_prep_ms_per_window": _json_safe_number(_avg_valid(prep_vals)),
        "avg_temporal_forward_ms_per_window": _json_safe_number(_avg_valid(forward_vals)),
        "avg_temporal_total_ms_per_window": _json_safe_number(_avg_valid(total_vals)),
        "avg_temporal_effective_ms_per_frame": _json_safe_number(_avg_valid(temp_eff_vals)),
        "num_frames_processed": int(len(frame_rows)),
        "num_windows_evaluated": int(len(window_rows)),
        "pose_infer_ms_p50": _json_safe_number(_percentile_valid(pose_vals, 50)),
        "pose_infer_ms_p95": _json_safe_number(_percentile_valid(pose_vals, 95)),
        "total_loop_ms_p50": _json_safe_number(_percentile_valid(loop_vals, 50)),
        "total_loop_ms_p95": _json_safe_number(_percentile_valid(loop_vals, 95)),
        "temporal_forward_ms_p50": _json_safe_number(_percentile_valid(forward_vals, 50)),
        "temporal_forward_ms_p95": _json_safe_number(_percentile_valid(forward_vals, 95)),
        "temporal_total_ms_p50": _json_safe_number(_percentile_valid(total_vals, 50)),
        "temporal_total_ms_p95": _json_safe_number(_percentile_valid(total_vals, 95)),
        "warmup_frames_excluded": int(wup_frames),
        "warmup_windows_excluded": int(wup_windows),
        "warmup_applied_to_summary": bool(wup_frames > 0 or wup_windows > 0),
        "frames_used_for_summary": int(len(kept_frames)),
        "windows_used_for_summary": int(len(kept_windows)),
    }

    # Legacy compatibility fields for downstream scripts that still read old keys.
    legacy_pre_vals = [
        _safe_float(r.get("pose_infer_ms", np.nan)) + _safe_float(r.get("track_ms", np.nan))
        for r in kept_frames
    ]
    legacy_inf_vals = [_safe_float(r.get("temporal_effective_ms", np.nan)) for r in kept_frames]
    legacy_post_vals = [
        _safe_float(r.get("render_ms", np.nan)) + _safe_float(r.get("writer_ms", np.nan))
        for r in kept_frames
    ]

    summary["preprocess_ms"] = {
        "mean": _json_safe_number(_avg_valid(legacy_pre_vals)),
        "median": _json_safe_number(_percentile_valid(legacy_pre_vals, 50)),
        "p95": _json_safe_number(_percentile_valid(legacy_pre_vals, 95)),
    }
    summary["inference_ms"] = {
        "mean": _json_safe_number(_avg_valid(legacy_inf_vals)),
        "median": _json_safe_number(_percentile_valid(legacy_inf_vals, 50)),
        "p95": _json_safe_number(_percentile_valid(legacy_inf_vals, 95)),
    }
    summary["postprocess_ms"] = {
        "mean": _json_safe_number(_avg_valid(legacy_post_vals)),
        "median": _json_safe_number(_percentile_valid(legacy_post_vals, 50)),
        "p95": _json_safe_number(_percentile_valid(legacy_post_vals, 95)),
    }
    summary["legacy_metric_semantics"] = (
        "preprocess_ms.mean = pose_infer_ms + track_ms per frame; "
        "inference_ms.mean = amortized temporal_total_ms per raw frame; "
        "postprocess_ms.mean = render_ms + writer_ms per frame."
    )

    ram_vals = [_safe_float(r.get("ram_used_pct", np.nan)) for r in hw_rows]
    cpu_vals = [_safe_float(r.get("cpu_pct", np.nan)) for r in hw_rows]
    gpu_vals = [_safe_float(r.get("gpu_pct", np.nan)) for r in hw_rows]
    cpu_temp_vals = [_safe_float(r.get("cpu_temp_c", np.nan)) for r in hw_rows]
    gpu_temp_vals = [_safe_float(r.get("gpu_temp_c", np.nan)) for r in hw_rows]
    power_vals = [_safe_float(r.get("power_w", np.nan)) for r in hw_rows]

    hw_backend = None
    for row in hw_rows:
        backend = str(row.get("backend") or "").strip()
        if backend:
            hw_backend = backend
            break

    summary["avg_cpu_pct"] = _json_safe_number(_avg_valid(cpu_vals))
    summary["avg_gpu_pct"] = _json_safe_number(_avg_valid(gpu_vals))
    summary["avg_ram_pct"] = _json_safe_number(_avg_valid(ram_vals))
    summary["avg_cpu_temp_c"] = _json_safe_number(_avg_valid(cpu_temp_vals))
    summary["avg_gpu_temp_c"] = _json_safe_number(_avg_valid(gpu_temp_vals))
    summary["avg_power_w"] = _json_safe_number(_avg_valid(power_vals))
    summary["hw_backend"] = hw_backend
    summary["hw_samples_collected"] = int(len(hw_rows))

    return summary


def _build_legacy_time_rows(frame_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in frame_rows:
        total_ms = _safe_float(row.get("total_loop_ms", np.nan))
        fps = float(1000.0 / total_ms) if np.isfinite(total_ms) and total_ms > 0.0 else float("nan")
        out.append(
            {
                "frame_idx": int(_safe_float(row.get("frame_idx_raw", 0), 0.0)),
                "fps": fps,
                "preprocess_ms": _safe_float(row.get("pose_infer_ms", np.nan)) + _safe_float(row.get("track_ms", np.nan)),
                "inference_ms": _safe_float(row.get("temporal_effective_ms", np.nan)),
                "postprocess_ms": _safe_float(row.get("render_ms", np.nan)) + _safe_float(row.get("writer_ms", np.nan)),
                "visualisation_ms": _safe_float(row.get("render_ms", np.nan)),
                "total_ms": total_ms,
            }
        )
    return out


def _assemble_window(
    sampled_xy: List[np.ndarray],
    sampled_cf: List[np.ndarray],
    sampled_raw_idx: List[int],
    sampled_start: int,
    sampled_len: int,
    cap_done: bool,
    pad_tail: bool,
    image_shape: Tuple[int, int],
    video_stem: str,
) -> Optional[Tuple[WindowData, float]]:
    t0 = time.perf_counter()

    start = int(sampled_start)
    T = int(sampled_len)
    total = int(len(sampled_xy))
    if start >= total:
        return None

    end = int(start + T)
    if end <= total:
        xy_seq = np.stack(sampled_xy[start:end], axis=0).astype(np.float32)
        cf_seq = np.stack(sampled_cf[start:end], axis=0).astype(np.float32)
        raw_idx_seq = [int(x) for x in sampled_raw_idx[start:end]]
    else:
        if not cap_done or (not pad_tail):
            return None
        if start >= total:
            return None
        pad_n = int(end - total)
        if pad_n >= T:
            return None

        xy_core = np.stack(sampled_xy[start:total], axis=0).astype(np.float32)
        cf_core = np.stack(sampled_cf[start:total], axis=0).astype(np.float32)
        raw_core = [int(x) for x in sampled_raw_idx[start:total]]
        last_xy = xy_core[-1:, :, :]
        last_cf = cf_core[-1:, :]
        xy_seq = np.concatenate([xy_core, np.repeat(last_xy, pad_n, axis=0)], axis=0)
        cf_seq = np.concatenate([cf_core, np.repeat(last_cf, pad_n, axis=0)], axis=0)
        raw_idx_seq = raw_core + [int(raw_core[-1])] * int(pad_n)

    sampled_end_idx = int(start + T - 1)
    raw_start = int(raw_idx_seq[0])
    raw_end = int(raw_idx_seq[-1])

    window = WindowData(
        xy_seq=xy_seq,
        conf_seq=cf_seq,
        sampled_start_idx=int(start),
        sampled_end_idx=int(sampled_end_idx),
        raw_start_idx=int(raw_start),
        raw_end_idx=int(raw_end),
        image_shape=(int(image_shape[0]), int(image_shape[1])),
        video_stem=str(video_stem),
    )

    assembly_ms = (time.perf_counter() - t0) * 1000.0
    return window, float(assembly_ms)


def run_shared_benchmark(
    config: BenchmarkRunConfig,
    pose_pipeline: PosePipeline,
    classifier: TemporalClassifierAdapter,
) -> BenchmarkRunResult:
    video_path = Path(config.video_path).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(src_fps) or src_fps <= 1e-3:
        src_fps = 30.0

    fps_target = float(config.display_fps) if float(config.display_fps) > 1e-3 else float(src_fps)
    frame_period_s = 1.0 / max(1e-6, float(fps_target))

    sync_cuda_timing = bool(str(getattr(classifier, "device", "")).startswith("cuda") and torch.cuda.is_available())

    writer: Optional[cv2.VideoWriter] = None
    writer_size: Optional[Tuple[int, int]] = None

    frame_rows: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []
    window_records: List[WindowEvalRecord] = []

    image_shape: Optional[Tuple[int, int]] = None
    video_stem = video_path.stem

    sampled_xy: List[np.ndarray] = []
    sampled_cf: List[np.ndarray] = []
    sampled_raw_idx: List[int] = []

    window_by_start: Dict[int, WindowEvalRecord] = {}
    next_window_start = 0
    next_window_id = 0

    raw_frame_idx = 0
    loop_count = 0
    fps_ema: Optional[float] = None
    ema_alpha = 0.1
    run_t0 = time.perf_counter()
    heartbeat_interval_s = 30.0 if bool(config.benchmark_mode) else 0.0
    last_heartbeat_t = run_t0

    user_exit = False
    cap_ended = False
    hw_rows: List[Dict[str, Any]] = []
    hw_sampler: Optional[HardwareSampler] = None

    policy = classifier.window_policy

    if bool(config.profile_enabled) and float(config.hw_sample_hz) > 0.0:
        try:
            candidate_sampler = HardwareSampler(sample_hz=float(config.hw_sample_hz))
            backend = candidate_sampler.start()
            if backend != "none":
                hw_sampler = candidate_sampler
        except Exception as exc:
            print(f"[benchmark][WARN] Could not start hardware sampler: {exc}")

    def stop_due_profile_duration() -> bool:
        if not bool(config.profile_enabled):
            return False
        if float(config.profile_duration_s) <= 0.0:
            return False
        return (time.perf_counter() - run_t0) >= float(config.profile_duration_s)

    def reset_loop_temporal_state() -> None:
        nonlocal sampled_xy, sampled_cf, sampled_raw_idx, window_by_start, next_window_start
        sampled_xy = []
        sampled_cf = []
        sampled_raw_idx = []
        window_by_start = {}
        next_window_start = 0

    def evaluate_ready_windows(cap_done: bool) -> None:
        nonlocal next_window_start, next_window_id

        while True:
            assembled = _assemble_window(
                sampled_xy=sampled_xy,
                sampled_cf=sampled_cf,
                sampled_raw_idx=sampled_raw_idx,
                sampled_start=int(next_window_start),
                sampled_len=int(policy.sampled_window_len),
                cap_done=bool(cap_done),
                pad_tail=bool(config.pad_tail),
                image_shape=(int(image_shape[0]), int(image_shape[1])) if image_shape is not None else (0, 0),
                video_stem=video_stem,
            )
            if assembled is None:
                break

            window_data, assembly_ms = assembled
            prepared_input, prep_metrics = classifier.prepare_window(window_data=window_data, sync_cuda_timing=sync_cuda_timing)
            prediction, infer_metrics = classifier.infer(prepared_input=prepared_input, sync_cuda_timing=sync_cuda_timing)

            temporal_prep_ms = float(_safe_float(prep_metrics.get("temporal_prep_ms", 0.0), 0.0))
            temporal_forward_ms = float(_safe_float(infer_metrics.get("temporal_forward_ms", 0.0), 0.0))
            label_post_ms = float(_safe_float(infer_metrics.get("label_post_ms", 0.0), 0.0))
            temporal_total_ms = float(temporal_prep_ms + temporal_forward_ms)

            frame_dir = f"{video_stem}_s{int(window_data.sampled_start_idx)}_len{int(policy.sampled_window_len)}"

            rec = WindowEvalRecord(
                window_id=int(next_window_id),
                sampled_start_idx=int(window_data.sampled_start_idx),
                sampled_end_idx=int(window_data.sampled_end_idx),
                window_start_raw=int(window_data.raw_start_idx),
                window_end_raw=int(window_data.raw_end_idx),
                raw_window_stride=int(policy.raw_window_stride),
                sampled_window_stride=int(policy.sampled_window_stride),
                window_assembly_ms=float(assembly_ms),
                temporal_prep_ms=float(temporal_prep_ms),
                temporal_forward_ms=float(temporal_forward_ms),
                temporal_total_ms=float(temporal_total_ms),
                label_post_ms=float(label_post_ms),
                prediction=prediction,
                frame_dir=str(frame_dir),
                xy_seq=window_data.xy_seq.copy() if bool(config.retain_window_payloads) else None,
                conf_seq=window_data.conf_seq.copy() if bool(config.retain_window_payloads) else None,
                image_shape=(int(window_data.image_shape[0]), int(window_data.image_shape[1])),
            )
            window_records.append(rec)
            window_by_start[int(window_data.sampled_start_idx)] = rec

            row = {
                "window_id": int(rec.window_id),
                "window_start_raw": int(rec.window_start_raw),
                "window_end_raw": int(rec.window_end_raw),
                "raw_window_stride": int(rec.raw_window_stride),
                "sampled_window_stride": int(rec.sampled_window_stride),
                "window_assembly_ms": float(rec.window_assembly_ms),
                "temporal_prep_ms": float(rec.temporal_prep_ms),
                "temporal_forward_ms": float(rec.temporal_forward_ms),
                "temporal_total_ms": float(rec.temporal_total_ms),
                "label_post_ms": float(rec.label_post_ms),
                "predicted_id": int(rec.prediction.pred_id),
                "predicted_label": str(rec.prediction.pred_label),
                "prediction_confidence": _safe_float(rec.prediction.confidence, float("nan")),
                "frame_dir": rec.frame_dir,
                "sampled_start_idx": int(rec.sampled_start_idx),
                "sampled_end_idx": int(rec.sampled_end_idx),
                "is_warmup_window": bool(int(rec.window_id) < int(max(0, config.warmup_windows))),
            }
            window_rows.append(row)

            next_window_id += 1
            next_window_start += int(policy.sampled_window_stride)

    try:
        if not bool(config.no_display):
            cv2.namedWindow(str(config.benchmark_name), cv2.WINDOW_NORMAL)

        print("[benchmark] loop_start", flush=True)
        while True:
            if stop_due_profile_duration():
                break

            t_loop0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if bool(config.benchmark_loop_video) and (not stop_due_profile_duration()):
                    cap.release()
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        print("[benchmark][WARN] Could not reopen video stream; stopping benchmark loop.")
                        cap_ended = True
                        break
                    loop_count += 1
                    pose_pipeline.reset_tracking_state()
                    reset_loop_temporal_state()
                    continue
                cap_ended = True
                break

            if image_shape is None:
                h0, w0 = frame.shape[:2]
                image_shape = (int(h0), int(w0))

            if int(raw_frame_idx) == 0:
                print("[benchmark] first_frame_start", flush=True)
            pose_out = pose_pipeline.process_frame(frame_bgr=frame, raw_frame_idx=int(raw_frame_idx), sync_cuda_timing=sync_cuda_timing)
            if pose_out.sampled:
                sampled_xy.append(pose_out.keypoints_xy.copy())
                sampled_cf.append(pose_out.keypoints_conf.copy())
                sampled_raw_idx.append(int(raw_frame_idx))

            evaluate_ready_windows(cap_done=False)

            sample_idx = int(raw_frame_idx) // int(max(1, policy.frame_step))
            active_window = _resolve_active_window(
                window_by_start=window_by_start,
                sample_idx=int(sample_idx),
                sampled_stride=int(policy.sampled_window_stride),
            )

            active_window_id: Optional[int] = None
            active_label = "..."
            active_conf = float("nan")
            temporal_effective_ms = 0.0

            if active_window is not None:
                active_window_id = int(active_window.window_id)
                active_label = str(active_window.prediction.pred_label)
                active_conf = _safe_float(active_window.prediction.confidence, float("nan"))

                charge_start = int(active_window.window_start_raw)
                charge_end = int(charge_start + int(policy.raw_window_stride) - 1)
                if int(raw_frame_idx) >= charge_start and int(raw_frame_idx) <= charge_end:
                    temporal_effective_ms = float(active_window.temporal_total_ms) / float(max(1, int(policy.raw_window_stride)))

            render_ms = 0.0
            writer_ms = 0.0
            display_wait_ms = 0.0

            frame_to_show = frame
            if (not bool(config.no_display)) or (writer is not None) or (config.save_path is not None):
                t_render0 = time.perf_counter()
                frame_to_show = frame.copy()
                frame_to_show = draw_pose(
                    frame=frame_to_show,
                    xy=pose_out.keypoints_xy,
                    conf=pose_out.keypoints_conf,
                    conf_thres=float(config.draw_conf_thres),
                )

                fps_for_hud = float(fps_ema) if fps_ema is not None else float(fps_target)
                hud = [
                    f"frame {int(raw_frame_idx) + 1}",
                    f"fps: {float(fps_for_hud):.1f}",
                    f"pred: {active_label}" + (f" ({float(active_conf):.2f})" if np.isfinite(active_conf) else ""),
                    (
                        f"raw T/stride={int(policy.raw_window_len)}/{int(policy.raw_window_stride)} "
                        f"sampled T/stride={int(policy.sampled_window_len)}/{int(policy.sampled_window_stride)} "
                        f"k={int(policy.frame_step)}"
                    ),
                ]
                frame_to_show = draw_hud(frame_to_show, hud)
                render_ms = (time.perf_counter() - t_render0) * 1000.0

            if writer is None and config.save_path is not None and image_shape is not None:
                writer_size = (int(image_shape[1]), int(image_shape[0]))
                writer = open_video_writer(save_path=Path(config.save_path), fps=float(src_fps), frame_size=writer_size)

            if writer is not None:
                t_writer0 = time.perf_counter()
                if writer_size is None:
                    writer_size = (int(frame_to_show.shape[1]), int(frame_to_show.shape[0]))
                out_w, out_h = int(writer_size[0]), int(writer_size[1])
                frame_write = frame_to_show
                if int(frame_to_show.shape[1]) != out_w or int(frame_to_show.shape[0]) != out_h:
                    frame_write = cv2.resize(frame_to_show, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                writer.write(frame_write)
                writer_ms = (time.perf_counter() - t_writer0) * 1000.0

            key = -1
            if not bool(config.no_display):
                t_display0 = time.perf_counter()
                cv2.imshow(str(config.benchmark_name), frame_to_show)
                elapsed_s = time.perf_counter() - t_loop0
                wait_s = float(frame_period_s) - float(elapsed_s)
                wait_ms = max(1, int(round(1000.0 * wait_s))) if wait_s > 0.0 else 1
                key = cv2.waitKey(int(wait_ms)) & 0xFF
                display_wait_ms = (time.perf_counter() - t_display0) * 1000.0

            total_loop_ms = (time.perf_counter() - t_loop0) * 1000.0
            inst_fps = 1000.0 / max(1e-6, total_loop_ms)
            fps_ema = inst_fps if fps_ema is None else (1.0 - ema_alpha) * float(fps_ema) + ema_alpha * inst_fps

            if int(raw_frame_idx) == 0:
                print(
                    f"[benchmark] first_frame_done pose_ms={float(pose_out.pose_infer_ms):.1f} "
                    f"loop_ms={float(total_loop_ms):.1f}",
                    flush=True,
                )

            now = time.perf_counter()
            if heartbeat_interval_s > 0.0 and (now - last_heartbeat_t) >= heartbeat_interval_s:
                fps_text = f"{float(fps_ema):.2f}" if fps_ema is not None else "nan"
                print(
                    f"[benchmark] heartbeat elapsed_s={float(now - run_t0):.1f} "
                    f"frames={int(raw_frame_idx) + 1} windows={len(window_rows)} fps_ema={fps_text}",
                    flush=True,
                )
                last_heartbeat_t = now

            frame_rows.append(
                {
                    "frame_idx_raw": int(raw_frame_idx),
                    "pose_infer_ms": float(pose_out.pose_infer_ms),
                    "track_ms": float(pose_out.track_ms),
                    "render_ms": float(render_ms),
                    "writer_ms": float(writer_ms),
                    "total_loop_ms": float(total_loop_ms),
                    "active_window_id": int(active_window_id) if active_window_id is not None else None,
                    "temporal_effective_ms": float(temporal_effective_ms),
                    "predicted_label": str(active_label) if active_window is not None else None,
                    "prediction_confidence": _safe_float(active_conf, float("nan")),
                    "display_wait_ms": float(display_wait_ms),
                    "is_warmup_frame": bool(int(raw_frame_idx) < int(max(0, config.warmup_frames))),
                }
            )

            raw_frame_idx += 1

            if (config.limit_frames is not None) and (not bool(config.benchmark_loop_video)) and int(raw_frame_idx) >= int(config.limit_frames):
                cap_ended = True
                break

            if key in (ord("q"), 27):
                user_exit = True
                break

        if cap_ended and image_shape is not None:
            evaluate_ready_windows(cap_done=True)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not bool(config.no_display):
            cv2.destroyAllWindows()
        if hw_sampler is not None:
            hw_sampler.stop()
            hw_rows = hw_sampler.get_samples()

    summary = _build_summary(
        frame_rows=frame_rows,
        window_rows=window_rows,
        hw_rows=hw_rows,
        warmup_frames=int(config.warmup_frames),
        warmup_windows=int(config.warmup_windows),
    )

    if bool(config.benchmark_loop_video):
        summary["benchmark_loop_count"] = int(loop_count)
        summary["benchmark_duration_s_requested"] = _json_safe_number(config.profile_duration_s)
    summary["benchmark_mode"] = bool(config.benchmark_mode)
    summary["user_exit"] = bool(user_exit)

    per_frame_csv = None
    per_window_csv = None
    per_hw_csv = None
    summary_json = None

    if bool(config.profile_enabled) and config.profile_out_dir is not None:
        out_dir = Path(config.profile_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        per_frame_csv = out_dir / "frame_metrics.csv"
        per_window_csv = out_dir / "window_metrics.csv"
        per_hw_csv = out_dir / "hw_metrics.csv"
        summary_json = out_dir / "summary.json"

        frame_fields = [
            "frame_idx_raw",
            "pose_infer_ms",
            "track_ms",
            "render_ms",
            "writer_ms",
            "total_loop_ms",
            "active_window_id",
            "temporal_effective_ms",
            "predicted_label",
            "prediction_confidence",
            "display_wait_ms",
            "is_warmup_frame",
        ]
        window_fields = [
            "window_id",
            "window_start_raw",
            "window_end_raw",
            "raw_window_stride",
            "sampled_window_stride",
            "window_assembly_ms",
            "temporal_prep_ms",
            "temporal_forward_ms",
            "temporal_total_ms",
            "predicted_label",
            "prediction_confidence",
            "predicted_id",
            "label_post_ms",
            "frame_dir",
            "sampled_start_idx",
            "sampled_end_idx",
            "is_warmup_window",
        ]

        _write_csv(per_frame_csv, frame_rows, frame_fields)
        _write_csv(per_window_csv, window_rows, window_fields)
        _write_csv(
            per_hw_csv,
            hw_rows,
            ["t_s", "ram_used_pct", "cpu_pct", "gpu_pct", "cpu_temp_c", "gpu_temp_c", "power_w", "backend"],
        )

        legacy_time_rows = _build_legacy_time_rows(frame_rows)
        _write_csv(
            out_dir / "time_metrics.csv",
            legacy_time_rows,
            ["frame_idx", "fps", "preprocess_ms", "inference_ms", "postprocess_ms", "visualisation_ms", "total_ms"],
        )

        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return BenchmarkRunResult(
        summary=summary,
        profile_out_dir=Path(config.profile_out_dir) if config.profile_out_dir is not None else None,
        per_frame_csv=per_frame_csv,
        per_window_csv=per_window_csv,
        summary_json=summary_json,
        frame_rows=frame_rows,
        window_rows=window_rows,
        window_records=window_records,
        image_shape=image_shape,
    )
