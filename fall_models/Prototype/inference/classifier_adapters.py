from __future__ import annotations

import abc
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import dataset_helpers.dataset as ds

from models.gcn.simple_gcn import GCNBaseline
from models.gru.simple_gru import GRUBaseline
from models.lstm.simple_lstm import LSTMBaseline
from models.mlp.simple_mlp import MLPBaseline
from models.complex_models import PaperLSTMClassifier, PaperSTGCNClassifier
from models.rf.train_rf import windows_to_sklearn_features
from models.stgcn.simple_stgcn import STGCNBaseline
from models.tcn.simple_tcn import TCNBaseline

try:
    from models.cnnlstm.cnn_lstm import CNNLSTMTwoHead
except Exception:
    CNNLSTMTwoHead = None


K = 17

KNOWN_ARCHES = ["tcn", "lstm", "paper_lstm", "gru", "gcn", "mlp", "stgcn", "paper_stgcn", "cnnlstm", "rf"]

FALL_MERGED_CLASS_NAMES = [
    "Fall",
    "Walking",
    "Standing",
    "Sitting",
    "Picking up an object",
    "Jumping",
    "Laying",
]

MB_CLASS_NAMES_DEFAULT = [
    "Falling forward using hands",
    "Falling forward using knees",
    "Falling backwards",
    "Falling sideward",
    "Falling sitting in an empty chair",
    "Walking",
    "Standing",
    "Sitting",
    "Picking up an object",
    "Jumping",
    "Laying",
]

MB_CLASS_NAMES_MERGED_DEFAULT = [
    "Fall",
    "Walking",
    "Standing",
    "Sitting",
    "Picking up an object",
    "Jumping",
    "Laying",
]

FALL_CLASS_IDS_DEFAULT = [0, 1, 2, 3, 4]


def _maybe_cuda_sync(sync_cuda: bool) -> None:
    if bool(sync_cuda):
        torch.cuda.synchronize()


def _timed_stage(sync_cuda: bool, fn):
    _maybe_cuda_sync(sync_cuda)
    t0 = time.perf_counter()
    out = fn()
    _maybe_cuda_sync(sync_cuda)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return out, float(dt_ms)


def ceil_div_pos(a: int, b: int) -> int:
    a_i = int(a)
    b_i = int(b)
    if a_i <= 0:
        raise ValueError(f"Expected positive integer, got {a_i}.")
    if b_i <= 0:
        raise ValueError(f"Expected positive divisor, got {b_i}.")
    return (a_i + b_i - 1) // b_i


def pick_device(device: Optional[str]) -> str:
    if not device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    d = str(device).lower().strip()
    if d.startswith("cuda") and (not torch.cuda.is_available()):
        return "cpu"
    return str(device)


@dataclass
class WindowPolicy:
    raw_window_len: int
    raw_window_stride: int
    sampled_window_len: int
    sampled_window_stride: int
    frame_step: int
    pad_tail: bool = False


@dataclass
class WindowData:
    xy_seq: np.ndarray
    conf_seq: np.ndarray
    sampled_start_idx: int
    sampled_end_idx: int
    raw_start_idx: int
    raw_end_idx: int
    image_shape: Tuple[int, int]
    video_stem: str


@dataclass
class Prediction:
    pred_id: int
    pred_label: str
    confidence: Optional[float]
    extra: Dict[str, Any]


class TemporalClassifierAdapter(abc.ABC):
    name: str
    window_policy: WindowPolicy
    class_names: List[str]

    @abc.abstractmethod
    def prepare_window(self, window_data: WindowData, sync_cuda_timing: bool) -> Tuple[Any, Dict[str, float]]:
        pass

    @abc.abstractmethod
    def infer(self, prepared_input: Any, sync_cuda_timing: bool) -> Tuple[Prediction, Dict[str, float]]:
        pass


# -----------------------------------------------------------------------------
# Generic temporal models (CNN-LSTM/ST-GCN/etc.)
# -----------------------------------------------------------------------------


def infer_arch_from_path(p: Path) -> Optional[str]:
    tokens = [p.name.lower(), p.stem.lower()] + [x.lower() for x in p.parts]
    for arch in sorted(KNOWN_ARCHES, key=len, reverse=True):
        if any(tok == arch for tok in tokens):
            return arch
        if any(tok.startswith(arch + "_") for tok in tokens):
            return arch
        if arch == "rf":
            if any(tok.startswith("rf") for tok in tokens):
                return arch
            continue
        if any(arch in tok for tok in tokens):
            return arch
    return None


def resolve_ckpt_and_arch(model_arg: str, arch_arg: Optional[str]) -> Tuple[Path, str]:
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
                raise ValueError("Inferred --arch rf for tensor checkpoint. RF checkpoints should be .pkl/.pickle.")
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
    try:
        with Path(ckpt_path).open("rb") as f:
            obj = pickle.load(f)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Failed to load RF checkpoint. Install scikit-learn with: pip install scikit-learn\n"
            f"Import error: {e}"
        ) from e

    if not isinstance(obj, dict) or "model" not in obj:
        raise TypeError(f"Unsupported RF checkpoint format: expected dict with key 'model'. Got: {type(obj)}")
    return obj


def load_class_names(num_classes: int, meta: Dict[str, object], labels_file: Optional[str]) -> List[str]:
    names: List[str] = []
    if labels_file:
        p = Path(labels_file).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"--labels-file not found: {p}")
        names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

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


def feature_layout(use_conf: bool, add_vel: bool, add_acc: bool) -> Dict[str, Optional[object]]:
    idx = 2
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


def expected_in_features(
    use_conf: bool,
    add_vel: bool,
    add_acc: bool,
    add_global: bool,
    add_mask: bool,
) -> int:
    if add_acc and (not add_vel):
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


def make_window_features(
    xy_seq: np.ndarray,
    conf_seq: np.ndarray,
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

    parts: List[np.ndarray] = [xy_used]
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
        g = ds._global_features(xy_used, conf_filled)
        parts.append(np.repeat(g[:, None, :], repeats=K, axis=1))

    Xf = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

    frac_valid = (conf_filled > float(conf_thres)).mean(axis=1)
    valid = frac_valid >= float(min_valid_frac)

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

    return seq.reshape(T, int(seq.shape[1]) * int(seq.shape[2])).astype(np.float32, copy=False)


def _rf_predict_proba_aligned(clf, X_feat: np.ndarray, num_classes: int) -> np.ndarray:
    if not hasattr(clf, "predict_proba"):
        raise TypeError("RF checkpoint model does not implement predict_proba().")

    raw = clf.predict_proba(X_feat)
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"Expected RF predict_proba output (N,C). Got shape {getattr(raw, 'shape', None)}")

    num_classes = int(num_classes)
    out = np.zeros((int(raw.shape[0]), int(num_classes)), dtype=np.float32)

    classes = getattr(clf, "classes_", None)
    if classes is None:
        if int(raw.shape[1]) != int(num_classes):
            raise ValueError(f"RF predict_proba returned C={int(raw.shape[1])}, expected num_classes={int(num_classes)}")
        return raw.astype(np.float32, copy=False)

    classes_np = np.asarray(classes).astype(np.int64, copy=False).reshape(-1)
    for j, cls_id in enumerate(classes_np.tolist()):
        if 0 <= int(cls_id) < int(num_classes) and j < int(raw.shape[1]):
            out[:, int(cls_id)] = raw[:, int(j)]

    return out


@torch.no_grad()
def infer_one_window(
    model: nn.Module,
    window_feat: np.ndarray,
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
        prob = torch.cat([prob[:, :5].sum(dim=1, keepdim=True), prob[:, 5:]], dim=1)

    pconf, pred = torch.max(prob, dim=-1)
    p_fall = None
    if fall_logit is not None:
        p_fall = float(torch.sigmoid(fall_logit.view(-1))[0].item())

    return int(pred.item()), float(pconf.item()), p_fall


def build_temporal_model(
    arch: str,
    in_features: int,
    num_classes: int,
    device: str,
    T_used: int,
    node_features: Optional[int],
) -> nn.Module:
    arch = str(arch).lower().strip()
    if arch == "tcn":
        model = TCNBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            kernel_size=3,
            dropout=0.1,
        )
    elif arch == "lstm":
        model = LSTMBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
            pool="last",
        )
    elif arch == "paper_lstm":
        model = PaperLSTMClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=80,
            num_layers=10,
            dropout=0.2,
            pool="last",
        )
    elif arch == "gru":
        model = GRUBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
            pool="last",
        )
    elif arch == "gcn":
        if node_features is None:
            raise ValueError("GCN requires node_features (in_features must be divisible by 17).")
        model = GCNBaseline(
            num_nodes=17,
            node_features=int(node_features),
            num_classes=num_classes,
            hidden_size=64,
            dropout=0.1,
        )
    elif arch == "mlp":
        model = MLPBaseline(
            T=int(T_used),
            in_features=in_features,
            num_classes=num_classes,
            hidden_sizes=(256, 128),
            dropout=0.2,
        )
    elif arch == "stgcn":
        if node_features is None:
            raise ValueError("STGCN requires node_features (in_features must be divisible by 17).")
        model = STGCNBaseline(
            num_nodes=17,
            node_features=int(node_features),
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            t_kernel=9,
            dropout=0.1,
        )
    elif arch == "paper_stgcn":
        if node_features is None:
            raise ValueError("paper_stgcn requires node_features (in_features must be divisible by 17).")
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

    return model.to(device)


@dataclass
class GenericAdapterConfig:
    model_arg: str
    arch_arg: Optional[str]
    device: str
    frame_step: int

    half: bool = False
    T_override: int = 0
    stride_override: int = 0

    normalize_mode_override: Optional[str] = None
    missing_mode_override: Optional[str] = None
    interp_mode_override: Optional[str] = None
    interp_group_override: int = 0
    rp_center_mode_override: Optional[str] = None
    rp_img_w_override: int = 0
    rp_img_h_override: int = 0

    labels_file: Optional[str] = None


class GenericTemporalAdapter(TemporalClassifierAdapter):
    def __init__(self, config: GenericAdapterConfig):
        self.device = pick_device(config.device)
        self.use_half_temporal = bool(config.half) and str(self.device).startswith("cuda")

        ckpt_path, arch = resolve_ckpt_and_arch(config.model_arg, config.arch_arg)
        self.checkpoint_path = ckpt_path
        self.arch = str(arch).lower().strip()
        self.name = self.arch

        is_rf = self.arch == "rf" or ckpt_path.suffix.lower() in {".pkl", ".pickle"}
        self._is_rf = bool(is_rf)

        rf_model = None
        rf_feature_mode = "flatten"
        rf_feature_dim = None

        if is_rf:
            meta = load_rf_checkpoint(ckpt_path)
            rf_model = meta.get("model", None)
            rf_feature_mode = str(meta.get("feature_mode", "flatten"))
            rf_feature_dim = meta.get("feature_dim", None)
            state = None
        else:
            state, meta = load_checkpoint(ckpt_path)
            state = clean_state_dict(state)

        T_raw = int(config.T_override) if int(config.T_override) > 0 else int(meta.get("T", meta.get("T_used", 64)) or 64)
        stride_raw = int(config.stride_override) if int(config.stride_override) > 0 else int(meta.get("stride", 16) or 16)
        T_raw = max(1, int(T_raw))
        stride_raw = max(1, int(stride_raw))

        frame_step = max(1, int(config.frame_step))
        T_final = max(1, int(ceil_div_pos(int(T_raw), frame_step)))
        stride_final = max(1, int(ceil_div_pos(int(stride_raw), frame_step)))

        self.window_policy = WindowPolicy(
            raw_window_len=int(T_raw),
            raw_window_stride=int(stride_raw),
            sampled_window_len=int(T_final),
            sampled_window_stride=int(stride_final),
            frame_step=int(frame_step),
            pad_tail=False,
        )

        self.use_conf = bool(meta.get("use_conf", True))
        self.normalize = bool(meta.get("normalize", True))
        self.normalize_mode = str(config.normalize_mode_override) if config.normalize_mode_override else str(meta.get("normalize_mode") or "center_scale")
        self.add_vel = bool(meta.get("add_vel", True))
        self.add_acc = bool(meta.get("add_acc", True))
        self.add_global = bool(meta.get("add_global", True))
        self.add_mask = bool(meta.get("add_mask_channel", True))
        self.conf_thres = float(meta.get("conf_thres", 0.2))
        self.max_interp_gap = int(meta.get("max_interp_gap", 5))
        self.missing_mode = str(config.missing_mode_override) if config.missing_mode_override else str(meta.get("missing_mode") or "conf_thres")
        self.interp_mode = str(config.interp_mode_override) if config.interp_mode_override else str(meta.get("interp_mode") or "short_gap_hold")
        self.interp_group = int(config.interp_group_override) if int(config.interp_group_override) > 0 else int(meta.get("interp_group", 100) or 100)
        self.rp_center_mode = str(config.rp_center_mode_override) if config.rp_center_mode_override else str(meta.get("rp_center_mode") or "auto")

        self.rp_img_w: Optional[int] = None
        self.rp_img_h: Optional[int] = None
        if int(config.rp_img_w_override) > 0:
            self.rp_img_w = int(config.rp_img_w_override)
        elif meta.get("rp_img_w", None) is not None:
            self.rp_img_w = int(meta.get("rp_img_w"))

        if int(config.rp_img_h_override) > 0:
            self.rp_img_h = int(config.rp_img_h_override)
        elif meta.get("rp_img_h", None) is not None:
            self.rp_img_h = int(meta.get("rp_img_h"))

        self.min_valid_frac = float(meta.get("min_valid_frac", 0.3))

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

        self.merge_fall_11_to_7 = int(num_classes) == 11
        display_num_classes = 7 if self.merge_fall_11_to_7 else int(num_classes)
        self.class_names = load_class_names(num_classes=display_num_classes, meta=meta, labels_file=config.labels_file)

        in_features = expected_in_features(
            use_conf=self.use_conf,
            add_vel=self.add_vel,
            add_acc=self.add_acc,
            add_global=self.add_global,
            add_mask=self.add_mask,
        )
        if in_features_meta > 0 and int(in_features) != int(in_features_meta):
            raise ValueError(f"Feature mismatch: expected in_features={in_features}, ckpt expects {in_features_meta}")

        self._rf_model = rf_model
        self._rf_feature_mode = rf_feature_mode
        self._rf_feature_dim = rf_feature_dim
        self._display_num_classes = int(display_num_classes)

        if self._is_rf:
            if self._rf_model is None:
                raise ValueError("RF checkpoint missing 'model'.")
            self._model = None
        else:
            node_features_meta = meta.get("node_features", None)
            if node_features_meta is None:
                nf = int(in_features // K)
                node_features_meta = nf if nf * K == int(in_features) else None

            model = build_temporal_model(
                arch=self.arch,
                in_features=int(in_features),
                num_classes=int(num_classes),
                device=self.device,
                T_used=int(T_final),
                node_features=int(node_features_meta) if node_features_meta is not None else None,
            )
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print("[WARN] missing keys:", missing[:8], "..." if len(missing) > 8 else "")
            if unexpected:
                print("[WARN] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
            if self.use_half_temporal:
                model.half()
            model.eval()
            self._model = model

    def prepare_window(self, window_data: WindowData, sync_cuda_timing: bool) -> Tuple[Any, Dict[str, float]]:
        def _prep_fn() -> Any:
            window_feat = make_window_features(
                xy_seq=window_data.xy_seq,
                conf_seq=window_data.conf_seq,
                T=int(self.window_policy.sampled_window_len),
                use_conf=self.use_conf,
                normalize=self.normalize,
                normalize_mode=self.normalize_mode,
                add_vel=self.add_vel,
                add_acc=self.add_acc,
                add_global=self.add_global,
                add_mask=self.add_mask,
                conf_thres=self.conf_thres,
                max_interp_gap=self.max_interp_gap,
                missing_mode=self.missing_mode,
                interp_mode=self.interp_mode,
                interp_group=int(self.interp_group),
                rp_center_mode=self.rp_center_mode,
                rp_img_w=self.rp_img_w,
                rp_img_h=self.rp_img_h,
                min_valid_frac=self.min_valid_frac,
            )
            if self._is_rf:
                X_feat = windows_to_sklearn_features(window_feat[None, ...], mode=str(self._rf_feature_mode))
                if self._rf_feature_dim is not None and int(self._rf_feature_dim) > 0 and int(X_feat.shape[1]) != int(self._rf_feature_dim):
                    raise ValueError(
                        f"RF feature_dim mismatch: extracted={int(X_feat.shape[1])}, ckpt feature_dim={int(self._rf_feature_dim)} "
                        f"(mode={str(self._rf_feature_mode)})"
                    )
                return X_feat
            return window_feat

        prepared, prep_ms = _timed_stage(sync_cuda=False, fn=_prep_fn)
        return prepared, {"temporal_prep_ms": float(prep_ms)}

    def infer(self, prepared_input: Any, sync_cuda_timing: bool) -> Tuple[Prediction, Dict[str, float]]:
        if self._is_rf:
            def _infer_rf() -> np.ndarray:
                assert self._rf_model is not None
                probs = _rf_predict_proba_aligned(self._rf_model, prepared_input, num_classes=int(self._display_num_classes))[0]
                return probs

            probs, forward_ms = _timed_stage(sync_cuda=False, fn=_infer_rf)
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id]) if probs.size > 0 else 0.0
            p_fall = float(probs[0]) if probs.size > 0 else None
            pred_label = self.class_names[pred_id] if 0 <= pred_id < len(self.class_names) else str(pred_id)
            pred = Prediction(
                pred_id=int(pred_id),
                pred_label=str(pred_label),
                confidence=float(confidence),
                extra={"p_fall": p_fall, "probs": probs.tolist()},
            )
            return pred, {"temporal_forward_ms": float(forward_ms), "label_post_ms": 0.0}

        if self._model is None:
            raise RuntimeError("Temporal model is not initialized.")

        def _forward_fn() -> Tuple[int, float, Optional[float]]:
            return infer_one_window(
                model=self._model,
                window_feat=prepared_input,
                device=self.device,
                use_half=self.use_half_temporal,
                merge_fall_11_to_7=self.merge_fall_11_to_7,
            )

        (pred_id, confidence, p_fall), forward_ms = _timed_stage(sync_cuda=sync_cuda_timing, fn=_forward_fn)
        pred_label = self.class_names[pred_id] if 0 <= pred_id < len(self.class_names) else str(pred_id)
        pred = Prediction(
            pred_id=int(pred_id),
            pred_label=str(pred_label),
            confidence=float(confidence),
            extra={"p_fall": p_fall},
        )
        return pred, {"temporal_forward_ms": float(forward_ms), "label_post_ms": 0.0}


# -----------------------------------------------------------------------------
# MotionBERT
# -----------------------------------------------------------------------------


def resolve_motionbert_root(repo_root: Path) -> Path:
    mb_root = repo_root / "models" / "MotionBERT"
    if mb_root.exists():
        return mb_root

    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        cand = parent / "models" / "MotionBERT"
        if cand.exists():
            return cand

    raise FileNotFoundError(f"MotionBERT root not found at: {mb_root.as_posix()}")


def resolve_motionbert_path(path: str, *, repo_root: Path, mb_root: Path, desc: str) -> Path:
    p = Path(path).expanduser()
    if p.exists():
        return p

    repo_rel = (repo_root / path).expanduser()
    if repo_rel.exists():
        return repo_rel

    mb_rel = (mb_root / path).expanduser()
    if mb_rel.exists():
        return mb_rel

    raise FileNotFoundError(f"{desc} not found: {path}")


def resolve_motionbert_checkpoint_path(ckpt: str, *, repo_root: Path, mb_root: Path) -> Path:
    p = resolve_motionbert_path(ckpt, repo_root=repo_root, mb_root=mb_root, desc="Checkpoint")
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


def _clean_state_dict_for_model(state: dict, model: nn.Module) -> dict:
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


def infer_fall_indices(class_names: Sequence[str]) -> List[int]:
    out = []
    for i, n in enumerate(class_names):
        s = str(n).lower()
        if s.startswith("fall") or "falling" in s:
            out.append(i)
    return out


def load_labels_file(path: Optional[str], *, repo_root: Path, mb_root: Path) -> Optional[List[str]]:
    if not path:
        return None
    p = resolve_motionbert_path(path, repo_root=repo_root, mb_root=mb_root, desc="Labels file")
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


def interpolate_missing_joints_inplace(kxy: np.ndarray, ksc: np.ndarray, missing_conf_thres: float = 0.0) -> None:
    T = int(kxy.shape[0])
    V = int(kxy.shape[1])
    if V != 17 or int(kxy.shape[2]) != 2:
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


@dataclass
class MotionBERTAdapterConfig:
    model_arg: str
    config_arg: str
    device: str
    frame_step: int

    win_len_raw: Optional[int] = None
    win_step_raw: int = 16
    labels_file: Optional[str] = None
    no_merge_fall: bool = False
    missing_conf_thres: float = 0.0

    repo_root: Path = _REPO_ROOT


class MotionBERTAdapter(TemporalClassifierAdapter):
    def __init__(self, config: MotionBERTAdapterConfig):
        self.device = pick_device(config.device)
        self.name = "motionbert"
        self.missing_conf_thres = float(config.missing_conf_thres)

        repo_root = Path(config.repo_root).resolve()
        mb_root = resolve_motionbert_root(repo_root)
        self._mb_root = mb_root
        mb_root_str = str(mb_root)
        if mb_root_str not in sys.path:
            sys.path.insert(0, mb_root_str)

        from lib.data.dataset_action import coco2h36m, human_tracking, make_cam
        from lib.model.model_action import ActionNet
        from lib.utils.learning import load_backbone
        from lib.utils.tools import get_config
        from lib.utils.utils_data import crop_scale, resample

        self._mb_make_cam = make_cam
        self._mb_human_tracking = human_tracking
        self._mb_coco2h36m = coco2h36m
        self._mb_crop_scale = crop_scale
        self._mb_resample = resample

        ckpt_path = resolve_motionbert_checkpoint_path(config.model_arg, repo_root=repo_root, mb_root=mb_root)
        cfg_path = resolve_motionbert_path(config.config_arg, repo_root=repo_root, mb_root=mb_root, desc="Config")
        self.checkpoint_path = ckpt_path
        self.config_path = cfg_path

        cfg = get_config(str(cfg_path))
        self.cfg = cfg

        clip_len_raw = int(config.win_len_raw) if config.win_len_raw is not None else int(getattr(cfg, "clip_len", 64))
        win_step_raw = max(1, int(config.win_step_raw))
        frame_step = max(1, int(config.frame_step))

        clip_len = max(1, int(ceil_div_pos(int(clip_len_raw), int(frame_step))))
        win_step = max(1, int(ceil_div_pos(int(win_step_raw), int(frame_step))))

        self.window_policy = WindowPolicy(
            raw_window_len=int(clip_len_raw),
            raw_window_stride=int(win_step_raw),
            sampled_window_len=int(clip_len),
            sampled_window_stride=int(win_step),
            frame_step=int(frame_step),
            pad_tail=False,
        )

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

        use_dp = str(self.device).startswith("cuda") and torch.cuda.device_count() > 1
        if use_dp:
            model = nn.DataParallel(model)
        model = model.to(self.device)

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        state = _clean_state_dict_for_model(state, model)
        model.load_state_dict(state, strict=True)
        model.eval()
        self._model = model

        self.num_classes = int(getattr(cfg, "action_classes", 11))
        self.scale_range = getattr(cfg, "scale_range_test", None)

        labels_file_names = load_labels_file(config.labels_file, repo_root=repo_root, mb_root=mb_root)
        unmerged_len_expected = len(MB_CLASS_NAMES_DEFAULT)
        merged_len_expected = len(MB_CLASS_NAMES_MERGED_DEFAULT)

        self.merge_fall = (not bool(config.no_merge_fall)) and (self.num_classes == unmerged_len_expected)
        if labels_file_names is not None:
            if self.merge_fall and len(labels_file_names) == merged_len_expected:
                class_names_out = list(labels_file_names)
            else:
                base_names = pad_or_trim(list(labels_file_names), self.num_classes)
                if self.merge_fall:
                    class_names_out = ["Fall"] + [base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
                else:
                    class_names_out = base_names
        else:
            if self.num_classes == merged_len_expected:
                class_names_out = list(MB_CLASS_NAMES_MERGED_DEFAULT)
            else:
                base_names = pad_or_trim(list(MB_CLASS_NAMES_DEFAULT), self.num_classes)
                if self.merge_fall:
                    class_names_out = ["Fall"] + [base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
                else:
                    class_names_out = base_names

        self.class_names = class_names_out
        self.fall_idx = infer_fall_indices(self.class_names)
        self._unmerged_len_expected = int(unmerged_len_expected)
        self._merged_len_expected = int(merged_len_expected)

    def _build_motion_input(self, window_data: WindowData) -> torch.Tensor:
        kxy = np.asarray(window_data.xy_seq, dtype=np.float32).copy()
        ksc = np.asarray(window_data.conf_seq, dtype=np.float32).copy()

        nonfinite_xy = ~np.isfinite(kxy)
        nonfinite_sc = ~np.isfinite(ksc)
        if nonfinite_xy.any() or nonfinite_sc.any():
            kxy[nonfinite_xy] = 0.0
            ksc[nonfinite_sc] = 0.0
            nonfinite_joint = nonfinite_xy.any(axis=2) | nonfinite_sc
            ksc[nonfinite_joint] = 0.0

        if (ksc < 0).any() or (ksc > 1).any():
            ksc = np.clip(ksc, 0.0, 1.0)

        interpolate_missing_joints_inplace(kxy, ksc, missing_conf_thres=self.missing_conf_thres)

        keypoint = kxy[None, ...].astype(np.float32)
        keypoint_score = ksc[None, ...].astype(np.float32)

        resample_id = self._mb_resample(
            ori_len=int(keypoint.shape[1]),
            target_len=int(self.window_policy.sampled_window_len),
            randomness=False,
        )
        motion_cam = self._mb_make_cam(x=keypoint, img_shape=window_data.image_shape)
        motion_cam = self._mb_human_tracking(motion_cam)
        motion_cam = self._mb_coco2h36m(motion_cam)
        motion_conf = keypoint_score[..., None]
        motion = np.concatenate((motion_cam[:, resample_id], motion_conf[:, resample_id]), axis=-1)

        if motion.shape[0] == 1:
            fake = np.zeros(motion.shape, dtype=motion.dtype)
            motion = np.concatenate((motion, fake), axis=0)

        if self.scale_range:
            motion = self._mb_crop_scale(motion, scale_range=self.scale_range)

        X = torch.from_numpy(motion.astype(np.float32)).unsqueeze(0).to(self.device)
        return X

    def prepare_window(self, window_data: WindowData, sync_cuda_timing: bool) -> Tuple[Any, Dict[str, float]]:
        motion_input, prep_ms = _timed_stage(sync_cuda=sync_cuda_timing, fn=lambda: self._build_motion_input(window_data))
        return motion_input, {"temporal_prep_ms": float(prep_ms)}

    def infer(self, prepared_input: Any, sync_cuda_timing: bool) -> Tuple[Prediction, Dict[str, float]]:
        self._model.eval()

        with torch.no_grad():
            out, forward_ms = _timed_stage(
                sync_cuda=sync_cuda_timing,
                fn=lambda: self._model(prepared_input),
            )

            t_post0 = time.perf_counter()
            probs_t = torch.softmax(out, dim=1).squeeze(0).detach().cpu().numpy()

            if self.merge_fall:
                if probs_t.shape[0] == self._merged_len_expected:
                    merged_probs = probs_t
                    fall_prob = float(merged_probs[0]) if merged_probs.size else 0.0
                elif probs_t.shape[0] == self._unmerged_len_expected:
                    fall_prob = float(np.sum(probs_t[FALL_CLASS_IDS_DEFAULT]))
                    nonfall_probs = [probs_t[i] for i in range(self._unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
                    merged_probs = np.array([fall_prob] + nonfall_probs, dtype=np.float32)
                else:
                    merged_probs = probs_t
                    fall_prob = float(np.sum(probs_t[self.fall_idx])) if self.fall_idx else 0.0
            else:
                merged_probs = probs_t
                fall_prob = float(np.sum(probs_t[self.fall_idx])) if self.fall_idx else 0.0

            pred_id = int(np.argmax(merged_probs))
            pred_conf = float(np.max(merged_probs))
            pred_name = self.class_names[pred_id] if 0 <= pred_id < len(self.class_names) else str(pred_id)
            label_post_ms = (time.perf_counter() - t_post0) * 1000.0

        pred = Prediction(
            pred_id=int(pred_id),
            pred_label=str(pred_name),
            confidence=float(pred_conf),
            extra={
                "p_fall": float(fall_prob),
                "probs": merged_probs.tolist() if isinstance(merged_probs, np.ndarray) else None,
            },
        )
        return pred, {
            "temporal_forward_ms": float(forward_ms),
            "label_post_ms": float(label_post_ms),
        }
