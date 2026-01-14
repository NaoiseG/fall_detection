"""
Run from project root, e.g.:

  python -m models.svm.train_svm --kernel rbf --C 10 --gamma scale
  python -m models.svm.train_svm --kernel linear --C 1

This trains an SVM on the same subject-split window tensors used by your deep models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import glob

import numpy as np

from dataset import WindowTensorDataset  # not used, but keeps parity with project structure
from build_windows import make_window_tensors


def find_keypoints_npzs_subjects(output_root: Path, camera: int = 1, subjects=range(1, 6)):
    npzs = []
    for s in subjects:
        subj_root = output_root / f"Subject{s}"
        if not subj_root.exists():
            continue
        pat = subj_root / "Activity*" / "Trial*" / f"Subject{s}Activity*Trial*Camera{camera}" / "keypoints.npz"
        npzs.extend(glob.glob(str(pat), recursive=True))
    return sorted(npzs)


def load_windows_from_npzs(npz_paths, T=None, use_conf: bool = True):
    X_all, y_all = [], []
    T_used = T

    for i, p in enumerate(npz_paths):
        if i == 0 and T_used is None:
            X, y, T_used = make_window_tensors(p, T=None, use_conf=use_conf)
        else:
            X, y, _ = make_window_tensors(p, T=T_used, use_conf=use_conf)

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        raise RuntimeError("No NPZs found / no windows loaded.")

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0), T_used


def parse_range(r: str):
    a, b = r.split("-")
    a, b = int(a), int(b)
    return range(a, b + 1)


def windows_to_features(X: np.ndarray) -> np.ndarray:
    """
    X can be (N,T,K,C) or (N,T,F). Convert to 2D (N, D) for SVM.
    Default: flatten entire window.
    """
    if X.ndim == 4:
        N, T, K, C = X.shape
        X = X.reshape(N, T, K * C)
    assert X.ndim == 3, f"Expected (N,T,F). Got {X.shape}"

    N, T, F = X.shape
    return X.reshape(N, T * F)  # (N, D)


def main():
    parser = argparse.ArgumentParser(description="Train SVM baseline on UP-Fall windowed pose tensors.")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--train-subjects", type=str, default="16-17")
    parser.add_argument("--val-subjects", type=str, default="1-1")
    parser.add_argument("--kernel", type=str, choices=["linear", "rbf"], default="rbf")
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--gamma", type=str, default="scale", help="Only used for RBF. e.g. 'scale' or 'auto'")
    parser.add_argument("--use-conf", action="store_true", help="Include confidence channel (default True in your pipeline)")
    args = parser.parse_args()

    # Import sklearn lazily so you get a clear error if not installed
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints/outputs_npz")

    train_subjects = parse_range(args.train_subjects)
    val_subjects = parse_range(args.val_subjects)

    train_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=args.camera, subjects=train_subjects)
    val_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=args.camera, subjects=val_subjects)

    print("Train sequences:", len(train_npzs))
    print("Val sequences:", len(val_npzs))
    if not train_npzs:
        raise RuntimeError("No training NPZs found.")
    if not val_npzs:
        raise RuntimeError("No validation NPZs found.")

    X_train, y_train_tags, T_used = load_windows_from_npzs(train_npzs, T=None, use_conf=args.use_conf or True)
    X_val, y_val_tags, _ = load_windows_from_npzs(val_npzs, T=T_used, use_conf=args.use_conf or True)

    # Multiclass 1..11 -> 0..10
    y_train = (y_train_tags.astype(int) - 1).astype(np.int64)
    y_val = (y_val_tags.astype(int) - 1).astype(np.int64)

    print("Train label ids:", np.unique(y_train))
    print("Val label ids:", np.unique(y_val))

    Xtr = windows_to_features(X_train)
    Xva = windows_to_features(X_val)

    print("Feature matrix:", Xtr.shape, "Val:", Xva.shape)

    # Build model
    if args.kernel == "linear":
        clf = LinearSVC(C=args.C, max_iter=5000)
    else:
        clf = SVC(kernel="rbf", C=args.C, gamma=args.gamma)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", clf),
    ])

    pipe.fit(Xtr, y_train)
    pred = pipe.predict(Xva)

    acc = accuracy_score(y_val, pred)
    print(f"\nSVM ({args.kernel}) val accuracy: {acc:.3f}")

    # Useful diagnostics
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, pred))

    print("\nClassification report:")
    print(classification_report(y_val, pred, digits=3))


if __name__ == "__main__":
    main()
