#!/usr/bin/env python3
"""
Generate pseudo-labels in YOLO pose format for a directory of images.

Output structure:
calibration_dataset/
├── images/
│   └── val/
│       ├── frame_0001.jpg
│       └── ...
└── labels/
    └── val/
        ├── frame_0001.txt
        └── ...

Each label line is:
cls x_center y_center width height kpt1_x kpt1_y kpt1_v ... kpt17_x kpt17_y kpt17_v

All coordinates are normalized to [0, 1].
Visibility is written as:
- 2 if keypoint confidence >= kpt_conf_thres
- 1 otherwise
"""

from pathlib import Path
import shutil
import argparse
import cv2
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO pose pseudo-labels")
    parser.add_argument("--model", required=True, help="Path to pose model, e.g. yolo11l-pose.pt")
    parser.add_argument("--input-dir", required=True, help="Directory containing input images")
    parser.add_argument("--output-root", required=True, help="Root of output dataset")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--person-class", type=int, default=0, help="Class id for person")
    parser.add_argument(
        "--kpt-conf-thres",
        type=float,
        default=0.5,
        help="Keypoint confidence threshold for visibility flag"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into output-root/images/val"
    )
    parser.add_argument(
        "--save-empty",
        action="store_true",
        help="Save empty .txt files when no person is detected"
    )
    return parser.parse_args()


def list_images(input_dir: Path):
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def ensure_dirs(output_root: Path):
    images_dir = output_root / "images" / "val"
    labels_dir = output_root / "labels" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def normalize_bbox_xyxy(xyxy, w, h):
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def build_label_line(cls_id, xyxy, keypoints_xy, keypoints_conf, img_w, img_h, kpt_conf_thres):
    xc, yc, bw, bh = normalize_bbox_xyxy(xyxy, img_w, img_h)
    values = [int(cls_id), clamp01(xc), clamp01(yc), clamp01(bw), clamp01(bh)]

    num_kpts = len(keypoints_xy)
    for i in range(num_kpts):
        kx, ky = keypoints_xy[i]
        kc = float(keypoints_conf[i]) if keypoints_conf is not None else 1.0

        kx_n = clamp01(float(kx) / img_w)
        ky_n = clamp01(float(ky) / img_h)

        # YOLO pose commonly uses visibility in {0,1,2}; 2 = visible/labeled
        vis = 2 if kc >= kpt_conf_thres else 1
        values.extend([kx_n, ky_n, vis])

    out = []
    for i, v in enumerate(values):
        if i == 0:
            out.append(str(v))
        elif isinstance(v, float):
            out.append(f"{v:.6f}")
        else:
            out.append(str(v))
    return " ".join(out)


def main():
    args = parse_args()

    model_path = Path(args.model)
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = list_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    images_out_dir, labels_out_dir = ensure_dirs(output_root)
    model = YOLO(str(model_path))

    written = 0
    empty = 0
    skipped = 0

    print(f"Found {len(images)} images")
    print(f"Model: {model_path}")
    print(f"Output: {output_root}")

    for idx, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{idx}/{len(images)}] Skipping unreadable image: {img_path.name}")
            skipped += 1
            continue

        h, w = img.shape[:2]

        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )

        label_path = labels_out_dir / f"{img_path.stem}.txt"

        if args.copy_images:
            shutil.copy2(img_path, images_out_dir / img_path.name)

        if not results:
            if args.save_empty:
                label_path.write_text("")
                empty += 1
            else:
                empty += 1
            print(f"[{idx}/{len(images)}] No results: {img_path.name}")
            continue

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            if args.save_empty:
                label_path.write_text("")
            empty += 1
            print(f"[{idx}/{len(images)}] No detections/keypoints: {img_path.name}")
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        boxes_cls = r.boxes.cls.cpu().numpy().astype(int)
        boxes_conf = r.boxes.conf.cpu().numpy()

        kpts_xy = r.keypoints.xy.cpu().numpy()  # shape: [N, K, 2]
        kpts_conf = None
        if r.keypoints.conf is not None:
            kpts_conf = r.keypoints.conf.cpu().numpy()  # shape: [N, K]

        # Keep only person detections
        candidates = [i for i, cls_id in enumerate(boxes_cls) if cls_id == args.person_class]

        if not candidates:
            if args.save_empty:
                label_path.write_text("")
            empty += 1
            print(f"[{idx}/{len(images)}] No person detections: {img_path.name}")
            continue

        # Use the highest-confidence person detection
        best_i = max(candidates, key=lambda i: float(boxes_conf[i]))

        line = build_label_line(
            cls_id=args.person_class,
            xyxy=boxes_xyxy[best_i],
            keypoints_xy=kpts_xy[best_i],
            keypoints_conf=kpts_conf[best_i] if kpts_conf is not None else None,
            img_w=w,
            img_h=h,
            kpt_conf_thres=args.kpt_conf_thres
        )

        label_path.write_text(line + "\n")
        written += 1
        print(f"[{idx}/{len(images)}] Wrote: {label_path.name}")

    print("\nDone.")
    print(f"Labeled: {written}")
    print(f"Empty:   {empty}")
    print(f"Skipped: {skipped}")
    print("\nUse this in data.yaml:")
    print(f"path: {output_root.resolve()}")
    print("train: images/val")
    print("val: images/val")
    print("names:")
    print("  0: person")
    print("kpt_shape: [17, 3]")


if __name__ == "__main__":
    main()
