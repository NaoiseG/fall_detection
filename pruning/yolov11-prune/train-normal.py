import argparse
from pathlib import Path

from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(ROOT / "weights/yolo11n-pose.pt"), help="model weights path")
    parser.add_argument("--data", type=str, default=str(ROOT / "coco-pose.yaml"), help="dataset yaml path")
    parser.add_argument("--cfg", type=str, default="ultralytics/cfg/default.yaml", help="default cfg path")
    parser.add_argument("--project", type=str, default=".", help="save project")
    parser.add_argument("--name", type=str, default="runs/train-normal", help="run name")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0,1", help="device ids, e.g. '0' or '0,1'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    return parser.parse_args()


def main(opt):
    model = YOLO(opt.weights)
    model.train(
        sr=0,
        data=opt.data,
        cfg=opt.cfg,
        project=opt.project,
        name=opt.name,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        resume=False,
        workers=opt.workers,
        optimizer=opt.optimizer,
        lr0=opt.lr0,
        patience=opt.patience,
        multi_scale=True,
        label_smoothing=True,
    )


if __name__ == "__main__":
    main(parse_opt())
