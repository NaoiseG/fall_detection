"""
ä¿®æ”¹çš„ä»£ç :
ultralytics/nn/modules/block.py: å¯¹C3k2å¢žåŠ ä¸€ä¸ªC3kå¸ƒå°”å€¼å±žæ€§
ultralytics/engine/trainer.py: ç¦ç”¨amp, æ¢¯åº¦è£å‰ª, å¢žåŠ æ¢¯åº¦æƒ©ç½šé¡¹ç³»æ•°
ultralytics/engine/model.py: ä¸»è¦æ˜¯å°†srå‚æ•°ç»‘å®šåˆ°self.trainerä¸Š
ultralytics/cfg/__init__.py: å¯¹é¢å¤–å‚æ•°finetuneçš„å¤„ç†, é˜²æ­¢DDPä¸‹æŠ¥é”™
ultralytics/engine/model.py: å¯¹sr, maskbndictç­‰é¢å¤–å‚æ•°çš„å¤„ç†
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train-normal/weights/best.pt", help="model weights path")
    parser.add_argument("--data", type=str, default=str(ROOT / "coco-pose.yaml"), help="dataset yaml path")
    parser.add_argument("--cfg", type=str, default="ultralytics/cfg/default.yaml", help="default cfg path")
    parser.add_argument("--project", type=str, default=".", help="save project")
    parser.add_argument("--name", type=str, default="runs/train-sparsity", help="run name")
    parser.add_argument("--device", type=str, default="0", help="device ids, e.g. '0'")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--sr", type=float, default=1e-3, help="sparsity regularization strength")
    return parser.parse_args()


def main(opt):
    model = YOLO(opt.weights)
    model.train(
        sr=opt.sr,
        data=opt.data,
        cfg=opt.cfg,
        project=opt.project,
        name=opt.name,
        device=opt.device,  # NOTE: currently intended for single-GPU sparsity training.
        epochs=opt.epochs,
        batch=opt.batch,
        optimizer=opt.optimizer,
        lr0=opt.lr0,
        patience=opt.patience,
    )


if __name__ == "__main__":
    main(parse_opt())
