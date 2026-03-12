import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Pose


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def detect_task(model):
    head = model.model.model[-1]
    if isinstance(head, Pose):
        return "pose"
    if isinstance(head, Detect):
        return "detect"
    raise RuntimeError(f"Unsupported terminal head type: {type(head).__name__}")


def resolve_task(user_task, detected_task):
    if user_task == "auto":
        return detected_task
    if user_task != detected_task:
        raise ValueError(f"--task={user_task} does not match loaded model task '{detected_task}'")
    return user_task


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/finetune/weights/best.pt", help="model weights path")
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "detect", "pose"], help="task mode")
    parser.add_argument("--data", type=str, default=None, help="dataset yaml path")
    parser.add_argument("--device", type=str, default="0", help="device id, e.g. '0' or 'cpu'")
    parser.add_argument("--batch", type=int, default=16, help="validation batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="validation image size")
    parser.add_argument("--project", type=str, default=".", help="save project")
    parser.add_argument("--name", type=str, default="runs/val", help="run name")
    parser.add_argument("--split", type=str, default="val", help="dataset split to validate")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--plots", action="store_true", help="save validation plots")
    return parser.parse_args()


def main(opt):
    model = YOLO(opt.weights)
    detected_task = detect_task(model)
    task = resolve_task(opt.task, detected_task)
    data = opt.data or (str(ROOT / "coco-pose.yaml") if task == "pose" else str(ROOT / "ultralytics/cfg/datasets/coco.yaml"))

    print(f"detected task: {detected_task}")
    print(f"val task: {task}")
    print(f"dataset: {data}")

    metrics = model.val(
        data=data,
        task=task,
        device=opt.device,
        batch=opt.batch,
        imgsz=opt.imgsz,
        project=opt.project,
        name=opt.name,
        split=opt.split,
        workers=opt.workers,
        plots=opt.plots,
    )
    print(metrics)


if __name__ == "__main__":
    main(parse_opt())
