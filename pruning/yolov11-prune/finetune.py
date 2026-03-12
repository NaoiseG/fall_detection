"""
ultralytics/cfg/__init__.pyä¸­ä¿®æ”¹äº†å¢žåŠ 'finetune'ä»Žoverridesä¸­popå’Œé‡èµ‹å€¼,é˜²æ­¢å‚æ•°æ£€æŸ¥æŠ¥é”™
ultralytics/engine/model.pyä¸­å¢žåŠ äº†å¯¹'maskbndict'çš„åŠ è½½
"""
import argparse

from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Pose


# =========================================helper=========================================
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
# =========================================helper=========================================


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/pruned.pt", help="pruned checkpoint path")
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "detect", "pose"], help="task mode")
    parser.add_argument("--data", type=str, default=None, help="dataset yaml path")
    parser.add_argument("--cfg", type=str, default="ultralytics/cfg/default.yaml", help="default cfg path")
    parser.add_argument("--project", type=str, default=".", help="save project")
    parser.add_argument("--name", type=str, default="runs/finetune", help="run name")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-val", action="store_true", help="run validation after training")
    return parser.parse_args()


def main(opt):
    model = YOLO(opt.weights)
    detected_task = detect_task(model)
    task = resolve_task(opt.task, detected_task)
    data = opt.data or ("ultralytics/cfg/datasets/coco8-pose.yaml" if task == "pose" else "ultralytics/cfg/datasets/coco.yaml")

    print(f"detected task: {detected_task}")
    print(f"finetune task: {task}")
    print(f"dataset: {data}")

    model.train(
        data=data,
        cfg=opt.cfg,
        task=task,
        project=opt.project,
        name=opt.name,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        optimizer=opt.optimizer,
        lr0=opt.lr0,
        finetune=True,
        device=opt.device,
        resume=False,
        workers=opt.workers,
        multi_scale=True,
        label_smoothing=True,
    )
    if opt.run_val:
        model.val(data=data, task=task)


if __name__ == "__main__":
    main(parse_opt())
