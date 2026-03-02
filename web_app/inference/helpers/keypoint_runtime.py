from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


K = 17

DEFAULT_ALPHAPOSE_CFG = "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
DEFAULT_ALPHAPOSE_CHECKPOINT = "pretrained_models/fast_res50_256x192.pth"
DEFAULT_ALPHAPOSE_DETECTOR_CFG = "detector/yolo/cfg/yolov3-spp.cfg"
DEFAULT_ALPHAPOSE_DETECTOR_WEIGHTS = "detector/yolo/data/yolov3-spp.weights"


@dataclass
class KeypointDetections:
    xy: np.ndarray
    conf: np.ndarray
    box_centers: np.ndarray
    box_conf: Optional[np.ndarray]
    raw: Optional[Dict[str, Any]] = None


def infer_keypoint_backend(model_path: Path, backend: Optional[str] = None) -> str:
    if backend is not None:
        normalized = str(backend).strip().lower()
        if normalized in {"yolo", "ultralytics"}:
            return "yolo"
        if normalized in {"alphapose", "alpha-pose"}:
            return "alphapose"
        raise ValueError(f"Unsupported keypoint backend: {backend}")

    path = Path(model_path).expanduser()
    path_str = str(path).lower()
    if path.is_dir():
        if (path / "alphapose").is_dir() or path.name.lower() == "alphapose":
            return "alphapose"
    if "alphapose" in path_str:
        return "alphapose"

    return "yolo"


class KeypointRuntime:
    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        backend: Optional[str] = None,
        alphapose_cfg: str = DEFAULT_ALPHAPOSE_CFG,
        alphapose_checkpoint: str = DEFAULT_ALPHAPOSE_CHECKPOINT,
        alphapose_detector_cfg: str = DEFAULT_ALPHAPOSE_DETECTOR_CFG,
        alphapose_detector_weights: str = DEFAULT_ALPHAPOSE_DETECTOR_WEIGHTS,
        alphapose_conf_thres: float = 0.1,
        alphapose_nms_thres: float = 0.6,
        alphapose_min_box_area: int = 0,
        alphapose_flip: bool = False,
        alphapose_vis_fast: bool = True,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = str(device)
        self.backend = infer_keypoint_backend(self.model_path, backend)
        self._is_cuda = str(self.device).lower().startswith("cuda") and torch.cuda.is_available()

        self._yolo_model = None
        self._alphapose_root: Optional[Path] = None
        self._ap_opt: Optional[_AlphaPoseOpt] = None
        self._ap_vis_thres: Optional[List[float]] = None
        self._ap_vis_fast = bool(alphapose_vis_fast)

        if self.backend == "yolo":
            from ultralytics import YOLO

            if not self.model_path.is_file():
                raise FileNotFoundError(f"YOLO keypoint weights not found: {self.model_path}")
            self._yolo_model = YOLO(str(self.model_path))
        else:
            self._init_alphapose(
                cfg_rel=alphapose_cfg,
                checkpoint_rel=alphapose_checkpoint,
                detector_cfg_rel=alphapose_detector_cfg,
                detector_weights_rel=alphapose_detector_weights,
                conf_thres=float(alphapose_conf_thres),
                nms_thres=float(alphapose_nms_thres),
                min_box_area=int(alphapose_min_box_area),
                flip=bool(alphapose_flip),
                vis_fast=bool(alphapose_vis_fast),
            )

    def predict(
        self,
        *,
        frame_bgr: np.ndarray,
        imgsz: int,
        conf: float,
        max_people: int,
        use_half: bool,
    ) -> KeypointDetections:
        if self.backend == "yolo":
            return self._predict_yolo(
                frame_bgr=frame_bgr,
                imgsz=int(imgsz),
                conf=float(conf),
                max_people=int(max_people),
                use_half=bool(use_half),
            )
        return self._predict_alphapose(frame_bgr=frame_bgr, max_people=int(max_people))

    def render(self, *, frame_bgr: np.ndarray, detections: KeypointDetections) -> np.ndarray:
        if self.backend != "alphapose":
            return frame_bgr
        if not detections.raw:
            return frame_bgr
        im_res = detections.raw.get("im_res")
        if not im_res or not im_res.get("result"):
            return frame_bgr
        if self._ap_opt is None or self._ap_vis_thres is None:
            return frame_bgr

        if self._ap_vis_fast:
            from alphapose.utils.vis import vis_frame_fast as vis_frame
        else:
            from alphapose.utils.vis import vis_frame

        return vis_frame(frame_bgr, im_res, self._ap_opt, self._ap_vis_thres.copy())

    def _predict_yolo(
        self,
        *,
        frame_bgr: np.ndarray,
        imgsz: int,
        conf: float,
        max_people: int,
        use_half: bool,
    ) -> KeypointDetections:
        if self._yolo_model is None:
            raise RuntimeError("YOLO keypoint runtime is not initialized.")

        results = self._yolo_model.predict(
            source=frame_bgr,
            imgsz=int(imgsz),
            conf=float(conf),
            verbose=False,
            device=self.device,
            half=bool(use_half) and self._is_cuda,
            max_det=max(1, int(max_people)),
        )
        if not results or len(results) == 0 or results[0].keypoints is None:
            return _empty_detections()

        r = results[0]
        kpts = r.keypoints
        xy_all = kpts.xy.cpu().numpy() if hasattr(kpts.xy, "cpu") else np.asarray(kpts.xy)
        cf_all = None
        if hasattr(kpts, "conf") and getattr(kpts, "conf", None) is not None:
            cf_all = kpts.conf.cpu().numpy() if hasattr(kpts.conf, "cpu") else np.asarray(kpts.conf)

        if xy_all.ndim != 3 or xy_all.shape[0] == 0:
            return _empty_detections()
        if int(xy_all.shape[1]) != int(K):
            raise ValueError(f"Expected {K} keypoints, got {xy_all.shape[1]}")

        box_centers, box_conf = _extract_yolo_box_centers_conf(r)
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
            return _empty_detections()

        xy = xy_all[:num_candidates].astype(np.float32, copy=False)
        box_centers = box_centers[:num_candidates].astype(np.float32, copy=False)
        conf_arr: np.ndarray
        if cf_all is not None:
            conf_arr = cf_all[:num_candidates].astype(np.float32, copy=False)
            if conf_arr.shape != (num_candidates, K):
                conf_arr = np.ones((num_candidates, K), dtype=np.float32)
        else:
            conf_arr = np.ones((num_candidates, K), dtype=np.float32)
        if box_conf is not None:
            box_conf = box_conf[:num_candidates].astype(np.float32, copy=False)

        return KeypointDetections(
            xy=xy,
            conf=conf_arr,
            box_centers=box_centers,
            box_conf=box_conf,
            raw={"result": r},
        )

    def _init_alphapose(
        self,
        *,
        cfg_rel: str,
        checkpoint_rel: str,
        detector_cfg_rel: str,
        detector_weights_rel: str,
        conf_thres: float,
        nms_thres: float,
        min_box_area: int,
        flip: bool,
        vis_fast: bool,
    ) -> None:
        self._alphapose_root = _resolve_alphapose_root(self.model_path)
        root_str = str(self._alphapose_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        cfg_path = _resolve_alphapose_path(self._alphapose_root, cfg_rel)
        checkpoint_path = _resolve_alphapose_path(self._alphapose_root, checkpoint_rel)
        detector_cfg_path = _resolve_alphapose_path(self._alphapose_root, detector_cfg_rel)
        detector_weights_path = _resolve_alphapose_path(self._alphapose_root, detector_weights_rel)

        for label, path in (
            ("AlphaPose cfg", cfg_path),
            ("AlphaPose checkpoint", checkpoint_path),
            ("AlphaPose detector cfg", detector_cfg_path),
            ("AlphaPose detector weights", detector_weights_path),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        from alphapose.models import builder
        from alphapose.utils.config import update_config
        from alphapose.utils.pPose_nms import pose_nms
        from alphapose.utils.presets import SimpleTransform
        from alphapose.utils.transforms import flip as ap_flip
        from alphapose.utils.transforms import flip_heatmap
        from alphapose.utils.transforms import get_func_heatmap_to_coord
        from detector import yolo_cfg as yolo_cfg_mod
        from detector.apis import get_detector

        yolo_cfg_mod.cfg.CONFIG = str(detector_cfg_path)
        yolo_cfg_mod.cfg.WEIGHTS = str(detector_weights_path)
        yolo_cfg_mod.cfg.CONFIDENCE = float(conf_thres)
        yolo_cfg_mod.cfg.NMS_THRES = float(nms_thres)

        gpus = [0] if self._is_cuda else [-1]
        ap_device = torch.device(f"cuda:{gpus[0]}" if gpus[0] >= 0 else "cpu")

        self._ap_opt = _AlphaPoseOpt(
            detector="yolo",
            gpus=gpus,
            device=ap_device,
            tracking=False,
            pose_track=False,
            pose_flow=False,
            min_box_area=int(min_box_area),
            flip=bool(flip),
            vis_fast=bool(vis_fast),
            showbox=False,
        )

        self._ap_flip = ap_flip
        self._ap_flip_heatmap = flip_heatmap
        self._ap_pose_nms = pose_nms
        self._ap_cfg = update_config(str(cfg_path))
        self._ap_detector = get_detector(self._ap_opt)
        self._ap_pose_model = builder.build_sppe(self._ap_cfg.MODEL, preset_cfg=self._ap_cfg.DATA_PRESET)
        self._ap_pose_model.load_state_dict(torch.load(str(checkpoint_path), map_location=ap_device))
        self._ap_pose_model.to(ap_device).eval()

        self._ap_pose_dataset = builder.retrieve_dataset(self._ap_cfg.DATASET.TRAIN)
        self._ap_heatmap_to_coord = get_func_heatmap_to_coord(self._ap_cfg)
        self._ap_norm_type = self._ap_cfg.LOSS.get("NORM_TYPE", None)
        self._ap_hm_size = self._ap_cfg.DATA_PRESET.HEATMAP_SIZE
        self._ap_eval_joints = list(range(self._ap_cfg.DATA_PRESET.NUM_JOINTS))

        if int(self._ap_cfg.DATA_PRESET.NUM_JOINTS) != int(K):
            raise ValueError(
                f"AlphaPose model joints={int(self._ap_cfg.DATA_PRESET.NUM_JOINTS)} do not match expected K={int(K)}."
            )

        self._ap_transformation = SimpleTransform(
            self._ap_pose_dataset,
            scale_factor=0,
            input_size=self._ap_cfg.DATA_PRESET.IMAGE_SIZE,
            output_size=self._ap_cfg.DATA_PRESET.HEATMAP_SIZE,
            rot=0,
            sigma=self._ap_cfg.DATA_PRESET.SIGMA,
            train=False,
            add_dpg=False,
            gpu_device=ap_device,
        )

        loss_type = self._ap_cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss")
        num_joints = int(self._ap_cfg.DATA_PRESET.NUM_JOINTS)
        self._ap_hand_face_num = None
        if loss_type == "MSELoss":
            self._ap_vis_thres = [0.4] * num_joints
        elif "JointRegression" in loss_type:
            self._ap_vis_thres = [0.05] * num_joints
        elif loss_type == "Combined":
            if num_joints == 68:
                self._ap_hand_face_num = 42
            else:
                self._ap_hand_face_num = 110
            self._ap_vis_thres = [0.4] * (num_joints - self._ap_hand_face_num) + [0.05] * self._ap_hand_face_num
        else:
            self._ap_vis_thres = [0.4] * num_joints

        self._ap_use_heatmap_loss = loss_type == "MSELoss"
        self._ap_min_box_area = int(min_box_area)

    def _predict_alphapose(self, *, frame_bgr: np.ndarray, max_people: int) -> KeypointDetections:
        if self._alphapose_root is None:
            raise RuntimeError("AlphaPose keypoint runtime is not initialized.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self._alphapose_infer(frame_rgb, image_name="frame")
        people = list(out.get("people", []))
        if not people:
            return _empty_detections(raw={"im_res": out.get("im_res"), "people": []})

        people.sort(key=lambda item: float(item.get("person_conf", 0.0)), reverse=True)
        people = people[: max(1, int(max_people))]

        xy = np.stack([np.asarray(p["kpts_xy"], dtype=np.float32) for p in people], axis=0)
        conf = np.stack([np.asarray(p["kpts_conf"], dtype=np.float32).reshape(-1) for p in people], axis=0)

        if xy.shape[1:] != (K, 2):
            raise ValueError(f"AlphaPose produced invalid keypoint shape: {xy.shape}")
        if conf.shape[1] != K:
            conf = np.ones((xy.shape[0], K), dtype=np.float32)

        box_centers_list: List[np.ndarray] = []
        box_conf_list: List[float] = []
        for person in people:
            box_xywh = np.asarray(person.get("box_xywh", []), dtype=np.float32).reshape(-1)
            if box_xywh.size >= 4 and np.all(np.isfinite(box_xywh[:4])):
                center = np.array(
                    [box_xywh[0] + 0.5 * box_xywh[2], box_xywh[1] + 0.5 * box_xywh[3]],
                    dtype=np.float32,
                )
            else:
                center = np.mean(np.asarray(person["kpts_xy"], dtype=np.float32), axis=0).astype(np.float32)
            box_centers_list.append(center)
            box_conf_list.append(float(person.get("person_conf", 0.0)))

        box_centers = np.stack(box_centers_list, axis=0).astype(np.float32, copy=False)
        box_conf = np.asarray(box_conf_list, dtype=np.float32)

        return KeypointDetections(
            xy=xy.astype(np.float32, copy=False),
            conf=conf.astype(np.float32, copy=False),
            box_centers=box_centers,
            box_conf=box_conf,
            raw={"im_res": out.get("im_res"), "people": people},
        )

    def _alphapose_infer(self, image_rgb: np.ndarray, image_name: str) -> Dict[str, Any]:
        if image_rgb is None:
            return {"im_res": {"imgname": image_name, "result": []}, "people": []}

        img = self._ap_detector.image_preprocess(image_rgb)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        im_dim = image_rgb.shape[1], image_rgb.shape[0]
        im_dim_list = torch.FloatTensor(im_dim).repeat(1, 2)

        with torch.no_grad():
            dets = self._ap_detector.images_detection(img, im_dim_list)
            if isinstance(dets, int) or dets is None or dets.shape[0] == 0:
                return {"im_res": {"imgname": image_name, "result": []}, "people": []}
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()

            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)

            boxes = boxes[dets[:, 0] == 0]
            scores = scores[dets[:, 0] == 0]
            ids = ids[dets[:, 0] == 0]
            if boxes.nelement() == 0:
                return {"im_res": {"imgname": image_name, "result": []}, "people": []}

            inps = torch.zeros(boxes.size(0), 3, *self._ap_cfg.DATA_PRESET.IMAGE_SIZE)
            cropped_boxes = torch.zeros(boxes.size(0), 4)

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self._ap_transformation.test_transform(image_rgb, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            inps = inps.to(self._ap_opt.device)
            if self._ap_opt.flip:
                inps = torch.cat((inps, self._ap_flip(inps)))

            hm = self._ap_pose_model(inps)
            if self._ap_opt.flip:
                hm_flip = self._ap_flip_heatmap(
                    hm[int(len(hm) / 2):],
                    self._ap_pose_dataset.joint_pairs,
                    shift=True,
                )
                hm = (hm[0: int(len(hm) / 2)] + hm_flip) / 2
            hm = hm.cpu()

            pose_coords = []
            pose_scores = []
            for i in range(hm.shape[0]):
                bbox = cropped_boxes[i].tolist()
                if isinstance(self._ap_heatmap_to_coord, list):
                    if self._ap_hand_face_num is None:
                        raise RuntimeError("AlphaPose combined loss expected hand/face joint count.")
                    pose_coords_body, pose_scores_body = self._ap_heatmap_to_coord[0](
                        hm[i][self._ap_eval_joints[:-self._ap_hand_face_num]],
                        bbox,
                        hm_shape=self._ap_hm_size,
                        norm_type=self._ap_norm_type,
                    )
                    pose_coords_fh, pose_scores_fh = self._ap_heatmap_to_coord[1](
                        hm[i][self._ap_eval_joints[-self._ap_hand_face_num:]],
                        bbox,
                        hm_shape=self._ap_hm_size,
                        norm_type=self._ap_norm_type,
                    )
                    pose_coord = np.concatenate((pose_coords_body, pose_coords_fh), axis=0)
                    pose_score = np.concatenate((pose_scores_body, pose_scores_fh), axis=0)
                else:
                    pose_coord, pose_score = self._ap_heatmap_to_coord(
                        hm[i][self._ap_eval_joints],
                        bbox,
                        hm_shape=self._ap_hm_size,
                        norm_type=self._ap_norm_type,
                    )
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            boxes, scores, ids, preds_img, preds_scores, _ = self._ap_pose_nms(
                boxes,
                scores,
                ids,
                preds_img,
                preds_scores,
                self._ap_min_box_area,
                use_heatmap_loss=self._ap_use_heatmap_loss,
            )
            norm_ids = _normalize_ids(ids, len(scores))

            result_list = []
            people = []
            for k in range(len(scores)):
                kp = preds_img[k]
                kp_score = preds_scores[k]
                proposal_score = torch.mean(kp_score) + scores[k] + 1.25 * torch.max(kp_score)
                box = boxes[k]
                box_xywh = [
                    float(box[0]),
                    float(box[1]),
                    float(box[2] - box[0]),
                    float(box[3] - box[1]),
                ]
                result_list.append(
                    {
                        "keypoints": kp,
                        "kp_score": kp_score,
                        "proposal_score": float(proposal_score),
                        "idx": norm_ids[k],
                        "box": box_xywh,
                    }
                )
                people.append(
                    {
                        "kpts_xy": kp.cpu().numpy().astype(np.float32),
                        "kpts_conf": kp_score.cpu().numpy().astype(np.float32).reshape(-1),
                        "person_conf": float(proposal_score),
                        "box_xywh": np.asarray(box_xywh, dtype=np.float32),
                        "track_id": int(norm_ids[k]),
                    }
                )

            return {"im_res": {"imgname": image_name, "result": result_list}, "people": people}


class _AlphaPoseOpt:
    def __init__(
        self,
        *,
        detector: str,
        gpus: List[int],
        device: torch.device,
        tracking: bool,
        pose_track: bool,
        pose_flow: bool,
        min_box_area: int,
        flip: bool,
        vis_fast: bool,
        showbox: bool,
    ) -> None:
        self.detector = detector
        self.gpus = gpus
        self.device = device
        self.tracking = tracking
        self.pose_track = pose_track
        self.pose_flow = pose_flow
        self.min_box_area = min_box_area
        self.flip = flip
        self.vis_fast = vis_fast
        self.showbox = showbox
        self.sp = True
        self.save_img = False
        self.vis = False
        self.format = None
        self.eval = False


def _resolve_alphapose_root(path: Path) -> Path:
    p = Path(path).expanduser().resolve()
    candidates: List[Path] = []
    if p.is_dir():
        candidates.append(p)
    else:
        candidates.append(p.parent)
    candidates.extend(list(candidates[0].parents))

    for candidate in candidates:
        if (candidate / "alphapose").is_dir() and (candidate / "configs").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not resolve AlphaPose root from path: "
        f"{p}. Expected a directory containing 'alphapose/' and 'configs/'."
    )


def _resolve_alphapose_path(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _coerce_int_id(value: Any) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0
        return int(value.view(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0
        return _coerce_int_id(value[0])
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_ids(ids: Any, count: int) -> List[int]:
    if ids is None:
        return [0] * count
    if torch.is_tensor(ids):
        if ids.numel() == 0:
            return [0] * count
        values = [_coerce_int_id(v) for v in ids.view(-1)]
    elif isinstance(ids, np.ndarray):
        if ids.size == 0:
            return [0] * count
        values = [_coerce_int_id(v) for v in ids.reshape(-1)]
    elif isinstance(ids, (list, tuple)):
        values = [_coerce_int_id(v) for v in ids]
    else:
        values = [_coerce_int_id(ids)]

    if len(values) < count:
        values.extend([0] * (count - len(values)))
    elif len(values) > count:
        values = values[:count]
    return values


def _to_numpy_or_none(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _extract_yolo_box_centers_conf(result: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if result is None or getattr(result, "boxes", None) is None:
        return np.empty((0, 2), dtype=np.float32), None

    boxes_xyxy = _to_numpy_or_none(getattr(result.boxes, "xyxy", None))
    if boxes_xyxy is None:
        return np.empty((0, 2), dtype=np.float32), None
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[0] == 0 or boxes_xyxy.shape[1] < 4:
        return np.empty((0, 2), dtype=np.float32), None

    boxes_xyxy = boxes_xyxy[:, :4]
    centers = np.column_stack(
        (
            0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]),
            0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]),
        )
    ).astype(np.float32, copy=False)

    box_conf = _to_numpy_or_none(getattr(result.boxes, "conf", None))
    if box_conf is not None:
        box_conf = np.asarray(box_conf, dtype=np.float32).reshape(-1)
        if box_conf.shape[0] < centers.shape[0]:
            box_conf = np.pad(
                box_conf,
                (0, centers.shape[0] - box_conf.shape[0]),
                mode="constant",
                constant_values=np.nan,
            )
        elif box_conf.shape[0] > centers.shape[0]:
            box_conf = box_conf[: centers.shape[0]]

    return centers, box_conf


def _empty_detections(*, raw: Optional[Dict[str, Any]] = None) -> KeypointDetections:
    return KeypointDetections(
        xy=np.zeros((0, K, 2), dtype=np.float32),
        conf=np.zeros((0, K), dtype=np.float32),
        box_centers=np.zeros((0, 2), dtype=np.float32),
        box_conf=None,
        raw=raw,
    )

