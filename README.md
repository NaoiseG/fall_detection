# Fall Detection From Pose Keypoints

This repository contains the code used to train, evaluate, optimise and run a
pose-based fall detection pipeline. The system uses pose keypoints extracted
from UP-Fall videos, groups those keypoints into temporal windows and classifies
each window using a temporal model.

The main final classifiers covered by these instructions are:

- `paper_stgcn`
- `cnnlstm`

The commands below are intended for a Linux environment. Dependency installation
is environment-specific and is not documented here.

## Pipeline Overview

1. Download UP-Fall from the official dataset source.
2. Extract pose keypoints from the UP-Fall videos.
3. Train a temporal fall classifier on the extracted keypoint windows.
4. Evaluate the trained classifier on the held-out test subjects.
5. Optionally run the web application on a Jetson device.
6. Optionally reproduce pruning, quantisation and final experiment sweeps.

## Data Preparation

UP-Fall is not included in this repository. Download the dataset from the
official UP-Fall source and place it somewhere accessible from the machine used
for training and evaluation.

Pose keypoints can be extracted with:

```text
fall_models/Prototype/dataset_helpers/get_keypoints_files.py
```

The training and evaluation scripts expect a keypoint/NPZ root containing the
processed pose data. In the example commands below this is represented as:

```bash
/path/to/keypoints/yolo11l-pose
```

Replace this with the actual keypoint directory generated for the pose backend
being evaluated.

## Train One Model

Run training from the `fall_models/Prototype` directory:

```bash
cd fall_models/Prototype
```

Example command for the final ST-GCN configuration:

```bash
python -m training.train_models \
  --model paper_stgcn \
  --train-subjects 1-12 \
  --val-subjects 13-15 \
  --npz-root /path/to/keypoints/yolo11l-pose \
  --camera 1 2 \
  --label-mode center \
  --drop-ambig-share 0 \
  --T 64 \
  --stride 48 \
  --epochs 300 \
  --normalize 1 \
  --normalize-mode paper_rp \
  --rp-center-mode pixel \
  --rp-img-w 640 \
  --rp-img-h 480 \
  --missing-mode zeros_only \
  --interp-mode paper_group_linear \
  --interp-group 100 \
  --selection-metric composite_fall_fbeta_macro_f1 \
  --selection-w 0.7 \
  --selection-beta 2.0 \
  --rare-class-boost 1.5 \
  --weighted-sampler 1 \
  --conf-thres 0.05
```

To train the CNN-LSTM model with the same settings, change:

```bash
--model paper_stgcn
```

to:

```bash
--model cnnlstm
```

Training writes model run directories and checkpoint files under the Prototype
model output structure.

## Evaluate One Model

Run evaluation from the `fall_models/Prototype` directory:

```bash
cd fall_models/Prototype
```

Example command for evaluating a trained ST-GCN checkpoint:

```bash
python -m evaluation.eval_models \
  --models paper_stgcn \
  --test-subjects 16-17 \
  --npz-root /path/to/keypoints/yolo11l-pose \
  --camera 1 2 \
  --label-mode center \
  --drop-ambig-share 0 \
  --T 64 \
  --stride 48 \
  --normalize 1 \
  --normalize-mode paper_rp \
  --rp-center-mode pixel \
  --rp-img-w 640 \
  --rp-img-h 480 \
  --missing-mode zeros_only \
  --interp-mode paper_group_linear \
  --interp-group 100 \
  --weights-path /path/to/final_classification_models/stgcn/yolo11l-pose/paper_stgcn_best.pt \
  --out-dir /path/to/evaluation_outputs/paper_stgcn_yolo11l-pose
```

To evaluate CNN-LSTM, change the model name and checkpoint path, for example:

```bash
--models cnnlstm
--weights-path /path/to/final_classification_models/cnnlstm/yolo11l-pose/cnnlstm_best.pt
```

Evaluation writes metrics, summaries and confusion matrices under the selected
`--out-dir`.

## Final Experiment Runners

The scripts in `fall_models/Prototype/final_scripts/` are convenience runners
used for the larger paper experiments. They call the training and evaluation
entrypoints above with fixed settings.

Useful runners include:

- `train_final_models.py`: trains final temporal classifiers across pose backends.
- `eval_final_models.py`: evaluates trained classifiers across keypoint variants.
- `eval_optimised_pipelines.py`: evaluates optimised/pruned/quantised pipelines.
- `evaluate_temporal_window_sweep.py`: reproduces temporal window ablations.
- `evaluate_temporal_downsample.py`: reproduces temporal downsampling ablations.
- `validate_quantised_models.py`: validates quantised pose models.
- `validate_pruned_quantised_models.py`: validates pruned and quantised pose models.

These scripts are useful for reproducing full paper tables, but they are not
required to understand the basic train/evaluate workflow.

## Web Application

The web application is in:

```text
web_app/
```

Run it on the Jetson device:

```bash
cd web_app
python run.py
```

The Flask app binds to `0.0.0.0` on port `5000`.

From the Jetson itself, open:

```text
http://127.0.0.1:5000/
```

From another device on the same network, open:

```text
http://<jetson-ip>:5000/
```

Alternatively, access it through SSH port forwarding:

```bash
ssh -L 5000:localhost:5000 user@<jetson-ip>
```

Then open this on the local machine:

```text
http://127.0.0.1:5000/
```

The supported Git-tracked web app configurations use:

- YOLO11s FP16
- YOLO11m FP16
- YOLO11l FP16
- YOLO11 pruned 90 FP16

## Quantisation

YOLO pose models were quantised/exported using the Ultralytics `yolo export`
command.

Representative FP16 TensorRT export:

```bash
yolo export model=/path/to/yolo11l-pose.pt format=engine half=True imgsz=640
```

Representative INT8 TensorRT export:

```bash
yolo export model=/path/to/yolo11l-pose.pt format=engine int8=True data=/path/to/data.yaml imgsz=640
```

Use the exported `.engine` files when benchmarking or deploying optimised pose
backends.

## Pruning

Pose model pruning is performed with:

```text
pruning/prune_pose.py
```

Example command:

```bash
python pruning/prune_pose.py \
  --models /path/to/yolo11n-pose.pt /path/to/yolo11s-pose.pt \
  --data pruning/coco-pose.yaml \
  --task pose \
  --imgsz 640 \
  --batch 16 \
  --patience 20 \
  --workers 8 \
  --flops 90% 80% 70% \
  --name_prefix pruned_pose \
  --project /path/to/output_runs \
  --device 0 \
  --cache
```

This runs pruning at the requested FLOP reduction levels and writes the resulting
models and training outputs under the selected `--project` directory.

