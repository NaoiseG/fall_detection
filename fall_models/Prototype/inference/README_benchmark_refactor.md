# Shared Benchmark Refactor Notes

This directory now uses one shared benchmark runner for CNN-LSTM, ST-GCN, and MotionBERT.

## Timing semantics

- `pose_infer_ms`: YOLO pose forward pass only.
- `track_ms`: target-lock / tracking selection only.
- `window_assembly_ms`: shared window extraction/padding from tracked sampled poses.
- `temporal_prep_ms`: classifier-specific preprocessing only.
- `temporal_forward_ms`: classifier model forward pass only.
- `temporal_total_ms`: `temporal_prep_ms + temporal_forward_ms`.
- `temporal_effective_ms` (per frame): amortized temporal cost, computed as `temporal_total_ms / raw_window_stride` and applied only over that window's stride coverage.

This avoids misreporting a full window cost as if it were re-run on every frame.

## Benchmark mode

- `--benchmark-mode` enables headless benchmarking behavior.
- In benchmark mode, display is disabled by default.
- Video writing stays off unless `--save` is explicitly provided.
- Fairness-critical pose/runtime settings are aligned across wrappers, including:
  - YOLO confidence / IOU
  - YOLO FP16 policy (`--half`)
  - YOLO `max_det` parity (`--max-det` / `--max-people`)
  - shared target-lock tracking controls

## Warm-up exclusion

- `--warmup-frames` and `--warmup-windows` exclude initial units from summary averages.
- CSV files still include all rows; warm-up rows are marked with `is_warmup_*` flags.

## Compatibility fields

`summary.json` keeps legacy keys (`preprocess_ms`, `inference_ms`, `postprocess_ms`) for downstream tools.
These are compatibility aliases with explicit semantics described in `legacy_metric_semantics`.

## MotionBERT compatibility outputs

`infer_motionbert_video.py` still writes:
- `--out-csv` legacy prediction CSV
- `--out-pkl` legacy MotionBERT-style window dataset

For fair benchmark-mode timing:
- MotionBERT benchmark runs disable per-window payload retention in the timed pass.
- If `--out-pkl` is requested in benchmark mode, a deferred compatibility export pass is used so payload work does not contaminate benchmark metrics.
