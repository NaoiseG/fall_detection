This directory is a marker path for the ViTPose keypoint backend.

Inference uses Hugging Face model IDs configured in code:
- Detector: `PekingU/rtdetr_r50vd_coco_o365`
- Pose: `usyd-community/vitpose-base`

Model weights are downloaded by `transformers` into the local HF cache at runtime.
