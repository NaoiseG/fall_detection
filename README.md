# Fall Detection From Pose Keypoints

This project detects falls by analysing human pose keypoints over time. It is trained and evaluated using the **UP-Fall** dataset. The overall idea is: extract pose keypoints for each frame, convert them into short time windows, then run a temporal model to classify each window as fall or non-fall (and optionally fall type).

## Pipeline Overview

1. **Input**
   - Video or a live camera feed

2. **Pose Estimation**
   - A pose estimator produces a set of keypoints per frame (for example, joints like shoulders, hips, knees)

3. **Preprocessing**
   - Keypoints are cleaned and transformed into a consistent representation across frames

4. **Windowing**
   - Frames are grouped into fixed-length sequences (windows) so the model can learn motion over time

5. **Temporal Inference**
   - A temporal model predicts the class for each window (fall vs non-fall, and optionally fall categories)

6. **Output**
   - Per-window predictions and summary metrics for offline runs
   - Live predictions for real-time runs

## Workflow

### 1) Prepare the dataset (UP-Fall)
- Use the UP-Fall dataset as the source of labelled fall and non-fall sequences
- Extract pose keypoints for each frame using your chosen pose estimator
- Organise the extracted keypoints into a consistent dataset structure for training, validation, and testing

### 2) Train models
- Train one or more temporal models on the windowed keypoint sequences
- Use a validation split to choose settings and compare models

### 3) Evaluate
- Evaluate trained models on held-out test data
- Review metrics to select a model that performs well on your target scenario

### 4) Run on video (real time or offline)
- Use the selected trained model to run inference on new videos or a live camera feed
- Inspect predictions to confirm behaviour on realistic examples

## What you can customise

- Pose estimator choice (the source of keypoints)
- Window length and step size (how much motion the model sees at once)
- Model type (different temporal architectures)
- Label setup (binary fall detection or multi-class fall categorisation)

## Outputs

- Trained model checkpoints for reuse
- Evaluation metrics for comparing models
- Predictions on videos or live streams
