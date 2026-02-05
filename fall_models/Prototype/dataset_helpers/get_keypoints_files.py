"""
Entry-point script to run pose extraction on the full UP-Fall directory tree.

This file ONLY contains the main() logic.
All pose logic, batching, timestamp parsing, etc. remain in pose.py unchanged.
"""

from pose import PoseExportConfig, batch_process_upfall_tree, run_pose_on_frames

from pathlib import Path
import glob
import os

def find_camera_folders_subjects(root, camera=1, subjects=range(1, 6)):
    folders = []
    for s in subjects:
        subj_root = Path(root) / f"Subject{s}"
        if not subj_root.exists():
            continue
        pat = subj_root / "**" / f"*Camera{camera}"
        folders.extend([str(p) for p in glob.glob(str(pat), recursive=True) if os.path.isdir(p)])
    return folders


def main():
    # --- Configure paths for your PC ---
    UPFALL_ROOT = Path("../../Datasets/UPFall")              # change if needed
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints/outputs_npz")   # change if needed

    cfg = PoseExportConfig(
        model_path=str(Path("yolo11l-pose.pt")),  # change if needed
        conf_thres=0.25,
        fps=30,
        max_people=1,
        save_csv=False,
        # if you add device="cpu" to PoseExportConfig later, set it here
    )

    camera_folders = find_camera_folders_subjects(
        root=str(UPFALL_ROOT),
        camera=1,
        subjects=range(12, 13),
    )

    print("Camera folders found:", len(camera_folders))
    total = len(camera_folders)
    results = []

    for i, frames_dir in enumerate(camera_folders, 1):
        print(f"\n[{i}/{total}] Processing: {frames_dir}")

        trial_dir = os.path.dirname(frames_dir)
        matches = glob.glob(os.path.join(trial_dir, "*Features1&0.5.csv"))
        if not matches:
            print("  ↳ no Features1&0.5.csv found, skipping")
            continue
        windows_csv = matches[0]

        rel = os.path.relpath(frames_dir, str(UPFALL_ROOT))
        out_dir = OUTPUT_ROOT / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "keypoints.npz").exists():
            print("  ↳ already exists, skipping")
            continue

        out_video, out_npz, _ = run_pose_on_frames(
            frames_dir=frames_dir,
            out_dir=str(out_dir),
            windows_csv=windows_csv,
            config=cfg,
            pattern="*.png",
        )

        print(f"  ✓ wrote {out_npz}")
        results.append(out_npz)

    print("\nDone.")
    print("Processed sequences:", len(results))


# def main():
#     cfg = PoseExportConfig(
#         model_path="yolo11l-pose.pt",
#         conf_thres=0.25,
#         fps=30,
#         max_people=1,
#         save_csv=False,
#     )

#     # Root of the UP-Fall dataset (Subject/Activity/Trial/...)
#     upfall_root = "../Datasets/UPFall"

#     # Where all keypoints.npz outputs will be written
#     output_root = "Prototype/outputs/outputs_npz"

#     batch_process_upfall_tree(
#         upfall_root=upfall_root,
#         output_root=output_root,
#         config=cfg,
#         camera=1,
#         features_suffix="Features1&0.5.csv",
#         pattern="*.png",
#     )


if __name__ == "__main__":
    main()
