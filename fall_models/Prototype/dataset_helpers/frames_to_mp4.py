#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

def natural_key(s: str):
    # Split into text/number chunks so img2 < img10, and timestamps sort sanely
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def require_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("Error: ffmpeg not found. Install it and ensure it's on PATH.", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert a directory of PNG frames into an MP4 using ffmpeg.")
    parser.add_argument("frames_dir", help="Directory containing PNG frames")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output MP4 filename (default: output.mp4)")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30)")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality (lower=better, default: 18)")
    parser.add_argument("--preset", default="medium", help="x264 preset (default: medium)")
    args = parser.parse_args()

    require_ffmpeg()

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not frames_dir.is_dir():
        print(f"Error: not a directory: {frames_dir}", file=sys.stderr)
        sys.exit(1)

    frames = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    if not frames:
        print(f"Error: no PNG frames found in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    frames.sort(key=lambda p: natural_key(p.name))

    # Build ffmpeg concat demuxer list file
    # -safe 0 allows absolute paths
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for p in frames:
            # Escape single quotes for ffmpeg concat format
            path_str = str(p).replace("'", r"'\''")
            f.write(f"file '{path_str}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-r", str(args.fps),
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c:v", "libx264",
            "-preset", str(args.preset),
            "-crf", str(args.crf),
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]

        print("Running:", " ".join(shlex.quote(c) for c in cmd))
        subprocess.run(cmd, check=True)
        print(f"Done: {out_path}")

    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass

if __name__ == "__main__":
    main()
