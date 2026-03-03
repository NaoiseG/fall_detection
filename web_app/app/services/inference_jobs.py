from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceJob:
    job_id: str
    status: str = "running"
    result_path: Optional[Path] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    last_frame: Optional[bytes] = None
    frame_queue: "queue.Queue[bytes]" = field(default_factory=lambda: queue.Queue(maxsize=4))
    done_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    t_start: Optional[float] = None
    t_worker_start: Optional[float] = None
    t_first_frame_produced: Optional[float] = None
    t_stream_connected: Optional[float] = None
    t_first_stream_yield: Optional[float] = None
    t_first_real_frame_yield: Optional[float] = None

    def snapshot(self) -> Dict[str, Optional[str]]:
        with self.lock:
            return {
                "status": self.status,
                "result_url": self.result_url,
                "error": self.error,
            }

    def set_done(self, result_path: Path, result_url: str) -> None:
        with self.lock:
            self.status = "done"
            self.result_path = result_path
            self.result_url = result_url
            self.error = None
        self.done_event.set()

    def set_error(self, message: str) -> None:
        with self.lock:
            self.status = "error"
            self.error = message
            self.result_path = None
            self.result_url = None
        self.done_event.set()

    def push_frame(self, jpg_frame: bytes) -> None:
        accepted = False
        try:
            self.frame_queue.put_nowait(jpg_frame)
            accepted = True
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.frame_queue.put_nowait(jpg_frame)
                accepted = True
            except queue.Full:
                pass

        if accepted:
            with self.lock:
                self.last_frame = jpg_frame

    def get_last_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.last_frame

    def mark_time(self, field_name: str, value: Optional[float] = None, overwrite: bool = False) -> float:
        stamp = float(time.perf_counter() if value is None else value)
        with self.lock:
            current_value = getattr(self, field_name, None)
            if current_value is None or overwrite:
                setattr(self, field_name, stamp)
                return stamp
            return float(current_value)

    def timing_snapshot(self) -> Dict[str, Optional[float]]:
        with self.lock:
            return {
                "t_start": self.t_start,
                "t_worker_start": self.t_worker_start,
                "t_first_frame_produced": self.t_first_frame_produced,
                "t_stream_connected": self.t_stream_connected,
                "t_first_stream_yield": self.t_first_stream_yield,
                "t_first_real_frame_yield": self.t_first_real_frame_yield,
            }


class InferenceJobManager:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, InferenceJob] = {}
        self._jobs_lock = threading.Lock()
        self._placeholder_jpeg: Optional[bytes] = None
        self._placeholder_lock = threading.Lock()

    @staticmethod
    def _mjpeg_chunk(frame: bytes) -> bytes:
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame)).encode("ascii") + b"\r\n\r\n" + frame + b"\r\n"
        )

    def _get_placeholder_jpeg(self) -> bytes:
        if self._placeholder_jpeg is not None:
            return self._placeholder_jpeg

        with self._placeholder_lock:
            if self._placeholder_jpeg is not None:
                return self._placeholder_jpeg

            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Starting inference...",
                (28, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Loading models...",
                (28, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                raise RuntimeError("Failed to generate placeholder JPEG.")

            self._placeholder_jpeg = encoded.tobytes()
            return self._placeholder_jpeg

    def start_job(
        self,
        *,
        video_path: Path,
        classification_model_path: Path,
        keypoint_model_path: Path,
        realtime: bool = True,
        display_fps: float = 0.0,
        inference_options: Optional[Dict[str, Any]] = None,
        request_start_ts: Optional[float] = None,
    ) -> InferenceJob:
        job_id = uuid.uuid4().hex
        raw_output_path = self.output_dir / f"{job_id}_raw.mp4"
        final_output_path = self.output_dir / f"{job_id}.mp4"
        job = InferenceJob(job_id=job_id, result_path=final_output_path)
        job.mark_time("t_start", request_start_ts, overwrite=True)

        with self._jobs_lock:
            self._jobs[job_id] = job

        manager_start = time.perf_counter()
        thread = threading.Thread(
            target=self._run_job,
            kwargs={
                "job": job,
                "video_path": video_path.resolve(),
                "classification_model_path": classification_model_path.resolve(),
                "keypoint_model_path": keypoint_model_path.resolve(),
                "raw_output_path": raw_output_path,
                "final_output_path": final_output_path,
                "realtime": bool(realtime),
                "display_fps": float(display_fps),
                "inference_options": dict(inference_options or {}),
            },
            daemon=True,
            name=f"inference-job-{job_id}",
        )
        thread.start()
        logger.info(
            "[timing] start_job returned job_id=%s after %.6fs",
            job_id,
            time.perf_counter() - manager_start,
        )
        return job

    def get_job(self, job_id: str) -> Optional[InferenceJob]:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def get_status(self, job_id: str) -> Optional[Dict[str, Optional[str]]]:
        job = self.get_job(job_id)
        if job is None:
            return None
        return job.snapshot()

    def get_debug_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.get_job(job_id)
        if job is None:
            return None

        status = job.snapshot()
        timings = job.timing_snapshot()

        def _delta(start: Optional[float], end: Optional[float]) -> Optional[float]:
            if start is None or end is None:
                return None
            return float(end - start)

        t_start = timings.get("t_start")
        return {
            "job_id": job.job_id,
            "status": status.get("status"),
            "result_url": status.get("result_url"),
            "error": status.get("error"),
            **timings,
            "dt_to_worker_start": _delta(t_start, timings.get("t_worker_start")),
            "dt_to_first_frame_produced": _delta(t_start, timings.get("t_first_frame_produced")),
            "dt_to_stream_connected": _delta(t_start, timings.get("t_stream_connected")),
            "dt_to_first_stream_yield": _delta(t_start, timings.get("t_first_stream_yield")),
            "dt_to_first_real_frame_yield": _delta(t_start, timings.get("t_first_real_frame_yield")),
        }

    def stream_generator(self, job_id: str) -> Optional[Iterator[bytes]]:
        job = self.get_job(job_id)
        if job is None:
            return None

        stream_connected_at = time.perf_counter()
        if job.mark_time("t_stream_connected", stream_connected_at) == stream_connected_at:
            logger.info("[timing] stream connected job_id=%s at %.6f", job_id, stream_connected_at)

        def _iter() -> Iterator[bytes]:
            placeholder = self._get_placeholder_jpeg()
            first_stream_yield_at = time.perf_counter()
            if job.mark_time("t_first_stream_yield", first_stream_yield_at) == first_stream_yield_at:
                logger.info("[timing] first stream bytes yielded job_id=%s at %.6f", job_id, first_stream_yield_at)
            yield self._mjpeg_chunk(placeholder)

            while True:
                if job.done_event.is_set() and job.frame_queue.empty():
                    break

                try:
                    frame = job.frame_queue.get(timeout=0.25)
                except queue.Empty:
                    heartbeat = job.get_last_frame() or placeholder
                    yield self._mjpeg_chunk(heartbeat)
                    continue

                first_real_yield_at = time.perf_counter()
                if job.mark_time("t_first_real_frame_yield", first_real_yield_at) == first_real_yield_at:
                    logger.info("[timing] first real frame yielded job_id=%s at %.6f", job_id, first_real_yield_at)
                yield self._mjpeg_chunk(frame)

        return _iter()

    def _run_job(
        self,
        *,
        job: InferenceJob,
        video_path: Path,
        classification_model_path: Path,
        keypoint_model_path: Path,
        raw_output_path: Path,
        final_output_path: Path,
        realtime: bool,
        display_fps: float,
        inference_options: Dict[str, Any],
    ) -> None:
        try:
            worker_start_at = time.perf_counter()
            if job.mark_time("t_worker_start", worker_start_at) == worker_start_at:
                logger.info("[timing] worker started job_id=%s at %.6f", job.job_id, worker_start_at)

            from inference.inference_on_video import run_inference_stream

            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            if raw_output_path.exists():
                raw_output_path.unlink()
            if final_output_path.exists():
                final_output_path.unlink()

            def push_frame_to_queue(frame_bgr) -> None:
                ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    return
                first_frame_at = time.perf_counter()
                if job.mark_time("t_first_frame_produced", first_frame_at) == first_frame_at:
                    logger.info(
                        "[timing] first frame produced job_id=%s at %.6f",
                        job.job_id,
                        first_frame_at,
                    )
                job.push_frame(encoded.tobytes())

            return_code = run_inference_stream(
                video_path=video_path,
                classification_model_path=classification_model_path,
                keypoint_model_path=keypoint_model_path,
                on_frame=push_frame_to_queue,
                save_path=raw_output_path,
                no_display=True,
                display_fps=float(display_fps),
                realtime=bool(realtime),
                **dict(inference_options),
            )
            if int(return_code) != 0:
                raise RuntimeError(f"Inference process failed with return code {return_code}.")
            if not raw_output_path.is_file():
                raise RuntimeError(f"Inference output was not created: {raw_output_path}")

            web_output_path = self._prepare_web_video(raw_output_path, final_output_path)
            if raw_output_path.exists():
                try:
                    raw_output_path.unlink()
                except OSError:
                    pass
            result_url = f"/static/inference_outputs/{web_output_path.name}"
            job.set_done(web_output_path, result_url)
        except Exception as error:
            job.set_error(str(error))

    def _prepare_web_video(self, raw_output_path: Path, final_output_path: Path) -> Path:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            command = [
                ffmpeg_path,
                "-y",
                "-i",
                str(raw_output_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(final_output_path),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0 and final_output_path.is_file():
                return final_output_path

        shutil.copyfile(raw_output_path, final_output_path)
        return final_output_path
