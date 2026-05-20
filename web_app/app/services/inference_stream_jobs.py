from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


logger = logging.getLogger(__name__)


def _format_sse_event(event_name: str, payload_json: str) -> str:
    return f"event: {event_name}\ndata: {payload_json}\n\n"


@dataclass
class InferenceStreamJob:
    job_id: str
    save_path: Optional[Path] = None
    accepts_client_frames: bool = False
    status: str = "running"  # running | done | error
    error: Optional[str] = None
    packet_queue: "queue.Queue[str]" = field(default_factory=lambda: queue.Queue(maxsize=64))
    client_frame_queue: "queue.Queue[bytes]" = field(default_factory=lambda: queue.Queue(maxsize=4))
    last_packet: Optional[str] = None
    done_event: threading.Event = field(default_factory=threading.Event)
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> Dict[str, Optional[str]]:
        with self.lock:
            return {
                "status": self.status,
                "error": self.error,
                "save_path": str(self.save_path) if self.save_path is not None else None,
            }

    def request_stop(self) -> None:
        self.stop_event.set()

    def set_done(self) -> None:
        with self.lock:
            self.status = "done"
            self.error = None
        self.done_event.set()

    def set_error(self, message: str) -> None:
        with self.lock:
            self.status = "error"
            self.error = message
        self.done_event.set()

    def push_packet(self, packet_dict: Dict[str, Any]) -> None:
        packet_json = json.dumps(packet_dict, separators=(",", ":"), ensure_ascii=True)
        accepted = False
        try:
            self.packet_queue.put_nowait(packet_json)
            accepted = True
        except queue.Full:
            try:
                self.packet_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.packet_queue.put_nowait(packet_json)
                accepted = True
            except queue.Full:
                accepted = False

        if accepted:
            with self.lock:
                self.last_packet = packet_json

    def push_client_frame(self, frame_bytes: bytes) -> bool:
        if not frame_bytes:
            return False
        try:
            self.client_frame_queue.put_nowait(frame_bytes)
            return True
        except queue.Full:
            try:
                self.client_frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.client_frame_queue.put_nowait(frame_bytes)
                return True
            except queue.Full:
                return False

    def pop_client_frame(self, timeout_s: float = 1.0) -> Optional[bytes]:
        try:
            return self.client_frame_queue.get(timeout=max(0.0, float(timeout_s)))
        except queue.Empty:
            return None


class InferenceStreamJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, InferenceStreamJob] = {}
        self._jobs_lock = threading.Lock()

    def start_job(
        self,
        *,
        video_path: Path,
        classification_model: str,
        classification_model_path: Path,
        keypoint_model_path: Path,
        inference_options: Optional[Dict[str, Any]] = None,
        save_path: Optional[Path] = None,
    ) -> InferenceStreamJob:
        resolved_save_path = Path(save_path).resolve() if save_path is not None else None
        job = InferenceStreamJob(job_id=uuid.uuid4().hex, save_path=resolved_save_path)
        with self._jobs_lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            kwargs={
                "job": job,
                "video_path": video_path.resolve(),
                "classification_model": str(classification_model),
                "classification_model_path": classification_model_path.resolve(),
                "keypoint_model_path": keypoint_model_path.resolve(),
                "inference_options": dict(inference_options or {}),
                "save_path": job.save_path,
            },
            daemon=True,
            name=f"inference-stream-job-{job.job_id}",
        )
        thread.start()
        return job

    def start_live_job(
        self,
        *,
        camera_index: int,
        classification_model: str,
        classification_model_path: Path,
        keypoint_model_path: Path,
        inference_options: Optional[Dict[str, Any]] = None,
        save_path: Optional[Path] = None,
    ) -> InferenceStreamJob:
        resolved_save_path = Path(save_path).resolve() if save_path is not None else None
        job = InferenceStreamJob(job_id=uuid.uuid4().hex, save_path=resolved_save_path)
        with self._jobs_lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_live_job,
            kwargs={
                "job": job,
                "camera_index": int(camera_index),
                "classification_model": str(classification_model),
                "classification_model_path": classification_model_path.resolve(),
                "keypoint_model_path": keypoint_model_path.resolve(),
                "inference_options": dict(inference_options or {}),
                "save_path": job.save_path,
            },
            daemon=True,
            name=f"inference-live-job-{job.job_id}",
        )
        thread.start()
        return job

    def start_client_live_job(
        self,
        *,
        classification_model: str,
        classification_model_path: Path,
        keypoint_model_path: Path,
        inference_options: Optional[Dict[str, Any]] = None,
        save_path: Optional[Path] = None,
    ) -> InferenceStreamJob:
        resolved_save_path = Path(save_path).resolve() if save_path is not None else None
        job = InferenceStreamJob(
            job_id=uuid.uuid4().hex,
            save_path=resolved_save_path,
            accepts_client_frames=True,
        )
        with self._jobs_lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_client_live_job,
            kwargs={
                "job": job,
                "classification_model": str(classification_model),
                "classification_model_path": classification_model_path.resolve(),
                "keypoint_model_path": keypoint_model_path.resolve(),
                "inference_options": dict(inference_options or {}),
                "save_path": job.save_path,
            },
            daemon=True,
            name=f"inference-client-live-job-{job.job_id}",
        )
        thread.start()
        return job

    def submit_client_frame(self, job_id: str, frame_bytes: bytes) -> bool:
        job = self.get_job(job_id)
        if job is None:
            return False
        if not job.accepts_client_frames:
            return False
        if job.done_event.is_set():
            return False
        return job.push_client_frame(frame_bytes)

    def stop_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None:
            return False
        job.request_stop()
        return True

    def get_job(self, job_id: str) -> Optional[InferenceStreamJob]:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def get_status(self, job_id: str) -> Optional[Dict[str, Optional[str]]]:
        job = self.get_job(job_id)
        if job is None:
            return None
        return job.snapshot()

    def stream_generator(self, job_id: str, heartbeat_interval_s: float = 10.0) -> Optional[Iterator[str]]:
        job = self.get_job(job_id)
        if job is None:
            return None

        def _iter() -> Iterator[str]:
            next_heartbeat_t = time.monotonic() + max(0.0, float(heartbeat_interval_s))
            while True:
                if job.done_event.is_set() and job.packet_queue.empty():
                    break
                try:
                    packet_json = job.packet_queue.get(timeout=0.5)
                    yield _format_sse_event("frame", packet_json)
                    next_heartbeat_t = time.monotonic() + max(0.0, float(heartbeat_interval_s))
                except queue.Empty:
                    now = time.monotonic()
                    if heartbeat_interval_s > 0.0 and now >= next_heartbeat_t:
                        payload = json.dumps({"t": time.time()}, separators=(",", ":"))
                        yield _format_sse_event("heartbeat", payload)
                        next_heartbeat_t = now + float(heartbeat_interval_s)

            snapshot = job.snapshot()
            if snapshot.get("status") == "error":
                payload = json.dumps(
                    {"type": "error", "message": snapshot.get("error") or "Inference failed."},
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                yield _format_sse_event("error", payload)
            else:
                payload_dict = {"type": "done", "save_path": snapshot.get("save_path")}
                payload = json.dumps(payload_dict, separators=(",", ":"), ensure_ascii=True)
                yield _format_sse_event("done", payload)

        return _iter()

    def _run_job(
        self,
        *,
        job: InferenceStreamJob,
        video_path: Path,
        classification_model: str,
        classification_model_path: Path,
        keypoint_model_path: Path,
        inference_options: Dict[str, Any],
        save_path: Optional[Path],
    ) -> None:
        try:
            model_key = str(classification_model).strip().lower()
            if model_key == "motionbert":
                from inference.infer_motionbert_video import run_inference_stream_packets
            else:
                from inference.inference_on_video import run_inference_stream_packets

            return_code = run_inference_stream_packets(
                video_path=video_path,
                classification_model_path=classification_model_path,
                keypoint_model_path=keypoint_model_path,
                on_packet=job.push_packet,
                save_path=save_path,
                no_display=True,
                **dict(inference_options),
            )
            if int(return_code) != 0:
                raise RuntimeError(f"Inference failed with return code {return_code}.")
            job.set_done()
        except Exception as error:
            logger.exception("Inference stream job failed: job_id=%s", job.job_id)
            job.set_error(str(error))

    def _run_live_job(
        self,
        *,
        job: InferenceStreamJob,
        camera_index: int,
        classification_model: str,
        classification_model_path: Path,
        keypoint_model_path: Path,
        inference_options: Dict[str, Any],
        save_path: Optional[Path],
    ) -> None:
        try:
            model_key = str(classification_model).strip().lower()
            if model_key == "motionbert":
                raise NotImplementedError("MotionBERT is not supported in live mode.")

            from inference.inference_on_live import run_inference_live

            return_code = run_inference_live(
                camera_index=camera_index,
                stop_event=job.stop_event,
                classification_model_path=classification_model_path,
                keypoint_model_path=keypoint_model_path,
                on_packet=job.push_packet,
                save_path=save_path,
                **dict(inference_options),
            )
            if int(return_code) != 0:
                raise RuntimeError(f"Live inference failed with return code {return_code}.")
            job.set_done()
        except Exception as error:
            logger.exception("Live inference job failed: job_id=%s", job.job_id)
            job.set_error(str(error))

    def _run_client_live_job(
        self,
        *,
        job: InferenceStreamJob,
        classification_model: str,
        classification_model_path: Path,
        keypoint_model_path: Path,
        inference_options: Dict[str, Any],
        save_path: Optional[Path],
    ) -> None:
        try:
            model_key = str(classification_model).strip().lower()
            if model_key == "motionbert":
                raise NotImplementedError("MotionBERT is not supported in live mode.")

            from inference.inference_on_live import run_inference_live_client

            return_code = run_inference_live_client(
                frame_bytes_source=job.pop_client_frame,
                stop_event=job.stop_event,
                classification_model_path=classification_model_path,
                keypoint_model_path=keypoint_model_path,
                on_packet=job.push_packet,
                save_path=save_path,
                **dict(inference_options),
            )
            if int(return_code) != 0:
                raise RuntimeError(f"Client live inference failed with return code {return_code}.")
            job.set_done()
        except Exception as error:
            logger.exception("Client live inference job failed: job_id=%s", job.job_id)
            job.set_error(str(error))
