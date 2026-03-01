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
    status: str = "running"  # running | done | error
    error: Optional[str] = None
    packet_queue: "queue.Queue[str]" = field(default_factory=lambda: queue.Queue(maxsize=4))
    last_packet: Optional[str] = None
    done_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> Dict[str, Optional[str]]:
        with self.lock:
            return {"status": self.status, "error": self.error}

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


class InferenceStreamJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, InferenceStreamJob] = {}
        self._jobs_lock = threading.Lock()

    def start_job(
        self,
        *,
        video_path: Path,
        classification_model_path: Path,
        keypoint_model_path: Path,
        inference_options: Optional[Dict[str, Any]] = None,
    ) -> InferenceStreamJob:
        job = InferenceStreamJob(job_id=uuid.uuid4().hex)
        with self._jobs_lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            kwargs={
                "job": job,
                "video_path": video_path.resolve(),
                "classification_model_path": classification_model_path.resolve(),
                "keypoint_model_path": keypoint_model_path.resolve(),
                "inference_options": dict(inference_options or {}),
            },
            daemon=True,
            name=f"inference-stream-job-{job.job_id}",
        )
        thread.start()
        return job

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
                payload = json.dumps({"type": "done"}, separators=(",", ":"), ensure_ascii=True)
                yield _format_sse_event("done", payload)

        return _iter()

    def _run_job(
        self,
        *,
        job: InferenceStreamJob,
        video_path: Path,
        classification_model_path: Path,
        keypoint_model_path: Path,
        inference_options: Dict[str, Any],
    ) -> None:
        try:
            from inference.inference_on_video import run_inference_stream_packets

            return_code = run_inference_stream_packets(
                video_path=video_path,
                classification_model_path=classification_model_path,
                keypoint_model_path=keypoint_model_path,
                on_packet=job.push_packet,
                no_display=True,
                **dict(inference_options),
            )
            if int(return_code) != 0:
                raise RuntimeError(f"Inference failed with return code {return_code}.")
            job.set_done()
        except Exception as error:
            logger.exception("Inference stream job failed: job_id=%s", job.job_id)
            job.set_error(str(error))
