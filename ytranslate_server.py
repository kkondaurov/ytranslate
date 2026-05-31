#!/usr/bin/env python3
import json
import logging
import mimetypes
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import unquote, urlparse

import ytranslate


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
CLIENT_HEADER = "X-YTranslate-Client"
CLIENT_HEADER_VALUE = "chrome-extension"
ACTIVE_JOB_STATUSES = {"queued", "running"}
MAX_HISTORY = 50
MAX_EVENTS = 200
DEFAULT_JOB_MAX_ATTEMPTS = 2
DEFAULT_JOB_RETRY_DELAY_SECONDS = 15
DEFAULT_STATE_DIR = Path.home() / "Library" / "Application Support" / "ytranslate"
DEFAULT_HISTORY_PATH = DEFAULT_STATE_DIR / "jobs.json"
FRONTEND_DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"

logger = logging.getLogger("ytranslate.server")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def history_path() -> Path:
    return Path(os.getenv("YTRANSLATE_HISTORY_PATH", str(DEFAULT_HISTORY_PATH)))


def event_record(level: str, message: str) -> Dict[str, str]:
    return {
        "at": utc_now(),
        "level": level,
        "message": message,
    }


def read_nonnegative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise RuntimeError(f"{name} must not be negative")
    return value


def job_max_attempts() -> int:
    return max(1, read_nonnegative_int_env("YTRANSLATE_JOB_MAX_ATTEMPTS", DEFAULT_JOB_MAX_ATTEMPTS))


def job_retry_delay_seconds() -> int:
    return read_nonnegative_int_env(
        "YTRANSLATE_JOB_RETRY_DELAY_SECONDS",
        DEFAULT_JOB_RETRY_DELAY_SECONDS,
    )


def is_retryable_job_error(exc: Exception) -> bool:
    message = str(exc).lower()
    permanent_markers = [
        "openai_api_key",
        "youtube_api_key",
        "could not extract video id",
        "transcripts are disabled",
        "no transcript found",
        "unsupported url",
        "must be an integer",
        "must not be negative",
    ]
    if any(marker in message for marker in permanent_markers):
        return False

    retryable_markers = [
        "api",
        "connection",
        "curl",
        "timeout",
        "timed out",
        "ssl",
        "rate limit",
        "temporarily",
        "server error",
        "pass returned",
        "response contained no text",
        "not valid json",
    ]
    return any(marker in message for marker in retryable_markers)


def parse_asr_chunk_message(message: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"(chunk-\d{3}\.mp3).*?\((\d+)/(\d+)\)", message)
    if not match:
        return None
    return {
        "chunk": match.group(1),
        "index": int(match.group(2)),
        "total": int(match.group(3)),
    }


def phase_for_message(message: str) -> tuple[str, str] | None:
    if message.startswith("Fetching metadata"):
        return "metadata", "Fetching metadata"
    if "YouTube transcript" in message or message.startswith("Checking YouTube transcript"):
        return "transcript", "Choosing transcript source"
    if "Downloading audio" in message:
        return "download", "Downloading audio"
    if "Compressing and chunking audio" in message:
        return "chunking", "Compressing and chunking audio"
    if "OpenAI ASR" in message or "ASR chunk" in message:
        return "asr", "Running OpenAI ASR"
    if "speaker" in message.lower() or "voice" in message.lower():
        return "speakers", "Resolving speakers"
    if message.startswith("Translating attributed turns"):
        return "translation", "Translating transcript"
    if message.startswith("Polishing Russian"):
        return "cleanup", "Polishing Russian text"
    if message.startswith("Adding targeted glossary"):
        return "annotation", "Adding glossary annotations"
    if message.startswith("Saved translated transcript"):
        return "render", "Writing DOCX/PDF output"
    if message.startswith("Finished generating"):
        return "done", "Finished"
    return None


def status_steps(job: "JobRecord") -> list[Dict[str, str]]:
    progress = job.progress
    asr_done = progress.get("asr_chunks_done")
    asr_total = progress.get("asr_chunks_total")
    asr_failed = progress.get("asr_failed_chunks") or []
    asr_text = ""
    if asr_total:
        asr_text = f"{asr_done or 0} / {asr_total}"
        if asr_failed:
            asr_text += f", {len(asr_failed)} failed"

    steps = [
        ("queued", "Queued", ""),
        ("metadata", "Metadata", ""),
        ("transcript", "Transcript source", ""),
        ("download", "Audio download", ""),
        ("chunking", "Audio chunking", ""),
        ("asr", "ASR chunks", asr_text),
        ("speakers", "Speaker reconciliation", ""),
        ("translation", "Translation", ""),
        ("cleanup", "Russian cleanup", ""),
        ("annotation", "Glossary annotations", ""),
        ("render", "DOCX/PDF output", ""),
    ]
    order = [key for key, _label, _detail in steps]
    phase_index = order.index(job.phase) if job.phase in order else -1
    rendered = []
    for index, (key, label, detail) in enumerate(steps):
        if job.status == "succeeded":
            state = "done"
        elif job.status == "failed" and key == job.phase:
            state = "failed"
        elif index < phase_index:
            state = "done"
        elif index == phase_index:
            state = "current"
        else:
            state = "pending"
        rendered.append({"key": key, "label": label, "state": state, "detail": detail})
    return rendered


@dataclass
class JobRecord:
    job_id: str
    url: str
    canonical_url: str
    target_language: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    docx_path: Optional[str] = None
    pdf_path: Optional[str] = None
    title: Optional[str] = None
    title_translated: Optional[str] = None
    error: Optional[str] = None
    duplicate_of: Optional[str] = None
    output_files: list[str] = field(default_factory=list)
    phase: str = "queued"
    phase_detail: str = "Queued"
    progress: Dict[str, Any] = field(default_factory=dict)
    events: list[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "JobRecord":
        known = {
            field_name: value.get(field_name)
            for field_name in cls.__dataclass_fields__
            if field_name in value
        }
        if "output_files" not in known or known["output_files"] is None:
            known["output_files"] = []
        if "progress" not in known or known["progress"] is None:
            known["progress"] = {}
        if "events" not in known or known["events"] is None:
            known["events"] = []
        return cls(**known)

    def record_event(self, level: str, message: str) -> None:
        self.events.append(event_record(level, message))
        self.events = self.events[-MAX_EVENTS:]
        self.apply_message_progress(message)

    def apply_message_progress(self, message: str) -> None:
        phase = phase_for_message(message)
        if phase:
            self.phase, self.phase_detail = phase
        if "failed" in message.lower():
            self.phase_detail = message

        chunk = parse_asr_chunk_message(message)
        if not chunk:
            return

        self.phase = "asr"
        self.phase_detail = message
        self.progress["current_chunk"] = chunk["chunk"]
        self.progress["asr_chunks_total"] = chunk["total"]
        if "failed" in message.lower():
            failed = set(self.progress.get("asr_failed_chunks") or [])
            failed.add(chunk["chunk"])
            self.progress["asr_failed_chunks"] = sorted(failed)
        elif "completed" in message or "cached" in message:
            completed = set(self.progress.get("asr_completed_chunks") or [])
            completed.add(chunk["chunk"])
            self.progress["asr_completed_chunks"] = sorted(completed)
            self.progress["asr_chunks_done"] = len(completed)
        self.progress["asr_chunks_done"] = int(self.progress.get("asr_chunks_done") or 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "target_language": self.target_language,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "docx_path": self.docx_path,
            "pdf_path": self.pdf_path,
            "title": self.title,
            "title_translated": self.title_translated,
            "error": self.error,
            "duplicate_of": self.duplicate_of,
            "output_files": list(self.output_files),
            "phase": self.phase,
            "phase_detail": self.phase_detail,
            "progress": dict(self.progress),
            "events": list(self.events),
            "steps": status_steps(self),
        }


def save_job_history(path: Path, jobs: Iterable[JobRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [job.to_dict() for job in list(jobs)[-MAX_HISTORY:]]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_job_history(path: Path) -> list[JobRecord]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    jobs = []
    for item in raw[-MAX_HISTORY:]:
        if isinstance(item, dict):
            job = JobRecord.from_dict(item)
            if job.status in ACTIVE_JOB_STATUSES:
                job.status = "failed"
                job.finished_at = job.finished_at or utc_now()
                job.error = "Server restarted before this job finished."
                job.record_event("error", job.error)
            jobs.append(job)
    return jobs


def status_payload(jobs: list[JobRecord]) -> Dict[str, Any]:
    ordered_jobs = sorted(jobs, key=lambda job: job.created_at)[-MAX_HISTORY:]
    latest = ordered_jobs[-1] if ordered_jobs else None
    counts: Dict[str, int] = {}
    for job in ordered_jobs:
        counts[job.status] = counts.get(job.status, 0) + 1

    return {
        "ok": True,
        "generated_at": utc_now(),
        "latest": latest.to_dict() if latest else None,
        "jobs": [job.to_dict() for job in reversed(ordered_jobs[-20:])],
        "counts": counts,
        "active_jobs": sum(1 for job in ordered_jobs if job.status in ACTIVE_JOB_STATUSES),
    }


def frontend_asset_path(path: str) -> Optional[Path]:
    if path == "/":
        relative_path = Path("index.html")
    else:
        relative_path = Path(unquote(path.lstrip("/")))

    candidate = (FRONTEND_DIST_DIR / relative_path).resolve()
    try:
        candidate.relative_to(FRONTEND_DIST_DIR.resolve())
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


class JobManager:
    def __init__(self, history_file: Optional[Path] = None) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._history_file = history_file or history_path()
        loaded_jobs = load_job_history(self._history_file)
        self._jobs: Dict[str, JobRecord] = {job.job_id: job for job in loaded_jobs}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self) -> None:
        self._worker.start()

    def submit(self, url: str) -> tuple[JobRecord, bool]:
        canonical_url = ytranslate.canonicalize_youtube_url(url)
        if not canonical_url:
            raise ValueError("Could not extract video ID from URL")

        target_language = ytranslate.resolve_target_language(None)
        with self._lock:
            for existing in self._jobs.values():
                if (
                    existing.canonical_url == canonical_url
                    and existing.status in ACTIVE_JOB_STATUSES
                ):
                    return existing, True

            job = JobRecord(
                job_id=uuid.uuid4().hex[:8],
                url=url,
                canonical_url=canonical_url,
                target_language=target_language,
                status="queued",
                created_at=utc_now(),
            )
            self._jobs[job.job_id] = job
            job.record_event("info", "Queued")
            self._persist_locked()
            self._queue.put(job.job_id)
            return job, False

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at)[-MAX_HISTORY:]

    def latest(self) -> Optional[JobRecord]:
        jobs = self.list_jobs()
        return jobs[-1] if jobs else None

    def _persist_locked(self) -> None:
        save_job_history(
            self._history_file,
            sorted(self._jobs.values(), key=lambda job: job.created_at),
        )

    def _update(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
            self._persist_locked()

    def _record_event(self, job_id: str, level: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.record_event(level, message)
            self._persist_locked()

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            started_at = utc_now()
            self._update(job_id, status="running", started_at=started_at, phase="metadata", phase_detail="Starting")
            job = self.get(job_id)
            if not job:
                self._queue.task_done()
                continue

            start_time = time.time()

            def log(message: str) -> None:
                logger.info("job=%s %s", job_id, message)
                self._record_event(job_id, "info", message)

            try:
                max_attempts = job_max_attempts()
                retry_delay = job_retry_delay_seconds()
                result: Optional[Dict[str, Any]] = None
                for attempt in range(1, max_attempts + 1):
                    self._update(
                        job_id,
                        status="running",
                        phase="metadata",
                        phase_detail=f"Starting attempt {attempt}/{max_attempts}",
                    )
                    if attempt > 1:
                        self._record_event(
                            job_id,
                            "info",
                            f"Retrying job attempt {attempt}/{max_attempts}",
                        )
                    try:
                        result = ytranslate.run_translation_job(
                            job.canonical_url,
                            job.target_language,
                            log=log,
                        )
                        break
                    except Exception as exc:
                        if attempt >= max_attempts or not is_retryable_job_error(exc):
                            raise
                        self._record_event(
                            job_id,
                            "error",
                            f"Attempt {attempt}/{max_attempts} failed: {exc}",
                        )
                        if retry_delay:
                            time.sleep(retry_delay)

                if result is None:
                    raise RuntimeError("Job retry loop finished without a result")

                duration = time.time() - start_time
                self._update(
                    job_id,
                    status="succeeded",
                    finished_at=utc_now(),
                    error=None,
                    phase="done",
                    phase_detail="Finished",
                    docx_path=result.get("docx_path"),
                    pdf_path=result.get("pdf_path"),
                    title=result.get("title"),
                    title_translated=result.get("title_translated"),
                    output_files=result.get("output_files", []),
                )
                self._record_event(job_id, "info", f"Completed in {duration:.1f}s")
                logger.info("job=%s completed in %.1fs", job_id, duration)
            except Exception as exc:
                duration = time.time() - start_time
                self._record_event(job_id, "error", str(exc))
                self._update(
                    job_id,
                    status="failed",
                    finished_at=utc_now(),
                    error=str(exc),
                )
                logger.exception("job=%s failed after %.1fs: %s", job_id, duration, exc)
            finally:
                self._queue.task_done()


def make_handler(job_manager: JobManager):
    class RequestHandler(BaseHTTPRequestHandler):
        server_version = "ytranslate-server/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-YTranslate-Client")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, status_code: int, html_body: str) -> None:
            body = html_body.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            body = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if path.name == "index.html":
                self.send_header("Cache-Control", "no-store")
            else:
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self._send_json(200, {"ok": True})

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                index_path = frontend_asset_path(path)
                if index_path:
                    self._send_file(index_path)
                else:
                    self._send_html(
                        503,
                        "<!doctype html><title>ytranslate status</title>"
                        "<h1>ytranslate status frontend is not built</h1>"
                        "<p>Run <code>npm --prefix frontend run build</code>.</p>",
                    )
                return

            if path == "/health":
                self._send_json(200, {"ok": True, "status": "healthy"})
                return

            if path == "/api/status":
                self._send_json(200, status_payload(job_manager.list_jobs()))
                return

            if path == "/jobs":
                self._send_json(
                    200,
                    {"ok": True, "jobs": [job.to_dict() for job in job_manager.list_jobs()]},
                )
                return

            if path == "/jobs/latest":
                job = job_manager.latest()
                self._send_json(200, {"ok": True, "job": job.to_dict() if job else None})
                return

            if path.startswith("/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                job = job_manager.get(job_id)
                if not job:
                    self._send_json(404, {"ok": False, "error": "Job not found"})
                    return
                self._send_json(200, {"ok": True, "job": job.to_dict()})
                return

            asset_path = frontend_asset_path(path)
            if asset_path:
                self._send_file(asset_path)
                return

            self._send_json(404, {"ok": False, "error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/jobs":
                self._send_json(404, {"ok": False, "error": "Not found"})
                return

            if self.headers.get(CLIENT_HEADER) != CLIENT_HEADER_VALUE:
                self._send_json(403, {"ok": False, "error": "Forbidden"})
                return

            length_header = self.headers.get("Content-Length")
            if not length_header:
                self._send_json(400, {"ok": False, "error": "Missing request body"})
                return

            try:
                length = int(length_header)
                payload = json.loads(self.rfile.read(length))
            except (ValueError, json.JSONDecodeError):
                self._send_json(400, {"ok": False, "error": "Invalid JSON body"})
                return

            url = (payload.get("url") or "").strip()
            if not url:
                self._send_json(400, {"ok": False, "error": "Missing URL"})
                return

            try:
                job, duplicate = job_manager.submit(url)
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return

            if duplicate:
                logger.info(
                    "duplicate request for %s returned existing job=%s",
                    job.canonical_url,
                    job.job_id,
                )
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "accepted": False,
                        "duplicate": True,
                        "job": job.to_dict(),
                    },
                )
                return

            logger.info(
                "received request job=%s url=%s target=%s",
                job.job_id,
                job.canonical_url,
                job.target_language,
            )
            self._send_json(
                202,
                {
                    "ok": True,
                    "accepted": True,
                    "duplicate": False,
                    "job": job.to_dict(),
                },
            )

    return RequestHandler


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    ytranslate.load_project_env()
    logger.info(
        "starting ytranslate server on http://%s:%s with default target language=%s",
        SERVER_HOST,
        SERVER_PORT,
        ytranslate.resolve_target_language(None),
    )

    job_manager = JobManager()
    job_manager.start()
    server = ThreadingHTTPServer((SERVER_HOST, SERVER_PORT), make_handler(job_manager))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down server")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
