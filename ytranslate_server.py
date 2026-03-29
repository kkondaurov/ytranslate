#!/usr/bin/env python3
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import ytranslate


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
CLIENT_HEADER = "X-YTranslate-Client"
CLIENT_HEADER_VALUE = "chrome-extension"
ACTIVE_JOB_STATUSES = {"queued", "running"}

logger = logging.getLogger("ytranslate.server")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        }


class JobManager:
    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: Dict[str, JobRecord] = {}
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
            self._queue.put(job.job_id)
            return job, False

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def _update(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            started_at = utc_now()
            self._update(job_id, status="running", started_at=started_at)
            job = self.get(job_id)
            if not job:
                self._queue.task_done()
                continue

            start_time = time.time()

            def log(message: str) -> None:
                logger.info("job=%s %s", job_id, message)

            try:
                result = ytranslate.run_translation_job(
                    job.canonical_url,
                    job.target_language,
                    log=log,
                )
                duration = time.time() - start_time
                self._update(
                    job_id,
                    status="succeeded",
                    finished_at=utc_now(),
                    docx_path=result.get("docx_path"),
                    pdf_path=result.get("pdf_path"),
                    title=result.get("title"),
                    title_translated=result.get("title_translated"),
                    output_files=result.get("output_files", []),
                )
                logger.info("job=%s completed in %.1fs", job_id, duration)
            except Exception as exc:
                duration = time.time() - start_time
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

        def do_OPTIONS(self) -> None:
            self._send_json(200, {"ok": True})

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"ok": True, "status": "healthy"})
                return

            if self.path.startswith("/jobs/"):
                job_id = self.path.rsplit("/", 1)[-1]
                job = job_manager.get(job_id)
                if not job:
                    self._send_json(404, {"ok": False, "error": "Job not found"})
                    return
                self._send_json(200, {"ok": True, "job": job.to_dict()})
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
