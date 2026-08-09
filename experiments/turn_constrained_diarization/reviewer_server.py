#!/usr/bin/env python3
"""Serve the diarization reviewer with byte-range support for audio seeking."""

from __future__ import annotations

import argparse
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


class RangeRequestHandler(SimpleHTTPRequestHandler):
    server_version = "DiarizationReviewer/1.0"

    def __init__(self, *args, **kwargs):
        self._byte_range: Optional[Tuple[int, int]] = None
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self) -> Optional[BinaryIO]:
        path = self.translate_path(self.path)
        if os.path.isdir(path) or not os.path.isfile(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match:
            self.send_error(416, "Invalid byte range")
            return None
        file_size = os.path.getsize(path)
        start_text, end_text = match.groups()
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        elif end_text:
            suffix_length = int(end_text)
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            self.send_error(416, "Invalid byte range")
            return None
        if start >= file_size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return None
        end = min(end, file_size - 1)
        content_type = self.guess_type(path)
        handle = open(path, "rb")
        self.send_response(206)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(os.fstat(handle.fileno()).st_mtime))
        self.end_headers()
        handle.seek(start)
        self._byte_range = (start, end)
        return handle

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        if self._byte_range is None:
            super().copyfile(source, outputfile)
            return
        start, end = self._byte_range
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(128 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RangeRequestHandler)
    print(f"Reviewer: http://{args.host}:{args.port}/experiments/turn_constrained_diarization/reviewer/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
