#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "asr_compare" / "output"
DEFAULT_CHUNK_SECONDS = 300
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
OPENAI_TRANSCRIBE_MODEL = "gpt-4o-transcribe-diarize"
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"


@dataclass(frozen=True)
class Episode:
    label: str
    url: str
    video_id: str
    title: str = ""


EPISODES = [
    Episode(
        label="all-in 1",
        url="https://youtu.be/HGbA6ze0_3M?si=d89VSal3nL8Su7o_",
        video_id="HGbA6ze0_3M",
    ),
    Episode(
        label="all-in 2",
        url="https://youtu.be/jJRAvZNGUvI?si=lGdFFt69llbMS4a_",
        video_id="jJRAvZNGUvI",
    ),
    Episode(
        label="easier podcast",
        url="https://youtu.be/DhD1zZ8w8Mw?si=lbIEwCxBa2D8JyL0",
        video_id="DhD1zZ8w8Mw",
    ),
]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def extract_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    query_id = parse_qs(parsed.query).get("v", [None])[0]
    if query_id:
        return query_id
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        return parsed.path.strip("/").split("/")[0] or None
    if "youtube.com" in host:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            return parts[1]
    return None


def safe_slug(text: str, limit: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:limit].strip("-") or "untitled"


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(command: List[str], cwd: Optional[Path] = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is required for the experiment") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def fetch_youtube_metadata(episode: Episode, episode_dir: Path) -> Episode:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.exists():
        raw = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--skip-download",
                "--dump-json",
                episode.url,
            ],
            text=True,
        )
        write_json(metadata_path, json.loads(raw))
    metadata = read_json(metadata_path)
    return Episode(
        label=episode.label,
        url=episode.url,
        video_id=episode.video_id,
        title=metadata.get("title") or episode.title or episode.video_id,
    )


def fetch_youtube_transcript(episode: Episode, episode_dir: Path) -> None:
    transcript_path = episode_dir / "youtube_transcript.json"
    if transcript_path.exists():
        return

    transcript_list = list_transcripts_for_video(episode.video_id)
    transcripts = list(transcript_list)
    if not transcripts:
        raise RuntimeError(f"No YouTube transcript found for {episode.video_id}")

    transcript = next((item for item in transcripts if not item.is_generated), None)
    if transcript is None:
        transcript = transcripts[0]

    fetched = transcript.fetch()
    segments = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
    write_json(
        transcript_path,
        {
            "language_code": transcript.language_code,
            "language": transcript.language,
            "is_generated": transcript.is_generated,
            "segments": segments,
        },
    )


def list_transcripts_for_video(video_id: str, api_cls: Optional[Any] = None) -> Any:
    if api_cls is None:
        from youtube_transcript_api import YouTubeTranscriptApi

        api_cls = YouTubeTranscriptApi
    if hasattr(api_cls, "list_transcripts"):
        return api_cls.list_transcripts(video_id)
    api = api_cls()
    if hasattr(api, "list"):
        return api.list(video_id)
    if hasattr(api, "list_transcripts"):
        return api.list_transcripts(video_id)
    raise RuntimeError("Unsupported youtube-transcript-api version")


def download_audio(episode: Episode, episode_dir: Path) -> Path:
    audio_dir = episode_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(audio_dir.glob("source.*"))
    if existing:
        return existing[0]

    output_template = str(audio_dir / "source.%(ext)s")
    run_command(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "ba[ext=m4a]/ba[ext=webm]/ba/bestaudio",
            "-o",
            output_template,
            "--no-playlist",
            episode.url,
        ]
    )
    existing = sorted(audio_dir.glob("source.*"))
    if not existing:
        raise RuntimeError(f"Audio download did not produce a source file for {episode.video_id}")
    return existing[0]


def chunk_output_path(base_dir: Path, video_id: str, index: int) -> Path:
    return base_dir / video_id / "chunks" / f"chunk-{index:03d}.mp3"


def transcode_and_chunk_audio(
    source_audio: Path,
    episode: Episode,
    output_dir: Path,
    chunk_seconds: int,
) -> List[Path]:
    chunk_dir = output_dir / episode.video_id / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(chunk_dir.glob("chunk-*.mp3"))
    if existing:
        return existing

    ffmpeg = get_ffmpeg_path()
    output_pattern = str(chunk_dir / "chunk-%03d.mp3")
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(source_audio),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            output_pattern,
        ]
    )

    chunks = sorted(chunk_dir.glob("chunk-*.mp3"))
    oversized = [chunk for chunk in chunks if chunk.stat().st_size > MAX_UPLOAD_BYTES]
    if oversized:
        names = ", ".join(chunk.name for chunk in oversized)
        raise RuntimeError(f"Chunk files exceed the upload guardrail: {names}")
    return chunks


def transcribe_chunk(
    chunk_path: Path,
    api_key: str,
    timeout_seconds: int,
    attempts: int = 8,
) -> Dict[str, Any]:
    import requests

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": OPENAI_TRANSCRIBE_MODEL,
        "response_format": "diarized_json",
        "chunking_strategy": "auto",
    }
    for attempt in range(1, attempts + 1):
        try:
            with chunk_path.open("rb") as audio_file:
                files = {"file": (chunk_path.name, audio_file, "audio/mpeg")}
                response = requests.post(
                    OPENAI_TRANSCRIBE_URL,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=timeout_seconds,
                )
        except requests.RequestException as exc:
            if attempt == attempts:
                raise RuntimeError(
                    f"OpenAI transcription request failed after retries for {chunk_path.name}: {exc}"
                ) from exc
            time.sleep(min(60, 5 * attempt))
            continue
        if response.status_code < 500 and response.status_code != 429:
            if response.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI transcription failed for {chunk_path.name}: "
                    f"{response.status_code} {response.text[:1000]}"
                )
            return response.json()
        if attempt == attempts:
            raise RuntimeError(
                f"OpenAI transcription failed after retries for {chunk_path.name}: "
                f"{response.status_code} {response.text[:1000]}"
            )
        time.sleep(min(60, 5 * attempt))
    raise RuntimeError("Unreachable transcription retry state")


def extract_diarized_segments(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = response.get("segments") or []
    normalized = []
    if segments:
        for segment in segments:
            normalized.append(
                {
                    "start": float(segment.get("start") or 0),
                    "end": float(segment.get("end") or segment.get("start") or 0),
                    "speaker": str(segment.get("speaker") or "speaker"),
                    "text": str(segment.get("text") or "").strip(),
                }
            )
        return normalized

    text = str(response.get("text") or "").strip()
    if text:
        normalized.append({"start": 0.0, "end": 0.0, "speaker": "speaker", "text": text})
    return normalized


def transcribe_episode(
    episode: Episode,
    output_dir: Path,
    chunk_paths: List[Path],
    chunk_offsets: List[float],
    timeout_seconds: int,
    jobs: int,
) -> None:
    result_path = output_dir / episode.video_id / "openai_diarized.json"
    if result_path.exists():
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    chunk_jobs = []
    for index, chunk_path in enumerate(chunk_paths):
        raw_path = output_dir / episode.video_id / "openai_chunks" / f"{chunk_path.stem}.json"
        offset_seconds = chunk_offsets[index]
        chunk_jobs.append((index, chunk_path, raw_path, offset_seconds))

    raw_by_index: Dict[int, Dict[str, Any]] = {}
    pending = []
    for index, chunk_path, raw_path, _offset_seconds in chunk_jobs:
        if raw_path.exists():
            raw_by_index[index] = read_json(raw_path)
        else:
            pending.append((index, chunk_path, raw_path))

    def run_one(item: tuple[int, Path, Path]) -> tuple[int, Dict[str, Any]]:
        index, chunk_path, raw_path = item
        print(f"Transcribing {episode.video_id} {chunk_path.name}", flush=True)
        raw = transcribe_chunk(chunk_path, api_key, timeout_seconds=timeout_seconds)
        write_json(raw_path, raw)
        return index, raw

    if pending:
        worker_count = max(1, min(jobs, len(pending)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_one, item) for item in pending]
            for future in as_completed(futures):
                index, raw = future.result()
                raw_by_index[index] = raw
                print(f"Finished {episode.video_id} {chunk_paths[index].name}", flush=True)

    chunk_results = []
    for index, chunk_path, _raw_path, offset_seconds in chunk_jobs:
        raw = raw_by_index[index]
        chunk_results.append(
            {
                "chunk": chunk_path.name,
                "offset_seconds": offset_seconds,
                "raw": raw,
                "segments": extract_diarized_segments(raw),
            }
        )

    merged = merge_diarized_segments(chunk_results)
    write_json(
        result_path,
        {
            "model": OPENAI_TRANSCRIBE_MODEL,
            "chunk_offsets": chunk_offsets,
            "chunks": [
                {
                    "chunk": item["chunk"],
                    "offset_seconds": item["offset_seconds"],
                    "segment_count": len(item["segments"]),
                }
                for item in chunk_results
            ],
            "segments": merged,
        },
    )


def merge_diarized_segments(chunk_results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for chunk in chunk_results:
        offset = float(chunk.get("offset_seconds") or 0)
        for segment in chunk.get("segments", []):
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            start = float(segment.get("start") or 0) + offset
            end = float(segment.get("end") or segment.get("start") or 0) + offset
            merged.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speaker": str(segment.get("speaker") or "speaker"),
                    "text": text,
                }
            )
    return merged


def probe_audio_duration_seconds(audio_path: Path) -> float:
    ffmpeg = get_ffmpeg_path()
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(audio_path), "-f", "null", "-"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", completed.stderr)
    if not match:
        raise RuntimeError(f"Unable to parse duration for {audio_path}")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def build_chunk_offsets(durations: List[float]) -> List[float]:
    offsets: List[float] = []
    current = 0.0
    for duration in durations:
        offsets.append(round(current, 3))
        current += float(duration)
    return offsets


def render_youtube_segments(segments: List[Dict[str, Any]]) -> str:
    blocks = []
    for segment in segments:
        start = float(segment.get("start") or 0)
        duration = float(segment.get("duration") or 0)
        end = start + duration
        text = html.escape(str(segment.get("text") or "").strip())
        if not text:
            continue
        blocks.append(
            f'<div class="turn"><div class="stamp">{format_timestamp(start)}'
            f' - {format_timestamp(end)}</div><p>{text}</p></div>'
        )
    return "\n".join(blocks)


def render_openai_segments(segments: List[Dict[str, Any]]) -> str:
    blocks = []
    for segment in segments:
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        speaker = html.escape(str(segment.get("speaker") or "speaker"))
        text = html.escape(str(segment.get("text") or "").strip())
        if not text:
            continue
        blocks.append(
            f'<div class="turn"><div class="stamp">{format_timestamp(start)}'
            f' - {format_timestamp(end)} <span>{speaker}</span></div><p>{text}</p></div>'
        )
    return "\n".join(blocks)


def render_report_html(episodes: List[Episode], output_dir: Path) -> str:
    sections = []
    for episode in episodes:
        episode_dir = output_dir / episode.video_id
        youtube = read_json(episode_dir / "youtube_transcript.json")
        openai_result = read_json(episode_dir / "openai_diarized.json")
        youtube_segments = youtube.get("segments") or []
        openai_segments = openai_result.get("segments") or []
        sections.append(
            f"""
            <section class="episode" id="{html.escape(episode.video_id)}">
              <header>
                <div>
                  <p>{html.escape(episode.label)}</p>
                  <h2>{html.escape(episode.title or episode.video_id)}</h2>
                  <a href="{html.escape(episode.url)}">{html.escape(episode.url)}</a>
                </div>
                <dl>
                  <div><dt>YouTube segments</dt><dd>{len(youtube_segments)}</dd></div>
                  <div><dt>OpenAI segments</dt><dd>{len(openai_segments)}</dd></div>
                  <div><dt>ASR model</dt><dd>{html.escape(openai_result.get("model") or OPENAI_TRANSCRIBE_MODEL)}</dd></div>
                </dl>
              </header>
              <div class="comparison-grid">
                <article>
                  <h3>YouTube transcript</h3>
                  <div class="scroll-pane">{render_youtube_segments(youtube_segments)}</div>
                </article>
                <article>
                  <h3>OpenAI diarized ASR</h3>
                  <div class="scroll-pane">{render_openai_segments(openai_segments)}</div>
                </article>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YouTube vs OpenAI ASR comparison</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f8;
      --surface: #ffffff;
      --text: #172026;
      --muted: #5d6872;
      --line: #d7dde3;
      --accent: #0f766e;
      --youtube: #b91c1c;
      --openai: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1500px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    .page-header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .page-header p {{
      max-width: 820px;
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }}
    nav {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    nav a {{
      color: var(--text);
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 10px;
      background: var(--surface);
      font-size: 13px;
    }}
    .episode {{
      margin-top: 22px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .episode > header {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
    }}
    .episode header p {{
      margin: 0 0 6px;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 12px;
      font-weight: 700;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .episode a {{
      color: var(--muted);
      font-size: 13px;
      word-break: break-all;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(3, minmax(120px, 1fr));
      gap: 10px;
      margin: 0;
      min-width: 430px;
    }}
    dt {{
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    dd {{
      margin: 0;
      font-size: 16px;
      font-weight: 700;
    }}
    .comparison-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      min-height: 720px;
    }}
    article {{
      min-width: 0;
      display: flex;
      flex-direction: column;
    }}
    article + article {{
      border-left: 1px solid var(--line);
    }}
    h3 {{
      position: sticky;
      top: 0;
      z-index: 1;
      margin: 0;
      padding: 12px 16px;
      background: #fbfcfd;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      letter-spacing: 0;
    }}
    article:first-child h3 {{ color: var(--youtube); }}
    article:last-child h3 {{ color: var(--openai); }}
    .scroll-pane {{
      height: 720px;
      overflow: auto;
      padding: 12px 16px 18px;
      scroll-behavior: smooth;
    }}
    .turn {{
      padding: 9px 0 11px;
      border-bottom: 1px solid #edf0f2;
    }}
    .stamp {{
      margin-bottom: 4px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      font-weight: 700;
    }}
    .stamp span {{
      color: var(--openai);
      margin-left: 8px;
    }}
    .turn p {{
      margin: 0;
      font-size: 15px;
      line-height: 1.5;
      white-space: pre-wrap;
    }}
    @media (max-width: 900px) {{
      main {{ width: min(100vw - 20px, 760px); }}
      .page-header,
      .episode > header {{
        display: block;
      }}
      nav {{ justify-content: flex-start; margin-top: 14px; }}
      dl {{
        min-width: 0;
        grid-template-columns: 1fr;
        margin-top: 14px;
      }}
      .comparison-grid {{
        grid-template-columns: 1fr;
      }}
      article + article {{
        border-left: 0;
        border-top: 1px solid var(--line);
      }}
      .scroll-pane {{
        height: 560px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="page-header">
      <div>
        <h1>YouTube vs OpenAI ASR</h1>
        <p>Side-by-side transcript comparison. The left column is the YouTube caption track; the right column is OpenAI diarized ASR from downloaded audio chunks.</p>
      </div>
      <nav>
        {''.join(f'<a href="#{html.escape(ep.video_id)}">{html.escape(ep.label)}</a>' for ep in episodes)}
      </nav>
    </div>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def build_episode(
    episode: Episode,
    output_dir: Path,
    chunk_seconds: int,
    timeout_seconds: int,
    jobs: int,
) -> Episode:
    episode_dir = output_dir / episode.video_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode = fetch_youtube_metadata(episode, episode_dir)
    fetch_youtube_transcript(episode, episode_dir)
    source_audio = download_audio(episode, episode_dir)
    chunks = transcode_and_chunk_audio(source_audio, episode, output_dir, chunk_seconds)
    chunk_offsets = build_chunk_offsets([probe_audio_duration_seconds(chunk) for chunk in chunks])
    transcribe_episode(
        episode,
        output_dir,
        chunks,
        chunk_offsets=chunk_offsets,
        timeout_seconds=timeout_seconds,
        jobs=jobs,
    )
    write_json(episode_dir / "episode.json", episode.__dict__)
    return episode


def load_episode_with_title(episode: Episode, output_dir: Path) -> Episode:
    episode_path = output_dir / episode.video_id / "episode.json"
    if episode_path.exists():
        data = read_json(episode_path)
        return Episode(**data)
    return episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YouTube captions with OpenAI diarized ASR.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for episode in EPISODES:
        video_id = extract_video_id(episode.url)
        if video_id != episode.video_id:
            raise RuntimeError(f"Video ID mismatch for {episode.url}: {video_id} != {episode.video_id}")
        if args.report_only:
            episodes.append(load_episode_with_title(episode, output_dir))
        else:
            episodes.append(
                build_episode(
                    episode,
                    output_dir,
                    chunk_seconds=args.chunk_seconds,
                    timeout_seconds=args.timeout_seconds,
                    jobs=args.jobs,
                )
            )

    html_text = render_report_html(episodes, output_dir)
    report_path = output_dir / "youtube-vs-openai-asr.html"
    report_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
