#!/usr/bin/env python3
import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ytranslate  # noqa: E402


DEFAULT_VIDEO_ID = "HGbA6ze0_3M"
DEFAULT_URL = f"https://youtu.be/{DEFAULT_VIDEO_ID}"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def default_cache_dir(video_id: str) -> Path:
    return Path(ytranslate.CACHE_DIR) / ytranslate.sanitize_filename(video_id)


def default_asr_json(video_id: str) -> Path:
    cache_dir = default_cache_dir(video_id)
    return cache_dir / f"openai-asr-{ytranslate.OPENAI_ASR_MODEL}-{ytranslate.ASR_CHUNK_SECONDS}s.json"


def default_audio_path(video_id: str) -> Path:
    audio_dir = default_cache_dir(video_id) / "audio"
    for candidate in sorted(audio_dir.glob("source.*")):
        if not candidate.name.endswith(".part"):
            return candidate
    return audio_dir / "source.m4a"


def speaker_label_from_local(local_speaker: str) -> str:
    label = str(local_speaker or "speaker")
    if label.lower().startswith("speaker"):
        return label
    return f"Speaker {label}"


def build_mapping_lookup(speaker_mapping: Optional[Dict[str, Any]]) -> Dict[Any, Dict[str, str]]:
    if not speaker_mapping:
        return {}
    speakers_by_id = {
        speaker.get("id"): speaker
        for speaker in speaker_mapping.get("speakers", [])
        if speaker.get("id")
    }
    lookup: Dict[Any, Dict[str, str]] = {}
    for item in speaker_mapping.get("local_speakers", []):
        speaker_id = item.get("speaker_id")
        speaker = speakers_by_id.get(speaker_id, {})
        chunk_index = int(item.get("chunk_index") or 0)
        local_speaker = str(item.get("local_speaker") or "speaker")
        lookup[(chunk_index, local_speaker)] = {
            "id": speaker_id or ytranslate.speaker_id_from_label(local_speaker),
            "label": speaker.get("label_short") or speaker_id or speaker_label_from_local(local_speaker),
            "label_full": speaker.get("label_full") or speaker.get("label_short") or speaker_id or "",
        }
    return lookup


def build_audit_rows(
    asr_segments: List[Dict[str, Any]],
    speaker_mapping: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    mapping_lookup = build_mapping_lookup(speaker_mapping)
    rows: List[Dict[str, Any]] = []
    for index, segment in enumerate(asr_segments, 1):
        chunk_index = int(segment.get("chunk_index") or 0)
        local_speaker = str(segment.get("local_speaker") or segment.get("speaker") or "speaker")
        mapped = mapping_lookup.get((chunk_index, local_speaker))
        if mapped:
            speaker_id = mapped["id"]
            label = mapped["label"]
        else:
            speaker_id = ytranslate.speaker_id_from_label(local_speaker)
            label = speaker_label_from_local(local_speaker)
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        rows.append(
            {
                "index": index,
                "chunk_index": chunk_index,
                "local_speaker": local_speaker,
                "mapped_speaker_id": speaker_id,
                "mapped_speaker_label": label,
                "start": round(start, 3),
                "end": round(end, 3),
                "start_timecode": ytranslate.format_timecode(start),
                "end_timecode": ytranslate.format_timecode(end),
                "text": ytranslate.clean_segment_text(segment.get("text", "")),
            }
        )
    return rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_chunk: Dict[int, Dict[str, int]] = {}
    by_mapped: Dict[str, int] = {}
    for row in rows:
        chunk_index = row["chunk_index"]
        local_speaker = row["local_speaker"]
        mapped_label = row["mapped_speaker_label"]
        by_chunk.setdefault(chunk_index, {})
        by_chunk[chunk_index][local_speaker] = by_chunk[chunk_index].get(local_speaker, 0) + 1
        by_mapped[mapped_label] = by_mapped.get(mapped_label, 0) + 1
    return {
        "by_chunk": by_chunk,
        "by_mapped": by_mapped,
    }


def render_count_table(title: str, counts: Dict[Any, Any]) -> str:
    rows = [f"<h2>{html.escape(title)}</h2>", "<table class=\"counts\"><tbody>"]
    for key, value in counts.items():
        if isinstance(value, dict):
            rendered_value = ", ".join(
                f"{html.escape(str(k))}: {int(v)}"
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            )
        else:
            rendered_value = str(value)
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(rendered_value)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def render_audit_html(
    title: str,
    video_id: str,
    source_url: str,
    audio_path: Path,
    rows: List[Dict[str, Any]],
    speaker_mapping: Optional[Dict[str, Any]],
    asr_summary: Dict[str, Any],
) -> str:
    abs_audio_path = audio_path.expanduser().resolve()
    audio_src = abs_audio_path.as_uri()
    speaker_options = sorted({row["mapped_speaker_label"] for row in rows})
    local_options = sorted({f"{row['chunk_index']}:{row['local_speaker']}" for row in rows})
    summary = summarize_rows(rows)

    table_rows = []
    for row in rows:
        local_key = f"{row['chunk_index']}:{row['local_speaker']}"
        table_rows.append(
            "<tr "
            f"data-chunk=\"{row['chunk_index']}\" "
            f"data-local=\"{html.escape(local_key)}\" "
            f"data-mapped=\"{html.escape(row['mapped_speaker_label'])}\">"
            f"<td>{row['index']}</td>"
            f"<td><button type=\"button\" onclick=\"seekTo({row['start']:.3f})\">{html.escape(row['start_timecode'])}</button></td>"
            f"<td>{html.escape(row['end_timecode'])}</td>"
            f"<td>{row['chunk_index']}</td>"
            f"<td>{html.escape(row['local_speaker'])}</td>"
            f"<td>{html.escape(row['mapped_speaker_label'])}</td>"
            f"<td class=\"text\">{html.escape(row['text'])}</td>"
            "</tr>"
        )

    speaker_option_html = "\n".join(
        f"<option value=\"{html.escape(option)}\">{html.escape(option)}</option>"
        for option in speaker_options
    )
    local_option_html = "\n".join(
        f"<option value=\"{html.escape(option)}\">{html.escape(option)}</option>"
        for option in local_options
    )
    mapping_json = html.escape(json.dumps(speaker_mapping or {}, ensure_ascii=False, indent=2))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - Speaker Audit</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #fafafa;
      --panel: #ffffff;
      --text: #1f2328;
      --muted: #667085;
      --border: #d0d7de;
      --accent: #0969da;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(250, 250, 250, 0.96);
      border-bottom: 1px solid var(--border);
      padding: 16px 20px;
    }}
    h1 {{
      font-size: 20px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 15px;
      margin: 20px 0 8px;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin-bottom: 12px;
    }}
    audio {{
      width: 100%;
      max-width: 900px;
      display: block;
      margin: 10px 0;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
      margin-top: 12px;
    }}
    label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    select, input {{
      min-width: 170px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 9px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
    }}
    main {{
      padding: 18px 20px 40px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 177px;
      background: #f6f8fa;
      z-index: 5;
      color: #344054;
      font-size: 12px;
    }}
    td.text {{
      max-width: 860px;
    }}
    button {{
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 5px 8px;
      background: #f6f8fa;
      color: var(--accent);
      cursor: pointer;
      font: inherit;
      white-space: nowrap;
    }}
    pre {{
      overflow: auto;
      border: 1px solid var(--border);
      background: var(--panel);
      padding: 12px;
      border-radius: 6px;
      max-height: 360px;
    }}
    .counts th {{
      position: static;
      width: 160px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="meta">
      <span>Video: <a href="{html.escape(source_url)}">{html.escape(video_id)}</a></span>
      <span>ASR model: {html.escape(str(asr_summary.get("model", "")))}</span>
      <span>Chunk seconds: {html.escape(str(asr_summary.get("chunk_seconds", "")))}</span>
      <span>Segments: {len(rows)}</span>
      <span>Audio: {html.escape(str(abs_audio_path))}</span>
    </div>
    <audio id="audio" controls src="{html.escape(audio_src)}"></audio>
    <div class="controls">
      <label>Mapped speaker
        <select id="speakerFilter" onchange="applyFilters()">
          <option value="">All mapped speakers</option>
          {speaker_option_html}
        </select>
      </label>
      <label>Chunk/local speaker
        <select id="localFilter" onchange="applyFilters()">
          <option value="">All local speakers</option>
          {local_option_html}
        </select>
      </label>
      <label>Text search
        <input id="textFilter" type="search" placeholder="Search source text" oninput="applyFilters()">
      </label>
    </div>
  </header>
  <main>
    {render_count_table("Local ASR Speaker Counts By Chunk", summary["by_chunk"])}
    {render_count_table("Mapped Speaker Counts", summary["by_mapped"])}
    <h2>Segments</h2>
    <table id="segments">
      <thead>
        <tr>
          <th>#</th>
          <th>Start</th>
          <th>End</th>
          <th>Chunk</th>
          <th>Raw ASR local speaker</th>
          <th>Mapped final speaker</th>
          <th>Source text</th>
        </tr>
      </thead>
      <tbody>
        {"".join(table_rows)}
      </tbody>
    </table>
    <h2>Speaker Mapping JSON</h2>
    <pre>{mapping_json}</pre>
  </main>
  <script>
    function seekTo(seconds) {{
      const audio = document.getElementById('audio');
      audio.currentTime = seconds;
      audio.play();
    }}
    function applyFilters() {{
      const speaker = document.getElementById('speakerFilter').value;
      const local = document.getElementById('localFilter').value;
      const text = document.getElementById('textFilter').value.toLowerCase();
      for (const row of document.querySelectorAll('#segments tbody tr')) {{
        const matchesSpeaker = !speaker || row.dataset.mapped === speaker;
        const matchesLocal = !local || row.dataset.local === local;
        const matchesText = !text || row.textContent.toLowerCase().includes(text);
        row.style.display = matchesSpeaker && matchesLocal && matchesText ? '' : 'none';
      }}
    }}
  </script>
</body>
</html>
"""


def load_or_create_metadata(video_id: str, output_dir: Path) -> Dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        return read_json(metadata_path)
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_key:
        return {"title": video_id, "description": "", "defaultLanguage": None, "defaultAudioLanguage": None}
    metadata = ytranslate.fetch_video_metadata(video_id, youtube_key)
    write_json(metadata_path, metadata)
    return metadata


def load_or_create_mapping(
    url: str,
    video_id: str,
    output_dir: Path,
    metadata: Dict[str, Any],
    segments: List[Dict[str, Any]],
    run_mapping: bool,
) -> Optional[Dict[str, Any]]:
    mapping_path = output_dir / "speaker-mapping.json"
    overrides = ytranslate.load_speaker_mapping_overrides(
        video_id,
        extra_paths=[str(output_dir / ytranslate.SPEAKER_OVERRIDES_FILENAME)],
    )
    if mapping_path.exists():
        mapping = read_json(mapping_path)
        mapping = ytranslate.apply_chunk_boundary_speaker_continuity(mapping, segments)
        if overrides:
            mapping = ytranslate.apply_speaker_mapping_overrides(mapping, overrides)
            write_json(output_dir / "speaker-mapping-effective.json", mapping)
        return mapping
    if not run_mapping:
        return None
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return None
    debug_sink: List[Dict[str, Any]] = []
    client = OpenAI(api_key=openai_key, timeout=ytranslate.OPENAI_TIMEOUT_SECONDS)
    model = os.getenv("OPENAI_MODEL", ytranslate.DEFAULT_MODEL)
    mapping = ytranslate.assign_global_speakers_for_diarized_segments(
        client,
        model,
        url,
        metadata.get("title") or video_id,
        metadata.get("description") or "",
        segments,
        metadata.get("defaultAudioLanguage") or metadata.get("defaultLanguage"),
        debug_sink=debug_sink,
    )
    write_json(mapping_path, mapping)
    mapping = ytranslate.apply_chunk_boundary_speaker_continuity(mapping, segments)
    if overrides:
        mapping = ytranslate.apply_speaker_mapping_overrides(mapping, overrides)
        write_json(output_dir / "speaker-mapping-effective.json", mapping)
    if debug_sink:
        write_json(output_dir / "speaker-mapping-debug.json", debug_sink[0])
    return mapping


def build_report(
    video_id: str,
    url: str,
    asr_json_path: Path,
    audio_path: Path,
    output_dir: Path,
    run_mapping: bool,
) -> Path:
    ytranslate.load_project_env()
    output_dir.mkdir(parents=True, exist_ok=True)
    asr_data = read_json(asr_json_path)
    metadata = load_or_create_metadata(video_id, output_dir)
    segments = asr_data.get("segments", [])
    mapping = load_or_create_mapping(url, video_id, output_dir, metadata, segments, run_mapping)
    rows = build_audit_rows(segments, mapping)
    write_json(output_dir / "audit-rows.json", rows)
    html_text = render_audit_html(
        title=metadata.get("title") or video_id,
        video_id=video_id,
        source_url=url,
        audio_path=audio_path,
        rows=rows,
        speaker_mapping=mapping,
        asr_summary={
            "model": asr_data.get("model"),
            "chunk_seconds": asr_data.get("chunk_seconds"),
            "chunk_count": len(asr_data.get("chunks", [])),
            "segment_count": len(segments),
        },
    )
    report_path = output_dir / "speaker-attribution-audit.html"
    write_text(report_path, html_text)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ASR speaker attribution audit page.")
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--asr-json", type=Path)
    parser.add_argument("--audio-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-run-mapping", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_id = args.video_id
    asr_json_path = args.asr_json or default_asr_json(video_id)
    audio_path = args.audio_path or default_audio_path(video_id)
    output_dir = args.output_dir / video_id
    report_path = build_report(
        video_id=video_id,
        url=args.url,
        asr_json_path=asr_json_path,
        audio_path=audio_path,
        output_dir=output_dir,
        run_mapping=not args.no_run_mapping,
    )
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
