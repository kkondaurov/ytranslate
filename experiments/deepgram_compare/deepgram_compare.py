#!/usr/bin/env python3
import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ytranslate  # noqa: E402


DEFAULT_VIDEO_ID = "HGbA6ze0_3M"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def default_audio_path(video_id: str) -> Path:
    audio_dir = Path(ytranslate.CACHE_DIR) / ytranslate.sanitize_filename(video_id) / "audio"
    for candidate in sorted(audio_dir.glob("source.*")):
        if not candidate.name.endswith(".part"):
            return candidate
    return audio_dir / "source.m4a"


def default_openai_rows_path(video_id: str) -> Path:
    return (
        ROOT
        / "experiments"
        / "speaker_audit"
        / "output"
        / video_id
        / "audit-rows.json"
    )


def default_deepgram_json_path(video_id: str) -> Path:
    return DEFAULT_OUTPUT_DIR / video_id / "deepgram-nova3-diarized.json"


def seconds(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def openai_rows_from_audit(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        start = seconds(row.get("start"))
        end = seconds(row.get("end"), start)
        label = str(row.get("mapped_speaker_label") or row.get("mapped_speaker_id") or "Speaker")
        normalized.append(
            {
                "source": "openai",
                "index": int(row.get("index") or len(normalized) + 1),
                "speaker": label,
                "speaker_raw": str(row.get("local_speaker") or ""),
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "text": clean_text(row.get("text")),
                "chunk": row.get("chunk_index"),
                "meta": str(row.get("speaker_id_source") or ""),
            }
        )
    return normalized


def deepgram_utterances(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = response.get("results") or {}
    utterances = results.get("utterances") or []
    if utterances:
        return utterances

    # Fallback for responses without utterances=true: build rough speaker runs from words.
    channels = results.get("channels") or []
    if not channels:
        return []
    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        return []
    words = alternatives[0].get("words") or []
    runs: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for word in words:
        speaker = word.get("speaker", "unknown")
        if current is None or current.get("speaker") != speaker:
            if current is not None:
                runs.append(current)
            current = {
                "speaker": speaker,
                "start": word.get("start"),
                "end": word.get("end"),
                "transcript": str(word.get("punctuated_word") or word.get("word") or ""),
            }
        else:
            token = str(word.get("punctuated_word") or word.get("word") or "")
            current["transcript"] = f"{current.get('transcript', '')} {token}".strip()
            current["end"] = word.get("end", current.get("end"))
    if current is not None:
        runs.append(current)
    return runs


def deepgram_rows(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, utterance in enumerate(deepgram_utterances(response), 1):
        start = seconds(utterance.get("start"))
        end = seconds(utterance.get("end"), start)
        speaker = utterance.get("speaker")
        confidence = utterance.get("confidence")
        rows.append(
            {
                "source": "deepgram",
                "index": index,
                "speaker": f"Speaker {speaker}",
                "speaker_raw": str(speaker),
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "text": clean_text(utterance.get("transcript")),
                "chunk": None,
                "meta": "" if confidence is None else f"confidence {float(confidence):.3f}",
            }
        )
    return rows


def speaker_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        speaker = str(row.get("speaker") or "Speaker")
        counts[speaker] = counts.get(speaker, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def speaker_overlap_mapping(
    openai_rows: List[Dict[str, Any]],
    dg_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    overlaps: Dict[Tuple[str, str], float] = {}
    for dg in dg_rows:
        dg_speaker = str(dg["speaker"])
        dg_start = seconds(dg.get("start"))
        dg_end = seconds(dg.get("end"), dg_start)
        for oa in openai_rows:
            overlap = min(dg_end, seconds(oa.get("end"))) - max(dg_start, seconds(oa.get("start")))
            if overlap <= 0:
                continue
            key = (dg_speaker, str(oa["speaker"]))
            overlaps[key] = overlaps.get(key, 0.0) + overlap

    result: Dict[str, Dict[str, Any]] = {}
    deepgram_speakers = sorted({str(row["speaker"]) for row in dg_rows})
    for dg_speaker in deepgram_speakers:
        candidates = [
            (oa_speaker, overlap)
            for (candidate_dg, oa_speaker), overlap in overlaps.items()
            if candidate_dg == dg_speaker
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        total = sum(overlap for _, overlap in candidates)
        result[dg_speaker] = {
            "best_openai_speaker": candidates[0][0] if candidates else "",
            "best_overlap_seconds": round(candidates[0][1], 3) if candidates else 0.0,
            "total_overlap_seconds": round(total, 3),
            "share": round(candidates[0][1] / total, 3) if candidates and total else 0.0,
            "candidates": [
                {"speaker": speaker, "overlap_seconds": round(overlap, 3)}
                for speaker, overlap in candidates[:6]
            ],
        }
    return result


def render_entries(rows: List[Dict[str, Any]], side: str) -> str:
    parts: List[str] = []
    for row in rows:
        text = html.escape(row["text"])
        speaker = html.escape(str(row["speaker"]))
        meta_bits = [
            f"#{int(row['index'])}",
            ytranslate.format_timecode(seconds(row["start"])),
            f"{row['duration']:.1f}s",
        ]
        if row.get("chunk") is not None:
            meta_bits.append(f"chunk {row['chunk']}")
        if row.get("meta"):
            meta_bits.append(str(row["meta"]))
        meta = html.escape(" · ".join(meta_bits))
        parts.append(
            "<article class=\"entry\" "
            f"data-side=\"{html.escape(side)}\" "
            f"data-speaker=\"{speaker}\" "
            f"data-start=\"{seconds(row['start']):.3f}\">"
            "<header>"
            f"<button type=\"button\" onclick=\"seekTo({seconds(row['start']):.3f})\">{html.escape(ytranslate.format_timecode(seconds(row['start'])))}</button>"
            f"<strong>{speaker}</strong>"
            f"<span>{meta}</span>"
            "</header>"
            f"<p>{text}</p>"
            "</article>"
        )
    return "\n".join(parts)


def render_speaker_options(rows: List[Dict[str, Any]]) -> str:
    speakers = sorted({str(row["speaker"]) for row in rows})
    return "\n".join(
        f"<option value=\"{html.escape(speaker)}\">{html.escape(speaker)}</option>"
        for speaker in speakers
    )


def render_mapping_table(mapping: Dict[str, Dict[str, Any]]) -> str:
    rows = [
        "<table class=\"mapping\"><thead><tr><th>Deepgram speaker</th><th>Best OpenAI overlap</th><th>Share</th><th>Top overlaps</th></tr></thead><tbody>"
    ]
    for dg_speaker, info in mapping.items():
        candidates = ", ".join(
            f"{item['speaker']}: {item['overlap_seconds']:.1f}s"
            for item in info.get("candidates", [])
        )
        rows.append(
            "<tr>"
            f"<th>{html.escape(dg_speaker)}</th>"
            f"<td>{html.escape(str(info.get('best_openai_speaker') or ''))}</td>"
            f"<td>{float(info.get('share') or 0):.3f}</td>"
            f"<td>{html.escape(candidates)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def render_html(
    video_id: str,
    audio_path: Path,
    openai_rows: List[Dict[str, Any]],
    dg_rows: List[Dict[str, Any]],
    mapping: Dict[str, Dict[str, Any]],
) -> str:
    audio_uri = audio_path.resolve().as_uri()
    openai_speakers = speaker_counts(openai_rows)
    dg_speakers = speaker_counts(dg_rows)
    duration = max(
        [seconds(row.get("end")) for row in openai_rows + dg_rows] or [0.0]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenAI vs Deepgram ASR - {html.escape(video_id)}</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1f2328;
      --muted: #667085;
      --border: #d0d7de;
      --openai: #0969da;
      --deepgram: #148f63;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header.page {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247, 248, 250, 0.96);
      border-bottom: 1px solid var(--border);
      padding: 14px 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 19px;
      letter-spacing: 0;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto auto auto;
      gap: 10px;
      align-items: center;
    }}
    audio {{ width: 100%; min-width: 260px; }}
    select, input, button {{
      min-height: 32px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 4px 8px;
      font: inherit;
    }}
    button {{ cursor: pointer; }}
    main {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 12px;
      height: calc(100vh - 92px);
    }}
    section.column {{
      min-width: 0;
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .column-title {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      border-bottom: 1px solid var(--border);
      background: #fff;
      padding: 10px 12px;
    }}
    .column-title h2 {{
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .column-title span {{ color: var(--muted); font-size: 12px; }}
    .entry {{
      border-bottom: 1px solid #e6e8eb;
      padding: 10px 12px 12px;
    }}
    .entry header {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 6px;
    }}
    .entry strong {{
      color: var(--openai);
      min-width: 88px;
    }}
    [data-side="deepgram"] strong {{ color: var(--deepgram); }}
    .entry span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .entry p {{
      margin: 0;
      white-space: pre-wrap;
    }}
    .entry.hidden {{ display: none; }}
    details {{
      border-top: 1px solid var(--border);
      background: #fff;
      padding: 10px 18px;
    }}
    summary {{ cursor: pointer; font-weight: 650; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 10px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 5px 7px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f3f4f6; }}
    @media (max-width: 900px) {{
      .controls {{ grid-template-columns: 1fr; }}
      main {{ grid-template-columns: 1fr; height: auto; }}
      section.column {{ max-height: 70vh; }}
    }}
  </style>
</head>
<body>
  <header class="page">
    <h1>OpenAI vs Deepgram ASR - {html.escape(video_id)}</h1>
    <div class="controls">
      <audio id="audio" src="{html.escape(audio_uri)}" controls preload="metadata"></audio>
      <input id="search" type="search" placeholder="Filter text">
      <select id="openaiSpeaker"><option value="">OpenAI: all speakers</option>{render_speaker_options(openai_rows)}</select>
      <select id="deepgramSpeaker"><option value="">Deepgram: all speakers</option>{render_speaker_options(dg_rows)}</select>
    </div>
  </header>
  <details>
    <summary>Summary and approximate speaker-overlap map</summary>
    <div class="summary-grid">
      <div>
        <h3>OpenAI audit rows</h3>
        <p>{len(openai_rows)} rows, duration {html.escape(ytranslate.format_timecode(duration))}</p>
        <table><tbody>{''.join(f'<tr><th>{html.escape(k)}</th><td>{v}</td></tr>' for k, v in openai_speakers.items())}</tbody></table>
      </div>
      <div>
        <h3>Deepgram utterances</h3>
        <p>{len(dg_rows)} rows, duration {html.escape(ytranslate.format_timecode(duration))}</p>
        <table><tbody>{''.join(f'<tr><th>{html.escape(k)}</th><td>{v}</td></tr>' for k, v in dg_speakers.items())}</tbody></table>
      </div>
    </div>
    <h3>Deepgram speaker mapped by time overlap with OpenAI labels</h3>
    {render_mapping_table(mapping)}
  </details>
  <main>
    <section class="column" id="openaiColumn">
      <div class="column-title"><h2>OpenAI `gpt-4o-transcribe-diarize` + current reconciliation</h2><span>{len(openai_rows)} rows</span></div>
      {render_entries(openai_rows, "openai")}
    </section>
    <section class="column" id="deepgramColumn">
      <div class="column-title"><h2>Deepgram Nova-3 + diarize_model=latest</h2><span>{len(dg_rows)} utterances</span></div>
      {render_entries(dg_rows, "deepgram")}
    </section>
  </main>
  <script>
    const audio = document.getElementById('audio');
    const search = document.getElementById('search');
    const openaiSpeaker = document.getElementById('openaiSpeaker');
    const deepgramSpeaker = document.getElementById('deepgramSpeaker');

    function seekTo(seconds) {{
      audio.currentTime = seconds;
      audio.play();
    }}

    function applyFilters() {{
      const query = search.value.trim().toLowerCase();
      const openai = openaiSpeaker.value;
      const deepgram = deepgramSpeaker.value;
      for (const entry of document.querySelectorAll('.entry')) {{
        const side = entry.dataset.side;
        const speaker = entry.dataset.speaker;
        const text = entry.innerText.toLowerCase();
        const speakerFilter = side === 'openai' ? openai : deepgram;
        const hidden = (speakerFilter && speaker !== speakerFilter) || (query && !text.includes(query));
        entry.classList.toggle('hidden', Boolean(hidden));
      }}
    }}

    function scrollColumnToTime(columnId, seconds) {{
      const column = document.getElementById(columnId);
      const entries = [...column.querySelectorAll('.entry')];
      let best = null;
      let bestDelta = Infinity;
      for (const entry of entries) {{
        if (entry.classList.contains('hidden')) continue;
        const start = Number(entry.dataset.start || 0);
        const delta = Math.abs(start - seconds);
        if (delta < bestDelta) {{
          best = entry;
          bestDelta = delta;
        }}
      }}
      if (best) best.scrollIntoView({{ block: 'center' }});
    }}

    audio.addEventListener('seeked', () => {{
      scrollColumnToTime('openaiColumn', audio.currentTime);
      scrollColumnToTime('deepgramColumn', audio.currentTime);
    }});
    search.addEventListener('input', applyFilters);
    openaiSpeaker.addEventListener('change', applyFilters);
    deepgramSpeaker.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Render OpenAI vs Deepgram diarized ASR comparison.")
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--openai-rows", type=Path)
    parser.add_argument("--deepgram-json", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    openai_rows_path = args.openai_rows or default_openai_rows_path(args.video_id)
    deepgram_json_path = args.deepgram_json or default_deepgram_json_path(args.video_id)
    audio_path = args.audio or default_audio_path(args.video_id)
    output_dir = args.output_dir / args.video_id

    openai_rows = openai_rows_from_audit(read_json(openai_rows_path))
    dg_rows = deepgram_rows(read_json(deepgram_json_path))
    mapping = speaker_overlap_mapping(openai_rows, dg_rows)

    write_json(output_dir / "normalized-openai-rows.json", openai_rows)
    write_json(output_dir / "normalized-deepgram-rows.json", dg_rows)
    write_json(output_dir / "speaker-overlap-map.json", mapping)
    html_text = render_html(args.video_id, audio_path, openai_rows, dg_rows, mapping)
    write_text(output_dir / "openai-vs-deepgram.html", html_text)
    print(output_dir / "openai-vs-deepgram.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
