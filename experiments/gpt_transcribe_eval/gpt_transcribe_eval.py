#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import difflib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "gpt_transcribe_eval" / "output"
CACHE_ROOT = Path.home() / "Library" / "Caches" / "ytranslate"
TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
TEXT_MODEL = "gpt-transcribe"
DIARIZE_MODEL = "gpt-4o-transcribe-diarize"
TEXT_PRICE_PER_MINUTE = 0.0045
DIARIZE_PRICE_PER_MINUTE = 0.006
REQUEST_TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class SpeakerReference:
    name: str
    label: str
    start: float
    end: float


@dataclass(frozen=True)
class TargetMoment:
    label: str
    start: float
    end: float
    note: str


@dataclass(frozen=True)
class Sample:
    key: str
    video_id: str
    title: str
    episode_url: str
    context: str
    keywords: tuple[str, ...]
    reference_kind: str
    reference_path: Path
    reference_names: dict[str, str]
    speaker_references: tuple[SpeakerReference, ...] = ()
    target_moments: tuple[TargetMoment, ...] = ()
    chunk_index: int = 0

    @property
    def cache_dir(self) -> Path:
        return CACHE_ROOT / self.video_id

    @property
    def chunk_path(self) -> Path:
        return self.cache_dir / "chunks-600s" / f"chunk-{self.chunk_index:03d}.mp3"

    @property
    def source_audio_path(self) -> Path:
        candidates = [
            item
            for item in sorted((self.cache_dir / "audio").glob("source.*"))
            if not item.name.endswith(".part")
        ]
        if not candidates:
            raise FileNotFoundError(f"No source audio found for {self.video_id}")
        return candidates[0]

    @property
    def baseline_raw_path(self) -> Path:
        return (
            self.cache_dir
            / "openai-asr-chunks-gpt-4o-transcribe-diarize-600s"
            / f"chunk-{self.chunk_index:03d}.json"
        )


SAMPLES = (
    Sample(
        key="all-in-audited",
        video_id="HGbA6ze0_3M",
        title="All-In: audited difficult episode",
        episode_url="https://youtu.be/HGbA6ze0_3M",
        context=(
            "An All-In Podcast panel discussion about artificial intelligence and model economics. "
            "The speakers are host Jason Calacanis, Chamath Palihapitiya, David Friedberg, "
            "and guest investor Gavin Baker."
        ),
        keywords=(
            "All-In Podcast",
            "Jason Calacanis",
            "Chamath Palihapitiya",
            "David Friedberg",
            "Gavin Baker",
            "Anthropic",
            "Andrej Karpathy",
            "OpenAI",
            "ARR",
            "recursive self-improvement",
        ),
        reference_kind="user-audited speaker labels",
        reference_path=(
            ROOT
            / "experiments"
            / "speaker_audit"
            / "output"
            / "HGbA6ze0_3M"
            / "audit-rows.json"
        ),
        reference_names={
            "Jason": "Jason Calacanis",
            "Chamath": "Chamath Palihapitiya",
            "Friedberg": "David Friedberg",
            "Gavin": "Gavin Baker",
        },
        speaker_references=(
            SpeakerReference("Jason Calacanis", "Jason", 91.668, 99.068),
            SpeakerReference("Chamath Palihapitiya", "Chamath", 193.252, 198.252),
            SpeakerReference("Gavin Baker", "Gavin", 414.832, 422.482),
            SpeakerReference("David Friedberg", "Friedberg", 517.172, 523.022),
        ),
        target_moments=(
            TargetMoment(
                "Names and terminology",
                28.0,
                52.0,
                "Dense proper names near the opening, including Anthropic and Karpathy.",
            ),
            TargetMoment(
                "Gavin to Jason handoff",
                418.0,
                442.0,
                "Previously reviewed by ear: the current diarized handoff was correct.",
            ),
            TargetMoment(
                "Chamath interruption",
                570.0,
                599.0,
                "Previously reviewed by ear: a short intervention around 09:45.",
            ),
        ),
    ),
    Sample(
        key="all-in-recent",
        video_id="wcV0SRPFK9s",
        title=(
            "All-In: The Fight Over Open Source AI, Anthropic's $1.5B Payout, "
            "NYC Socialists: Evictions = Violence?"
        ),
        episode_url="https://youtu.be/wcV0SRPFK9s",
        context=(
            "An All-In Podcast panel discussion about Chinese open-source AI models, "
            "distillation, Anthropic, and United States technology policy. The speakers are "
            "Jason Calacanis, Chamath Palihapitiya, David Sacks, and David Friedberg."
        ),
        keywords=(
            "All-In Podcast",
            "Jason Calacanis",
            "Chamath Palihapitiya",
            "David Sacks",
            "David Friedberg",
            "Kimi K3",
            "Moonshot AI",
            "Anthropic",
            "distillation",
            "Michael Kratsios",
            "Howard Lutnick",
            "Polymarket",
        ),
        reference_kind="prior reconciled transcript (not human ground truth)",
        reference_path=(
            CACHE_ROOT / "wcV0SRPFK9s" / "openai-asr-resolved-segments.json"
        ),
        reference_names={
            "speaker_jason_calacanis": "Jason Calacanis",
            "speaker_chamath_palihapitiya": "Chamath Palihapitiya",
            "speaker_david_sacks": "David Sacks",
            "speaker_david_friedberg": "David Friedberg",
        },
        speaker_references=(
            SpeakerReference("Jason Calacanis", "speaker_jason_calacanis", 6.3, 12.25),
            SpeakerReference("David Sacks", "speaker_david_sacks", 183.796, 190.046),
            SpeakerReference("David Friedberg", "speaker_david_friedberg", 372.18, 380.73),
            SpeakerReference(
                "Chamath Palihapitiya",
                "speaker_chamath_palihapitiya",
                527.22,
                534.82,
            ),
        ),
    ),
    Sample(
        key="two-person-recent",
        video_id="XDB5beon4DY",
        title="Sam Altman on AGI, Compute, and Human Agency",
        episode_url="https://youtu.be/XDB5beon4DY",
        context=(
            "A two-person interview with Sam Altman about AGI, compute, OpenAI, human agency, "
            "and the concentration of power in artificial intelligence."
        ),
        keywords=(
            "Sam Altman",
            "AGI",
            "OpenAI",
            "compute",
            "human agency",
            "artificial intelligence",
        ),
        reference_kind="prior reconciled transcript (anonymous speakers)",
        reference_path=(
            CACHE_ROOT / "XDB5beon4DY" / "openai-asr-resolved-segments.json"
        ),
        reference_names={
            "speaker_1": "Interviewer",
            "speaker_2": "Sam Altman",
        },
    ),
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def format_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold())


def token_edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_token in enumerate(right, 1):
        current = [right_index]
        for left_index, left_token in enumerate(left, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def text_metrics(reference: str, candidate: str) -> dict[str, Any]:
    reference_words = normalize_words(reference)
    candidate_words = normalize_words(candidate)
    distance = token_edit_distance(reference_words, candidate_words)
    denominator = max(1, len(reference_words))
    return {
        "reference_words": len(reference_words),
        "candidate_words": len(candidate_words),
        "token_edit_distance": distance,
        "disagreement_rate": round(distance / denominator, 4),
        "sequence_similarity": round(
            difflib.SequenceMatcher(None, reference_words, candidate_words, autojunk=False).ratio(),
            4,
        ),
    }


def keyword_hits(text: str, keywords: Iterable[str]) -> list[str]:
    normalized_text = " ".join(normalize_words(text))
    return [
        keyword
        for keyword in keywords
        if " ".join(normalize_words(keyword)) in normalized_text
    ]


def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is required for the experiment") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration_seconds(path: Path) -> float:
    completed = subprocess.run(
        [get_ffmpeg_path(), "-hide_banner", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", completed.stderr)
    if not match:
        raise RuntimeError(f"Could not determine audio duration for {path}")
    return (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + float(match.group(3))
    )


def extract_audio(
    source: Path,
    destination: Path,
    start: float,
    end: float,
    codec: str,
) -> None:
    duration = end - start
    if duration <= 0:
        raise ValueError("Audio extraction end must be after start")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    command = [
        get_ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if codec == "wav":
        command.extend(["-c:a", "pcm_s16le"])
    elif codec == "mp3":
        command.extend(["-b:a", "64k"])
    else:
        raise ValueError(f"Unsupported audio codec: {codec}")
    command.append(str(destination))
    subprocess.run(command, check=True)


def curl_config_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{escaped}"'


def call_transcription_api(
    audio_path: Path,
    api_key: str,
    fields: Sequence[tuple[str, str]],
    attempts: int = 4,
) -> tuple[dict[str, Any], float]:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for the evaluation")
    form_file = f"file=@{audio_path};type=audio/mpeg;filename={audio_path.name}"
    config_lines = [
        f"url = {curl_config_quote(TRANSCRIPTIONS_URL)}",
        "request = POST",
        f"header = {curl_config_quote(f'Authorization: Bearer {api_key}')}",
    ]
    config_lines.extend(
        f"form-string = {curl_config_quote(f'{name}={value}')}" for name, value in fields
    )
    config_lines.extend(
        [
            f"form = {curl_config_quote(form_file)}",
            f"max-time = {REQUEST_TIMEOUT_SECONDS}",
            "connect-timeout = 30",
            "http1.1",
            "silent",
            "show-error",
            "fail-with-body",
        ]
    )
    config = "\n".join(config_lines)
    last_error = ""
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        completed = subprocess.run(
            [curl, "--config", "-"],
            input=config,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS + 45,
            check=False,
        )
        elapsed = time.monotonic() - started
        if completed.returncode == 0:
            try:
                return json.loads(completed.stdout), elapsed
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Transcription API returned invalid JSON: {completed.stdout[:1000]}"
                ) from exc
        last_error = (completed.stdout or completed.stderr or "unknown curl failure").strip()
        if attempt < attempts:
            time.sleep(min(30, 4 * attempt))
    raise RuntimeError(
        f"Transcription API failed after {attempts} attempts for {audio_path.name}: "
        f"{last_error[:1500]}"
    )


def safe_request_fields(fields: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    safe = []
    for name, value in fields:
        if name == "known_speaker_references[]":
            value = f"<data-url {len(value)} chars>"
        safe.append({"name": name, "value": value})
    return safe


def cached_api_call(
    output_path: Path,
    audio_path: Path,
    api_key: str,
    fields: Sequence[tuple[str, str]],
    price_per_minute: float,
    force: bool,
) -> dict[str, Any]:
    if output_path.exists() and not force:
        return read_json(output_path)
    response, elapsed = call_transcription_api(audio_path, api_key, fields)
    duration = probe_duration_seconds(audio_path)
    result = {
        "request": {
            "audio": str(audio_path),
            "fields": safe_request_fields(fields),
        },
        "duration_seconds": round(duration, 3),
        "elapsed_seconds": round(elapsed, 3),
        "estimated_cost_usd": round(duration / 60 * price_per_minute, 6),
        "response": response,
    }
    write_json(output_path, result)
    return result


def gpt_transcribe_fields(sample: Sample, contextual: bool) -> list[tuple[str, str]]:
    fields = [("model", TEXT_MODEL), ("response_format", "json")]
    if contextual:
        fields.extend(
            [
                ("prompt", sample.context),
                ("languages[]", "en"),
            ]
        )
        fields.extend(("keywords[]", keyword) for keyword in sample.keywords)
    return fields


def encode_audio_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def diarize_with_references_fields(
    references: Sequence[tuple[SpeakerReference, Path]],
) -> list[tuple[str, str]]:
    fields = [
        ("model", DIARIZE_MODEL),
        ("response_format", "diarized_json"),
        ("chunking_strategy", "auto"),
    ]
    for reference, _path in references:
        fields.append(("known_speaker_names[]", reference.name))
    for _reference, path in references:
        fields.append(("known_speaker_references[]", encode_audio_data_url(path)))
    return fields


def response_text(result: dict[str, Any]) -> str:
    return str(result.get("response", {}).get("text") or "").strip()


def raw_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "start": float(segment.get("start") or 0),
            "end": float(segment.get("end") or segment.get("start") or 0),
            "speaker": str(segment.get("speaker") or "speaker"),
            "text": str(segment.get("text") or "").strip(),
        }
        for segment in payload.get("segments") or []
        if str(segment.get("text") or "").strip()
    ]


def load_reference_segments(sample: Sample) -> list[dict[str, Any]]:
    data = read_json(sample.reference_path)
    segments = []
    for item in data:
        if sample.reference_kind.startswith("user-audited"):
            chunk_index = int(item.get("chunk_index") or 1) - 1
            speaker_key = str(item.get("mapped_speaker_label") or "speaker")
        else:
            chunk_index = int(item.get("chunk_index") or 1) - 1
            speaker_key = str(item.get("speaker_id") or item.get("speaker") or "speaker")
        if chunk_index != sample.chunk_index:
            continue
        segments.append(
            {
                "start": float(item.get("start") or 0),
                "end": float(item.get("end") or item.get("start") or 0),
                "speaker": sample.reference_names.get(speaker_key, speaker_key),
                "text": str(item.get("text") or "").strip(),
            }
        )
    return [segment for segment in segments if segment["text"]]


def transcript_text(segments: Sequence[dict[str, Any]]) -> str:
    return " ".join(str(segment.get("text") or "").strip() for segment in segments).strip()


def speaker_alias(label: str, expected_names: Iterable[str]) -> str:
    normalized = " ".join(normalize_words(label))
    names = list(expected_names)
    normalized_names = {name: " ".join(normalize_words(name)) for name in names}
    for name, full in normalized_names.items():
        if normalized == full:
            return name
    tokens = normalized.split()
    for name, full in normalized_names.items():
        last = full.rsplit(" ", 1)[-1]
        if last in tokens:
            return name
    first_name_counts: dict[str, int] = {}
    for full in normalized_names.values():
        first = full.split(" ", 1)[0]
        first_name_counts[first] = first_name_counts.get(first, 0) + 1
    for name, full in normalized_names.items():
        first = full.split(" ", 1)[0]
        if first_name_counts[first] == 1 and first in tokens:
            return name
    return label.strip()


def overlap_seconds(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(left["end"]), float(right["end"]))
        - max(float(left["start"]), float(right["start"])),
    )


def speaker_overlap_score(
    reference: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    expected_names: Iterable[str],
) -> dict[str, Any]:
    expected_names = list(expected_names)
    matched = 0.0
    scored = 0.0
    rows = []
    for item in candidate:
        best = max(reference, key=lambda ref: overlap_seconds(ref, item), default=None)
        if best is None:
            continue
        overlap = overlap_seconds(best, item)
        if overlap <= 0:
            continue
        expected = speaker_alias(str(best["speaker"]), expected_names)
        actual = speaker_alias(str(item["speaker"]), expected_names)
        scored += overlap
        if expected == actual:
            matched += overlap
        rows.append(
            {
                "start": round(float(item["start"]), 3),
                "end": round(float(item["end"]), 3),
                "expected": expected,
                "actual": actual,
                "match": expected == actual,
                "overlap_seconds": round(overlap, 3),
                "text": item.get("text", ""),
            }
        )
    return {
        "matched_overlap_seconds": round(matched, 3),
        "scored_overlap_seconds": round(scored, 3),
        "agreement": round(matched / scored, 4) if scored else None,
        "mismatches": [row for row in rows if not row["match"]],
    }


def speaker_change_points(segments: Sequence[dict[str, Any]]) -> list[float]:
    changes = []
    previous = None
    for segment in sorted(segments, key=lambda item: (item["start"], item["end"])):
        speaker = str(segment["speaker"])
        if previous is not None and speaker != previous:
            changes.append(float(segment["start"]))
        previous = speaker
    return changes


def boundary_score(
    reference: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    tolerance_seconds: float = 1.5,
) -> dict[str, Any]:
    expected = speaker_change_points(reference)
    actual = speaker_change_points(candidate)
    unmatched = set(range(len(actual)))
    matches = 0
    for point in expected:
        choices = [
            index
            for index in unmatched
            if abs(actual[index] - point) <= tolerance_seconds
        ]
        if not choices:
            continue
        best = min(choices, key=lambda index: abs(actual[index] - point))
        unmatched.remove(best)
        matches += 1
    precision = matches / len(actual) if actual else 0.0
    recall = matches / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_changes": len(expected),
        "candidate_changes": len(actual),
        "matched_changes": matches,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tolerance_seconds": tolerance_seconds,
    }


def segments_in_window(
    segments: Sequence[dict[str, Any]], start: float, end: float
) -> list[dict[str, Any]]:
    return [
        segment
        for segment in segments
        if float(segment["end"]) > start and float(segment["start"]) < end
    ]


def render_segments(segments: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for segment in segments:
        blocks.append(
            '<div class="turn">'
            f'<button class="stamp" data-seek="{float(segment["start"]):.3f}">'
            f'{format_time(float(segment["start"]))}</button>'
            f'<strong>{html.escape(str(segment["speaker"]))}</strong>'
            f'<p>{html.escape(str(segment["text"]))}</p>'
            "</div>"
        )
    return "\n".join(blocks)


def render_plain_text(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    return "\n".join(f"<p>{html.escape(part)}</p>" for part in paragraphs if part)


def metric(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(evaluations: Sequence[dict[str, Any]], output_dir: Path) -> str:
    total_cost = sum(
        call["estimated_cost_usd"]
        for evaluation in evaluations
        for call in evaluation["api_calls"]
    )
    summary_rows = []
    sample_sections = []
    nav = []
    for evaluation in evaluations:
        sample = evaluation["sample"]
        key = sample["key"]
        nav.append(f'<a href="#{html.escape(key)}">{html.escape(key)}</a>')
        plain = evaluation["text_metrics"]["plain"]
        contextual = evaluation["text_metrics"]["contextual"]
        speaker = evaluation.get("known_speaker_evaluation") or {}
        speaker_agreement = speaker.get("speaker_overlap", {}).get("agreement")
        summary_rows.append(
            "<tr>"
            f'<td><a href="#{html.escape(key)}">{html.escape(sample["title"])}</a></td>'
            f'<td>{plain["disagreement_rate"]:.1%}</td>'
            f'<td>{contextual["disagreement_rate"]:.1%}</td>'
            f'<td>{f"{speaker_agreement:.1%}" if speaker_agreement is not None else "n/a"}</td>'
            f'<td>{metric(speaker.get("boundaries", {}).get("f1"), 3)}</td>'
            f'<td>${evaluation["incremental_cost_usd"]:.3f}</td>'
            "</tr>"
        )

        call_by_name = {call["name"]: call for call in evaluation["api_calls"]}
        target_html = []
        for target in evaluation.get("targets", []):
            known_segments = target.get("known_segments") or []
            target_html.append(
                f"""
                <article class="target">
                  <div class="target-head">
                    <div>
                      <h4>{html.escape(target['label'])}</h4>
                      <p>{html.escape(target['note'])}</p>
                    </div>
                    <span>{format_time(target['start'])}-{format_time(target['end'])}</span>
                  </div>
                  <audio controls preload="metadata" src="{html.escape(target['audio'])}"></audio>
                  <div class="target-grid">
                    <div><h5>Audited/current diarization</h5>{render_segments(target['reference_segments'])}</div>
                    <div><h5>GPT-Transcribe, contextual</h5>{render_plain_text(target['contextual_text'])}</div>
                    <div><h5>GPT-Transcribe, plain</h5>{render_plain_text(target['plain_text'])}</div>
                    <div><h5>Known-speaker diarization</h5>{render_segments(known_segments) if known_segments else '<p class="muted">Not run.</p>'}</div>
                  </div>
                </article>
                """
            )

        known = evaluation.get("known_speaker_evaluation")
        known_metrics = ""
        if known:
            overlap = known["speaker_overlap"]
            boundaries = known["boundaries"]
            known_metrics = f"""
              <dl class="metrics">
                <div><dt>Speaker-label agreement</dt><dd>{overlap['agreement']:.1%}</dd></div>
                <div><dt>Boundary F1</dt><dd>{boundaries['f1']:.3f}</dd></div>
                <div><dt>Reference basis</dt><dd>{html.escape(sample['reference_kind'])}</dd></div>
                <div><dt>Mismatched segments</dt><dd>{len(overlap['mismatches'])}</dd></div>
              </dl>
            """

        known_segments_html = (
            render_segments(evaluation.get("known_speaker_segments") or [])
            or '<p class="muted">Known-speaker references were not tested for this sample.</p>'
        )
        plain_call = call_by_name["gpt-transcribe plain"]
        contextual_call = call_by_name["gpt-transcribe contextual"]
        sample_sections.append(
            f"""
            <section class="sample" id="{html.escape(key)}" data-audio="{html.escape(evaluation['audio'])}">
              <header class="sample-head">
                <div>
                  <p class="eyebrow">{html.escape(key)}</p>
                  <h2>{html.escape(sample['title'])}</h2>
                  <a href="{html.escape(sample['episode_url'])}">{html.escape(sample['episode_url'])}</a>
                </div>
                <dl class="metrics compact">
                  <div><dt>Audio</dt><dd>{evaluation['duration_seconds'] / 60:.1f} min</dd></div>
                  <div><dt>Plain latency</dt><dd>{plain_call['elapsed_seconds']:.1f}s</dd></div>
                  <div><dt>Context latency</dt><dd>{contextual_call['elapsed_seconds']:.1f}s</dd></div>
                  <div><dt>Experiment cost</dt><dd>${evaluation['incremental_cost_usd']:.3f}</dd></div>
                </dl>
              </header>
              <div class="audio-band">
                <audio controls preload="metadata" src="{html.escape(evaluation['audio'])}"></audio>
                <span>Timestamp buttons seek this player.</span>
              </div>
              <div class="metric-band">
                <dl class="metrics">
                  <div><dt>Plain disagreement</dt><dd>{plain['disagreement_rate']:.1%}</dd></div>
                  <div><dt>Context disagreement</dt><dd>{contextual['disagreement_rate']:.1%}</dd></div>
                  <div><dt>Plain keyword hits</dt><dd>{len(evaluation['keyword_hits']['plain'])}/{len(sample['keywords'])}</dd></div>
                  <div><dt>Context keyword hits</dt><dd>{len(evaluation['keyword_hits']['contextual'])}/{len(sample['keywords'])}</dd></div>
                  <div><dt>Context-only keywords</dt><dd>{html.escape(', '.join(evaluation['keyword_hits']['context_only']) or 'none')}</dd></div>
                </dl>
                {known_metrics}
              </div>
              <p class="evidence">Text disagreement is measured against the current diarized transcript, not a verbatim human reference. Speaker agreement is only ground-truth-like for the explicitly user-audited episode.</p>
              <div class="transcript-grid">
                <article>
                  <h3>Current diarized transcript</h3>
                  <div class="scroll-pane">{render_segments(evaluation['reference_segments'])}</div>
                </article>
                <article>
                  <h3>GPT-Transcribe with context</h3>
                  <div class="scroll-pane prose">{render_plain_text(evaluation['contextual_text'])}</div>
                </article>
                <article>
                  <h3>GPT-Transcribe plain</h3>
                  <div class="scroll-pane prose">{render_plain_text(evaluation['plain_text'])}</div>
                </article>
                <article>
                  <h3>Diarization with known voices</h3>
                  <div class="scroll-pane">{known_segments_html}</div>
                </article>
              </div>
              {''.join(target_html)}
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GPT-Transcribe evaluation</title>
  <style>
    :root {{ color-scheme: light; --bg:#f3f5f6; --paper:#fff; --ink:#182126; --muted:#657078; --line:#d5dade; --teal:#0d6b64; --amber:#9a5b05; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.48 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1680px,calc(100vw - 32px)); margin:auto; padding:26px 0 56px; }}
    a {{ color:var(--teal); }}
    h1,h2,h3,h4,h5 {{ letter-spacing:0; }}
    h1 {{ margin:0; font-size:34px; line-height:1.1; }}
    h2 {{ margin:0 0 7px; font-size:23px; line-height:1.2; }}
    h3 {{ margin:0; padding:12px 14px; border-bottom:1px solid var(--line); font-size:15px; }}
    h4 {{ margin:0 0 4px; font-size:17px; }}
    h5 {{ margin:0 0 9px; font-size:13px; }}
    .page-head {{ display:flex; justify-content:space-between; gap:24px; align-items:end; margin-bottom:20px; }}
    .page-head p {{ margin:8px 0 0; max-width:800px; color:var(--muted); }}
    nav {{ display:flex; gap:7px; flex-wrap:wrap; justify-content:flex-end; }}
    nav a {{ padding:6px 9px; border:1px solid var(--line); background:var(--paper); border-radius:5px; text-decoration:none; font-size:12px; }}
    .summary {{ width:100%; border-collapse:collapse; background:var(--paper); margin-bottom:26px; }}
    .decision {{ margin:0 0 18px; padding:16px 18px; border:1px solid #c6d2d0; border-left:4px solid var(--teal); background:#eef4f3; }}
    .decision h2 {{ margin-bottom:8px; }}
    .decision p {{ max-width:1200px; margin:7px 0; }}
    .decision .sources {{ color:var(--muted); font-size:12px; }}
    th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
    th {{ background:#e9edef; font-size:12px; color:#465159; }}
    .sample {{ margin:0 -16px 30px; padding:22px 16px 28px; border-top:2px solid #9aa4aa; background:var(--paper); }}
    .sample-head {{ display:flex; justify-content:space-between; gap:20px; align-items:start; }}
    .eyebrow {{ margin:0 0 5px; color:var(--teal); font-size:11px; font-weight:700; text-transform:uppercase; }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:0; margin:0; }}
    .metrics div {{ min-width:145px; padding:8px 12px; border-left:1px solid var(--line); }}
    .metrics dt {{ color:var(--muted); font-size:11px; }}
    .metrics dd {{ margin:2px 0 0; font-weight:650; }}
    .compact {{ justify-content:flex-end; }}
    .audio-band {{ display:flex; align-items:center; gap:14px; margin:18px 0 0; padding:10px 12px; background:#e9efef; border-block:1px solid #c5d6d4; }}
    audio {{ width:min(720px,100%); height:34px; }}
    .audio-band span,.evidence,.muted {{ color:var(--muted); font-size:12px; }}
    .metric-band {{ display:flex; justify-content:space-between; gap:18px; border-bottom:1px solid var(--line); }}
    .evidence {{ margin:9px 0 14px; }}
    .transcript-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
    .transcript-grid article {{ border:1px solid var(--line); min-width:0; }}
    .scroll-pane {{ height:430px; overflow:auto; padding:4px 12px 14px; }}
    .turn {{ display:grid; grid-template-columns:56px 120px minmax(0,1fr); gap:8px; padding:8px 0; border-bottom:1px solid #e5e8ea; align-items:start; }}
    .turn p {{ margin:0; }}
    .stamp {{ width:52px; padding:2px 4px; border:1px solid #b9c1c5; background:#fff; color:var(--teal); border-radius:4px; cursor:pointer; font-size:11px; }}
    .prose p {{ margin:9px 0; }}
    .target {{ margin-top:18px; padding-top:17px; border-top:2px solid #c3c9cc; }}
    .target-head {{ display:flex; justify-content:space-between; gap:20px; }}
    .target-head p {{ margin:0 0 8px; color:var(--muted); }}
    .target > audio {{ margin:8px 0 12px; }}
    .target-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); border:1px solid var(--line); }}
    .target-grid > div {{ padding:12px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); min-height:120px; }}
    .target-grid .turn {{ grid-template-columns:52px 105px minmax(0,1fr); }}
    @media (max-width:900px) {{
      main {{ width:min(100% - 18px,1680px); }}
      .page-head,.sample-head,.metric-band {{ display:block; }}
      nav,.compact {{ justify-content:flex-start; margin-top:12px; }}
      .transcript-grid,.target-grid {{ grid-template-columns:1fr; }}
      .scroll-pane {{ height:360px; }}
      .turn {{ grid-template-columns:52px 92px minmax(0,1fr); }}
      .summary-wrap {{ overflow:auto; }}
    }}
  </style>
</head>
<body>
<main>
  <header class="page-head">
    <div>
      <h1>GPT-Transcribe evaluation</h1>
      <p>Controlled comparison against ytranslate's current diarized ASR. Total incremental API cost: ${total_cost:.3f}. Generated from cached 10-minute source chunks; production was not modified.</p>
    </div>
    <nav>{''.join(nav)}</nav>
  </header>
  <section class="decision">
    <p class="eyebrow">Production decision</p>
    <h2>Keep the current diarized pipeline.</h2>
    <p><strong>Do not add GPT-Transcribe as a default second pass.</strong> It has no speaker labels or timestamps, its context benefit was small and inconsistent across these samples, and aligning its prose back onto interruption-heavy turns would add a new failure surface. One additional text pass would raise ASR spend from $0.006/min to $0.0105/min, a 75% increase.</p>
    <p><strong>Do not replace speaker reconciliation with known voice clips yet.</strong> The same mechanism reached 99.5% label agreement on the user-audited episode but only 84.1% against the recent All-In reconciliation, with several long speaker swaps rather than crosstalk-only errors.</p>
    <p class="sources">Model capabilities: <a href="https://developers.openai.com/api/docs/guides/transcription">transcription overview</a> and <a href="https://developers.openai.com/api/docs/guides/speech-to-text">file transcription guide</a>. Prices: <a href="https://developers.openai.com/api/docs/pricing#transcription-models">OpenAI transcription pricing</a>.</p>
  </section>
  <div class="summary-wrap">
    <table class="summary">
      <thead><tr><th>Sample</th><th>Plain text disagreement</th><th>Context text disagreement</th><th>Known-voice label agreement</th><th>Boundary F1</th><th>Incremental cost</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody>
    </table>
  </div>
  {''.join(sample_sections)}
</main>
<script>
  document.addEventListener('click', (event) => {{
    const button = event.target.closest('[data-seek]');
    if (!button) return;
    const sample = button.closest('.sample');
    const audio = sample ? sample.querySelector('.audio-band audio') : null;
    if (!audio) return;
    audio.currentTime = Number(button.dataset.seek || 0);
    audio.play();
  }});
</script>
</body>
</html>
"""


def build_evaluation(
    sample: Sample,
    output_dir: Path,
    api_key: str,
    force: bool,
) -> dict[str, Any]:
    print(f"\n[{sample.key}] preparing cached inputs", flush=True)
    for required in (sample.chunk_path, sample.source_audio_path, sample.baseline_raw_path, sample.reference_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing required artifact: {required}")

    sample_dir = output_dir / sample.key
    api_dir = sample_dir / "api"
    assets_dir = sample_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    audio_asset = assets_dir / "source-chunk.mp3"
    if not audio_asset.exists():
        shutil.copy2(sample.chunk_path, audio_asset)

    baseline_raw = read_json(sample.baseline_raw_path)
    baseline_segments = raw_segments(baseline_raw)
    reference_segments = load_reference_segments(sample)
    baseline_text = str(baseline_raw.get("text") or transcript_text(baseline_segments)).strip()

    print(f"[{sample.key}] calling {TEXT_MODEL} without context", flush=True)
    plain = cached_api_call(
        api_dir / "gpt-transcribe-plain.json",
        sample.chunk_path,
        api_key,
        gpt_transcribe_fields(sample, contextual=False),
        TEXT_PRICE_PER_MINUTE,
        force,
    )
    print(f"[{sample.key}] calling {TEXT_MODEL} with context", flush=True)
    contextual = cached_api_call(
        api_dir / "gpt-transcribe-contextual.json",
        sample.chunk_path,
        api_key,
        gpt_transcribe_fields(sample, contextual=True),
        TEXT_PRICE_PER_MINUTE,
        force,
    )

    api_calls = [
        {"name": "gpt-transcribe plain", **{key: plain[key] for key in ("duration_seconds", "elapsed_seconds", "estimated_cost_usd")}},
        {"name": "gpt-transcribe contextual", **{key: contextual[key] for key in ("duration_seconds", "elapsed_seconds", "estimated_cost_usd")}},
    ]
    known_result = None
    known_segments: list[dict[str, Any]] = []
    known_evaluation = None
    if sample.speaker_references:
        encoded_references = []
        for index, reference in enumerate(sample.speaker_references):
            duration = reference.end - reference.start
            if not 2 <= duration <= 10:
                raise ValueError(f"Known-speaker reference must be 2-10 seconds: {reference}")
            reference_path = assets_dir / f"speaker-{index + 1}-{reference.label}.wav"
            extract_audio(
                sample.source_audio_path,
                reference_path,
                reference.start,
                reference.end,
                "wav",
            )
            encoded_references.append((reference, reference_path))
        print(f"[{sample.key}] calling diarization with {len(encoded_references)} known voices", flush=True)
        known_result = cached_api_call(
            api_dir / "diarize-known-speakers.json",
            sample.chunk_path,
            api_key,
            diarize_with_references_fields(encoded_references),
            DIARIZE_PRICE_PER_MINUTE,
            force,
        )
        known_segments = raw_segments(known_result["response"])
        expected_names = sample.reference_names.values()
        known_evaluation = {
            "speaker_overlap": speaker_overlap_score(
                reference_segments,
                known_segments,
                expected_names,
            ),
            "boundaries": boundary_score(reference_segments, known_segments),
        }
        api_calls.append(
            {
                "name": "diarize known speakers",
                **{
                    key: known_result[key]
                    for key in ("duration_seconds", "elapsed_seconds", "estimated_cost_usd")
                },
            }
        )

    targets = []
    for index, target in enumerate(sample.target_moments, 1):
        target_asset = assets_dir / f"target-{index:02d}.mp3"
        extract_audio(sample.chunk_path, target_asset, target.start, target.end, "mp3")
        print(f"[{sample.key}] transcribing target {index}/{len(sample.target_moments)} plain", flush=True)
        target_plain = cached_api_call(
            api_dir / f"target-{index:02d}-plain.json",
            target_asset,
            api_key,
            gpt_transcribe_fields(sample, contextual=False),
            TEXT_PRICE_PER_MINUTE,
            force,
        )
        print(f"[{sample.key}] transcribing target {index}/{len(sample.target_moments)} contextual", flush=True)
        target_contextual = cached_api_call(
            api_dir / f"target-{index:02d}-contextual.json",
            target_asset,
            api_key,
            gpt_transcribe_fields(sample, contextual=True),
            TEXT_PRICE_PER_MINUTE,
            force,
        )
        for name, result in (
            (f"target {index} plain", target_plain),
            (f"target {index} contextual", target_contextual),
        ):
            api_calls.append(
                {
                    "name": name,
                    **{
                        key: result[key]
                        for key in ("duration_seconds", "elapsed_seconds", "estimated_cost_usd")
                    },
                }
            )
        known_window = segments_in_window(known_segments, target.start, target.end)
        targets.append(
            {
                **asdict(target),
                "audio": str(target_asset.relative_to(output_dir)),
                "reference_segments": segments_in_window(
                    reference_segments, target.start, target.end
                ),
                "known_segments": known_window,
                "plain_text": response_text(target_plain),
                "contextual_text": response_text(target_contextual),
            }
        )

    plain_text = response_text(plain)
    contextual_text = response_text(contextual)
    plain_hits = keyword_hits(plain_text, sample.keywords)
    contextual_hits = keyword_hits(contextual_text, sample.keywords)
    evaluation = {
        "sample": {
            "key": sample.key,
            "video_id": sample.video_id,
            "title": sample.title,
            "episode_url": sample.episode_url,
            "context": sample.context,
            "keywords": list(sample.keywords),
            "reference_kind": sample.reference_kind,
        },
        "audio": str(audio_asset.relative_to(output_dir)),
        "duration_seconds": round(probe_duration_seconds(sample.chunk_path), 3),
        "reference_segments": reference_segments,
        "baseline_text": baseline_text,
        "plain_text": plain_text,
        "contextual_text": contextual_text,
        "known_speaker_segments": known_segments,
        "text_metrics": {
            "plain": text_metrics(baseline_text, plain_text),
            "contextual": text_metrics(baseline_text, contextual_text),
        },
        "keyword_hits": {
            "baseline": keyword_hits(baseline_text, sample.keywords),
            "plain": plain_hits,
            "contextual": contextual_hits,
            "context_only": sorted(set(contextual_hits) - set(plain_hits)),
        },
        "known_speaker_evaluation": known_evaluation,
        "targets": targets,
        "api_calls": api_calls,
        "incremental_cost_usd": round(
            sum(call["estimated_cost_usd"] for call in api_calls), 6
        ),
    }
    write_json(sample_dir / "evaluation.json", evaluation)
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GPT-Transcribe against ytranslate's current diarized ASR."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample",
        action="append",
        choices=[sample.key for sample in SAMPLES],
        help="Run only the named sample; may be repeated.",
    )
    parser.add_argument("--force", action="store_true", help="Repeat cached API calls.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [sample for sample in SAMPLES if not args.sample or sample.key in args.sample]
    evaluations = [
        build_evaluation(sample, output_dir, api_key, args.force) for sample in selected
    ]
    write_json(output_dir / "evaluation.json", evaluations)
    report_path = output_dir / "gpt-transcribe-evaluation.html"
    report_path.write_text(render_report(evaluations, output_dir), encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(
        "Incremental API cost: "
        f"${sum(item['incremental_cost_usd'] for item in evaluations):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
