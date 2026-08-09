#!/usr/bin/env python3
"""Replay ytranslate's text pipeline with GPT-5.6 Luna variants.

The experiment starts from cached diarized ASR segments. It deliberately does
not rerun or modify transcription, production caches, or the launch-managed
server configuration.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from docx import Document
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ytranslate  # noqa: E402


MODEL = "gpt-5.6-luna"
DEFAULT_EFFORTS = ("none", "low", "medium")
VALID_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
INPUT_PRICE_PER_MILLION = 0.20
CACHED_INPUT_PRICE_PER_MILLION = 0.02
OUTPUT_PRICE_PER_MILLION = 1.20
SEGMENT_RECONCILIATION_CHUNKS_PER_BATCH = 2
SEGMENT_RECONCILIATION_CONTEXT_SEGMENTS = 12
KNOWN_ATTRIBUTION_WINDOWS = (
    {
        "group": "page_5",
        "name": "Friedberg infrastructure conclusion",
        "start": 405.008,
        "end": 429.414,
        "expected": "Friedberg",
    },
    {
        "group": "page_5",
        "name": "Jason asks Brad",
        "start": 429.414,
        "end": 431.114,
        "expected": "Jason",
    },
    {
        "group": "page_5",
        "name": "Brad answers",
        "start": 431.264,
        "end": 536.222,
        "expected": "Brad",
    },
    {
        "group": "pages_11_12",
        "name": "Friedberg mixed-model conclusion",
        "start": 1080.342,
        "end": 1093.170,
        "expected": "Friedberg",
    },
    {
        "group": "pages_11_12",
        "name": "Jason asks about token pricing",
        "start": 1093.370,
        "end": 1118.830,
        "expected": "Jason",
    },
    {
        "group": "pages_11_12",
        "name": "Brad answers",
        "start": 1119.744,
        "end": 1215.000,
        "expected": "Brad",
    },
)


def object_value(value: Any, name: str, default: Any = 0) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def usage_record(response: Any) -> Dict[str, int]:
    usage = object_value(response, "usage", None)
    input_details = object_value(usage, "input_tokens_details", None)
    output_details = object_value(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(object_value(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(object_value(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(object_value(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(object_value(output_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(object_value(usage, "total_tokens", 0) or 0),
    }


def estimated_cost_usd(records: Iterable[Dict[str, Any]]) -> float:
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    for record in records:
        if record.get("status") != "ok":
            continue
        input_tokens += int(record.get("input_tokens") or 0)
        cached_input_tokens += int(record.get("cached_input_tokens") or 0)
        output_tokens += int(record.get("output_tokens") or 0)
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    return (
        uncached_input_tokens * INPUT_PRICE_PER_MILLION
        + cached_input_tokens * CACHED_INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000


class RequestRecorder:
    def __init__(self, effort: str) -> None:
        self.effort = effort
        self.stage = "unknown"
        self.records: List[Dict[str, Any]] = []
        self.phase_seconds: Dict[str, float] = defaultdict(float)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous_stage = self.stage
        self.stage = name
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phase_seconds[name] += time.perf_counter() - started
            self.stage = previous_stage

    def summarize(self) -> Dict[str, Any]:
        successful = [record for record in self.records if record.get("status") == "ok"]
        totals = {
            key: sum(int(record.get(key) or 0) for record in successful)
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
            )
        }
        by_stage: Dict[str, Dict[str, Any]] = {}
        stage_names = sorted({record.get("stage", "unknown") for record in self.records})
        for stage in stage_names:
            stage_records = [record for record in self.records if record.get("stage") == stage]
            stage_successful = [record for record in stage_records if record.get("status") == "ok"]
            by_stage[stage] = {
                "requests": len(stage_records),
                "successful_requests": len(stage_successful),
                "api_seconds": round(
                    sum(float(record.get("seconds") or 0) for record in stage_records), 3
                ),
                "input_tokens": sum(
                    int(record.get("input_tokens") or 0) for record in stage_successful
                ),
                "cached_input_tokens": sum(
                    int(record.get("cached_input_tokens") or 0) for record in stage_successful
                ),
                "output_tokens": sum(
                    int(record.get("output_tokens") or 0) for record in stage_successful
                ),
                "reasoning_tokens": sum(
                    int(record.get("reasoning_tokens") or 0) for record in stage_successful
                ),
                "estimated_cost_usd": round(estimated_cost_usd(stage_records), 6),
            }
        return {
            "effort": self.effort,
            "request_count": len(self.records),
            "successful_request_count": len(successful),
            "api_seconds": round(
                sum(float(record.get("seconds") or 0) for record in self.records), 3
            ),
            "phase_seconds": {
                key: round(value, 3) for key, value in self.phase_seconds.items()
            },
            **totals,
            "estimated_cost_usd": round(estimated_cost_usd(self.records), 6),
            "by_stage": by_stage,
            "requests": self.records,
        }


class InstrumentedResponses:
    def __init__(self, responses: Any, recorder: RequestRecorder) -> None:
        self._responses = responses
        self._recorder = recorder

    def create(self, **kwargs: Any) -> Any:
        request_kwargs = dict(kwargs)
        request_kwargs.pop("temperature", None)
        request_kwargs["reasoning"] = {"effort": self._recorder.effort}
        started = time.perf_counter()
        try:
            response = self._responses.create(**request_kwargs)
        except Exception as exc:
            self._recorder.records.append(
                {
                    "stage": self._recorder.stage,
                    "status": "error",
                    "seconds": round(time.perf_counter() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise
        record = {
            "stage": self._recorder.stage,
            "status": "ok",
            "seconds": round(time.perf_counter() - started, 3),
            **usage_record(response),
        }
        self._recorder.records.append(record)
        return response


class InstrumentedClient:
    def __init__(self, client: OpenAI, recorder: RequestRecorder) -> None:
        self.responses = InstrumentedResponses(client.responses, recorder)


def render_source_markdown(
    title: str,
    speakers: List[Dict[str, str]],
    turns: List[Dict[str, str]],
    metadata: Dict[str, Any],
) -> str:
    source_turns = [
        {**turn, "text_translated": turn.get("text_source") or ""}
        for turn in turns
    ]
    return ytranslate.render_markdown_transcript(title, speakers, source_turns, metadata)


def render_variant_pdf(
    output_dir: Path,
    base_name: str,
    title: str,
    speakers: List[Dict[str, str]],
    turns: List[Dict[str, str]],
) -> Path:
    temporary_docx = output_dir / f"{base_name}.docx"
    ytranslate.render_docx(title, speakers, turns, str(temporary_docx))
    try:
        return Path(ytranslate.convert_docx_to_pdf(str(temporary_docx)))
    finally:
        temporary_docx.unlink(missing_ok=True)


def baseline_docx_to_markdown(docx_path: Path, title: str) -> str:
    document = Document(str(docx_path))
    lines = [f"# {title}", "", "## Baseline", "", "- **model**: gpt-5.4-mini", ""]
    speakers: List[str] = []
    transcript: List[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text or text == title:
            continue
        style_name = str(getattr(paragraph.style, "name", "") or "")
        if style_name.startswith("List Bullet"):
            speakers.append(text)
        elif ": " in text:
            speaker, body = text.split(": ", 1)
            transcript.extend([f"**{speaker}:** {body}", ""])
    if speakers:
        lines.extend(["## Speakers", ""])
        lines.extend(f"- {speaker}" for speaker in speakers)
        lines.append("")
    lines.extend(["## Transcript", ""])
    lines.extend(transcript)
    return "\n".join(lines).rstrip() + "\n"


def speaker_label(speakers: List[Dict[str, str]], speaker_id: str) -> str:
    for speaker in speakers:
        if speaker.get("id") == speaker_id:
            return str(
                speaker.get("label_full")
                or speaker.get("label_short")
                or speaker_id
            )
    return speaker_id


def segment_reconciliation_schema(
    segment_count: int,
    speaker_ids: Sequence[str],
) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "assignments": {
                "type": "array",
                "minItems": segment_count,
                "maxItems": segment_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "segment_id": {"type": "integer"},
                        "speaker_id": {"type": "string", "enum": list(speaker_ids)},
                    },
                    "required": ["segment_id", "speaker_id"],
                },
            }
        },
        "required": ["assignments"],
    }


def build_segment_reconciliation_system_prompt() -> str:
    return (
        "You perform final segment-level speaker attribution for a diarized conversation. "
        "The local diarizer label and current speaker assignment are noisy hints, not identity constraints. "
        "The current assignment can be systematically wrong across a long passage, so audit it rather than copying it. "
        "A single local label can contain multiple real speakers, including a question from one person "
        "followed by an answer from another person without any local-label change. "
        "Conversely, the diarizer can change local labels in the middle of one person's uninterrupted answer. "
        "When adjacent or overlapping segments form one grammatical sentence or an uninterrupted argument, "
        "keep the same speaker across them unless the words contain a genuine conversational handoff. "
        "Never switch speakers solely because the local label or current assignment changes. "
        "Infer the speaker of every TARGET segment from the words, direct address, question-answer flow, "
        "self-reference, interruptions, and neighboring context. A prompt such as 'Brad, what do you think?' "
        "belongs to the person addressing Brad; the answer that follows likely belongs to Brad. "
        "A candidate roster can include people who are absent, so never assign someone merely because they are listed. "
        "Return every TARGET segment exactly once, keep its exact segment_id, select only a provided speaker_id, "
        "and do not rewrite, merge, or split transcript text."
    )


def clean_prompt_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def segment_prompt_line(
    marker: str,
    segment_id: int,
    segment: Dict[str, Any],
    speakers: List[Dict[str, str]],
) -> str:
    start = float(segment.get("start") or 0)
    end = float(segment.get("end") or start)
    current_id = str(segment.get("speaker_id") or "")
    current_label = speaker_label(speakers, current_id) if current_id else "unassigned"
    chunk_index, local_speaker = ytranslate.get_segment_local_key(segment)
    return (
        f"{marker} {segment_id} | {ytranslate.format_timecode(start)}-"
        f"{ytranslate.format_timecode(end)} | local={chunk_index}/{local_speaker} | "
        f"current={current_id or 'unassigned'} ({current_label}) | "
        f"text={clean_prompt_text(segment.get('text'))}"
    )


def build_segment_reconciliation_user_prompt(
    *,
    canonical_url: str,
    title: str,
    description: str,
    source_language_hint: Optional[str],
    speakers: List[Dict[str, str]],
    indexed_segments: Sequence[Tuple[int, Dict[str, Any]]],
    target_segment_ids: Sequence[int],
) -> str:
    target_ids = set(target_segment_ids)
    lines = [
        f"Video URL: {canonical_url}",
        f"Title: {title}",
        f"Description: {clean_prompt_text(description)[:4000]}",
    ]
    if source_language_hint:
        lines.append(f"Source language hint: {source_language_hint}")
    lines.extend(["", "Candidate speakers:"])
    for speaker in speakers:
        aliases = ", ".join(str(alias) for alias in speaker.get("aliases", []))
        lines.append(
            f"- {speaker['id']}: {speaker.get('label_full') or speaker.get('label_short') or speaker['id']}"
            + (f" (aliases: {aliases})" if aliases else "")
        )
    lines.extend(
        [
            "",
            "Transcript context. Assign only lines marked TARGET; BEFORE and AFTER are context only:",
        ]
    )
    for segment_id, segment in indexed_segments:
        marker = "TARGET" if segment_id in target_ids else ("BEFORE" if segment_id < min(target_ids) else "AFTER")
        lines.append(segment_prompt_line(marker, segment_id, segment, speakers))
    return "\n".join(lines)


def segment_reconciliation_batches(
    segments: List[Dict[str, Any]],
    chunks_per_batch: int = SEGMENT_RECONCILIATION_CHUNKS_PER_BATCH,
    context_segments: int = SEGMENT_RECONCILIATION_CONTEXT_SEGMENTS,
) -> List[Tuple[List[Tuple[int, Dict[str, Any]]], List[int]]]:
    chunk_order: List[int] = []
    for segment in segments:
        chunk_index = int(segment.get("chunk_index") or 0)
        if chunk_index not in chunk_order:
            chunk_order.append(chunk_index)

    batches: List[Tuple[List[Tuple[int, Dict[str, Any]]], List[int]]] = []
    for offset in range(0, len(chunk_order), chunks_per_batch):
        target_chunks = set(chunk_order[offset : offset + chunks_per_batch])
        target_ids = [
            index
            for index, segment in enumerate(segments)
            if int(segment.get("chunk_index") or 0) in target_chunks
        ]
        if not target_ids:
            continue
        context_start = max(0, target_ids[0] - context_segments)
        context_end = min(len(segments), target_ids[-1] + context_segments + 1)
        indexed_context = list(enumerate(segments[context_start:context_end], context_start))
        batches.append((indexed_context, target_ids))
    return batches


def apply_segment_assignments(
    segments: List[Dict[str, Any]],
    assignments: Sequence[Dict[str, Any]],
    expected_segment_ids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], int]:
    expected = set(expected_segment_ids)
    assigned: Dict[int, str] = {}
    for assignment in assignments:
        segment_id = int(assignment.get("segment_id"))
        if segment_id in assigned:
            raise RuntimeError(f"Duplicate segment attribution for segment {segment_id}")
        assigned[segment_id] = str(assignment.get("speaker_id") or "")
    if set(assigned) != expected:
        missing = sorted(expected - set(assigned))
        extra = sorted(set(assigned) - expected)
        raise RuntimeError(
            f"Segment attribution IDs do not match target batch; missing={missing[:10]}, extra={extra[:10]}"
        )

    reconciled = [dict(segment) for segment in segments]
    changed_count = 0
    for segment_id, speaker_id in assigned.items():
        segment = reconciled[segment_id]
        previous_id = str(segment.get("speaker_id") or "")
        segment["speaker_id_before_text_reconciliation"] = previous_id
        segment["speaker_id_source_before_text_reconciliation"] = segment.get("speaker_id_source")
        segment["speaker_id"] = speaker_id
        segment["speaker_id_source"] = "text_segment_reconciliation"
        if speaker_id != previous_id:
            changed_count += 1
    return reconciled, changed_count


def reconcile_segments_with_text_model(
    *,
    client: Any,
    model: str,
    canonical_url: str,
    title: str,
    description: str,
    source_language_hint: Optional[str],
    segments: List[Dict[str, Any]],
    speakers: List[Dict[str, str]],
    log: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    speaker_ids = [str(speaker.get("id")) for speaker in speakers if speaker.get("id")]
    if not segments or not speaker_ids:
        return [dict(segment) for segment in segments], {"status": "skipped", "batches": []}

    reconciled = [dict(segment) for segment in segments]
    batch_debug: List[Dict[str, Any]] = []
    total_changed = 0
    batches = segment_reconciliation_batches(reconciled)
    for batch_number, (indexed_context, target_ids) in enumerate(batches, 1):
        if log:
            log(
                f"Segment speaker reconciliation batch {batch_number}/{len(batches)} "
                f"({len(target_ids)} target segments)..."
            )
        result = ytranslate.call_openai_with_retry(
            client,
            model,
            build_segment_reconciliation_system_prompt(),
            build_segment_reconciliation_user_prompt(
                canonical_url=canonical_url,
                title=title,
                description=description,
                source_language_hint=source_language_hint,
                speakers=speakers,
                indexed_segments=indexed_context,
                target_segment_ids=target_ids,
            ),
            schema_name=f"segment_speaker_attribution_{batch_number}",
            schema=segment_reconciliation_schema(len(target_ids), speaker_ids),
            temperature=0.0,
        )
        reconciled, changed_count = apply_segment_assignments(
            reconciled,
            result.get("assignments", []),
            target_ids,
        )
        total_changed += changed_count
        batch_debug.append(
            {
                "batch": batch_number,
                "target_segment_count": len(target_ids),
                "first_segment_id": target_ids[0],
                "last_segment_id": target_ids[-1],
                "changed_count": changed_count,
            }
        )
    return reconciled, {
        "status": "ok",
        "batch_count": len(batches),
        "segment_count": len(reconciled),
        "changed_count": total_changed,
        "batches": batch_debug,
    }


def expected_label_matches(expected: str, label: str) -> bool:
    normalized = ytranslate.normalize_identity_text(label)
    expected_normalized = ytranslate.normalize_identity_text(expected)
    if expected_normalized == "friedberg":
        return "friedberg" in normalized or "freeberg" in normalized or "freiberg" in normalized
    if expected_normalized == "jason":
        return "jason" in normalized or "calacanis" in normalized
    if expected_normalized == "brad":
        return "brad" in normalized or "gerstner" in normalized
    return expected_normalized in normalized


def evaluate_known_attribution_windows(
    segments: List[Dict[str, Any]],
    speakers: List[Dict[str, str]],
) -> Dict[str, Any]:
    windows = []
    groups: Dict[str, List[bool]] = defaultdict(list)
    for expected_window in KNOWN_ATTRIBUTION_WINDOWS:
        duration_by_label: Dict[str, float] = defaultdict(float)
        total_duration = 0.0
        correct_duration = 0.0
        for segment in segments:
            start = float(segment.get("start") or 0)
            end = float(segment.get("end") or start)
            overlap = max(
                0.0,
                min(end, float(expected_window["end"]))
                - max(start, float(expected_window["start"])),
            )
            if overlap <= 0:
                continue
            label = speaker_label(speakers, str(segment.get("speaker_id") or ""))
            duration_by_label[label] += overlap
            total_duration += overlap
            if expected_label_matches(str(expected_window["expected"]), label):
                correct_duration += overlap
        accuracy = correct_duration / total_duration if total_duration else 0.0
        passed = accuracy >= 0.95
        groups[str(expected_window["group"])].append(passed)
        windows.append(
            {
                **expected_window,
                "accuracy": round(accuracy, 4),
                "passed": passed,
                "speaker_seconds": {
                    label: round(duration, 3)
                    for label, duration in sorted(
                        duration_by_label.items(), key=lambda item: item[1], reverse=True
                    )
                },
            }
        )
    return {
        "windows": windows,
        "groups": {group: all(results) for group, results in groups.items()},
        "all_groups_passed": all(all(results) for results in groups.values()),
    }


def run_variant(
    *,
    effort: str,
    variant_number: int,
    output_dir: Path,
    metadata: Dict[str, Any],
    canonical_url: str,
    video_id: str,
    asr_result: Dict[str, Any],
    openai_key: str,
    segment_reconciliation: bool = False,
    attribution_only: bool = False,
) -> Dict[str, Any]:
    recorder = RequestRecorder(effort)
    real_client = OpenAI(api_key=openai_key, timeout=ytranslate.OPENAI_TIMEOUT_SECONDS)
    client = InstrumentedClient(real_client, recorder)
    started = time.perf_counter()
    logs: List[str] = []

    def log(message: str) -> None:
        stamped = f"[{time.strftime('%H:%M:%S')}] {message}"
        logs.append(stamped)
        print(f"[{effort}] {message}", flush=True)

    title = metadata.get("title") or "Untitled"
    description = metadata.get("description", "")
    source_language_hint = metadata.get("defaultAudioLanguage") or metadata.get("defaultLanguage")
    raw_segments = asr_result.get("segments", [])
    diagnostics_dir = output_dir / "diagnostics" / effort
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    segment_debug: Dict[str, Any] = {"status": "disabled"}
    try:
        with recorder.phase("speaker_mapping"):
            speaker_mapping_model = ytranslate.assign_global_speakers_for_diarized_segments(
                client,
                MODEL,
                canonical_url,
                title,
                description,
                raw_segments,
                source_language_hint,
                metadata=metadata,
            )
            speaker_mapping = speaker_mapping_model
            known_speaker_roster = ytranslate.infer_known_speaker_roster(metadata)
            speaker_identity_evidence: Optional[Dict[str, Any]] = None
            if known_speaker_roster:
                speaker_identity_evidence = ytranslate.build_speaker_identity_evidence(
                    raw_segments,
                    known_speaker_roster,
                )
                speaker_mapping = ytranslate.apply_speaker_identity_evidence(
                    speaker_mapping,
                    speaker_identity_evidence,
                )
            speaker_overrides = ytranslate.load_speaker_mapping_overrides(video_id)
            if speaker_overrides:
                speaker_mapping = ytranslate.apply_speaker_mapping_overrides(
                    speaker_mapping,
                    speaker_overrides,
                )

        with recorder.phase("voice_reconciliation"):
            resolved_segments, voice_debug = ytranslate.reconcile_diarized_segments_with_voice(
                canonical_url,
                video_id,
                raw_segments,
                speaker_mapping,
                log,
            )

        if segment_reconciliation:
            with recorder.phase("segment_attribution"):
                resolved_segments, segment_debug = reconcile_segments_with_text_model(
                    client=client,
                    model=MODEL,
                    canonical_url=canonical_url,
                    title=title,
                    description=description,
                    source_language_hint=source_language_hint,
                    segments=resolved_segments,
                    speakers=speaker_mapping.get("speakers", []),
                    log=log,
                )

        with recorder.phase("turn_attribution"):
            resolved_segments, effective_speakers, role_merge_debug = (
                ytranslate.collapse_role_speaker_identities(
                    resolved_segments,
                    speaker_mapping.get("speakers", []),
                )
            )
            speaker_mapping_for_turns = dict(speaker_mapping)
            speaker_mapping_for_turns["speakers"] = effective_speakers
            attributed = ytranslate.attributed_turns_from_diarized_segments(
                resolved_segments,
                speaker_mapping_for_turns,
            )
            contradictions = ytranslate.find_speaker_identity_contradictions(
                attributed.get("speakers", []),
                attributed.get("turns", []),
            )
            attribution_evaluation = evaluate_known_attribution_windows(
                resolved_segments,
                effective_speakers,
            )

        if attribution_only:
            result = {
                "title_translated": title,
                "speakers": attributed.get("speakers", []),
                "turns": [
                    {**turn, "text_translated": turn.get("text_source") or ""}
                    for turn in attributed.get("turns", [])
                ],
            }
        else:
            with recorder.phase("translation"):
                result = ytranslate.translate_attributed_turns(
                    client,
                    MODEL,
                    canonical_url,
                    title,
                    description,
                    "Russian",
                    attributed.get("speakers", []),
                    attributed.get("turns", []),
                    source_language_hint,
                    log=log,
                )

            with recorder.phase("cleanup"):
                result["turns"] = ytranslate.cleanup_russian_turns(
                    client,
                    MODEL,
                    result.get("title_translated", "").strip() or title,
                    result.get("turns", []),
                    log=log,
                )

            with recorder.phase("annotation"):
                result["turns"] = ytranslate.annotate_russian_turns(
                    client,
                    MODEL,
                    result.get("title_translated", "").strip() or title,
                    result.get("turns", []),
                    log=log,
                )

        total_seconds = time.perf_counter() - started
        metrics = recorder.summarize()
        metrics["model"] = MODEL
        metrics["total_seconds"] = round(total_seconds, 3)
        metrics["source_segment_count"] = len(raw_segments)
        metrics["source_turn_count"] = len(attributed.get("turns", []))
        metrics["final_turn_count"] = len(result.get("turns", []))
        metrics["speaker_count"] = len(attributed.get("speakers", []))
        metrics["speaker_labels"] = [
            speaker.get("label_full") or speaker.get("label_short")
            for speaker in attributed.get("speakers", [])
        ]
        first_turn = (attributed.get("turns") or [{}])[0]
        opening_label = speaker_label(
            attributed.get("speakers", []),
            str(first_turn.get("speaker_id") or ""),
        )
        metrics["opening_speaker"] = opening_label
        metrics["opening_speaker_is_jason"] = "jason" in opening_label.lower()
        metrics["contradiction_count"] = len(contradictions)
        metrics["segment_reconciliation"] = segment_reconciliation
        metrics["attribution_only"] = attribution_only
        metrics["segment_reconciliation_changed_count"] = int(
            segment_debug.get("changed_count") or 0
        )
        metrics["known_attribution_groups"] = attribution_evaluation.get("groups", {})
        metrics["known_attribution_all_passed"] = attribution_evaluation.get(
            "all_groups_passed", False
        )
        metrics["pricing"] = {
            "input_per_million": INPUT_PRICE_PER_MILLION,
            "cached_input_per_million": CACHED_INPUT_PRICE_PER_MILLION,
            "output_per_million": OUTPUT_PRICE_PER_MILLION,
        }

        strategy_suffix = "-segment-reconciled" if segment_reconciliation else ""
        if attribution_only:
            strategy_suffix += "-attribution-only"
        prefix = f"{variant_number:02d}-{MODEL}-{effort}{strategy_suffix}"
        title_translated = result.get("title_translated", "").strip() or title
        markdown = ytranslate.render_markdown_transcript(
            title_translated,
            result.get("speakers", []),
            result.get("turns", []),
            metadata={
                "model": MODEL,
                "reasoning_effort": effort,
                "total_seconds": metrics["total_seconds"],
                "api_seconds": metrics["api_seconds"],
                "input_tokens": metrics["input_tokens"],
                "cached_input_tokens": metrics["cached_input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "reasoning_tokens": metrics["reasoning_tokens"],
                "estimated_text_cost_usd": metrics["estimated_cost_usd"],
                "opening_speaker": opening_label,
                "speaker_strategy": (
                    "segment reconciliation" if segment_reconciliation else "local and voice mapping"
                ),
            },
        )
        markdown_path = output_dir / f"{prefix}.md"
        ytranslate.write_text_file(str(markdown_path), markdown)
        pdf_path = render_variant_pdf(
            output_dir,
            prefix,
            title_translated,
            result.get("speakers", []),
            result.get("turns", []),
        )

        ytranslate.write_json_file(
            str(diagnostics_dir / "speaker-mapping-model.json"),
            speaker_mapping_model,
        )
        ytranslate.write_json_file(
            str(diagnostics_dir / "speaker-mapping-effective.json"),
            speaker_mapping,
        )
        if speaker_identity_evidence is not None:
            ytranslate.write_json_file(
                str(diagnostics_dir / "speaker-identity-evidence.json"),
                ytranslate.serialize_speaker_identity_evidence(speaker_identity_evidence),
            )
        ytranslate.write_json_file(
            str(diagnostics_dir / "voice-reconciliation.json"),
            {**voice_debug, "role_speaker_identity_merge": role_merge_debug},
        )
        ytranslate.write_json_file(
            str(diagnostics_dir / "segment-reconciliation.json"),
            segment_debug,
        )
        ytranslate.write_json_file(
            str(diagnostics_dir / "resolved-segments.json"),
            resolved_segments,
        )
        ytranslate.write_json_file(
            str(diagnostics_dir / "known-attribution-windows.json"),
            attribution_evaluation,
        )
        ytranslate.write_json_file(
            str(diagnostics_dir / "source-attributed-turns.json"),
            attributed,
        )
        ytranslate.write_text_file(
            str(diagnostics_dir / "source-attributed-turns.md"),
            render_source_markdown(
                title,
                attributed.get("speakers", []),
                attributed.get("turns", []),
                {
                    "model": MODEL,
                    "reasoning_effort": effort,
                    "purpose": "speaker mapping audit before translation",
                },
            ),
        )
        ytranslate.write_json_file(
            str(diagnostics_dir / "speaker-identity-contradictions.json"),
            contradictions,
        )
        ytranslate.write_json_file(str(diagnostics_dir / "final.json"), result)
        ytranslate.write_json_file(str(diagnostics_dir / "metrics.json"), metrics)
        ytranslate.write_text_file(str(diagnostics_dir / "run.log"), "\n".join(logs) + "\n")
        return {
            "model": MODEL,
            "reasoning_effort": effort,
            "segment_reconciliation": segment_reconciliation,
            "attribution_only": attribution_only,
            "markdown_path": str(markdown_path),
            "pdf_path": str(pdf_path),
            "diagnostics_dir": str(diagnostics_dir),
            **{key: value for key, value in metrics.items() if key != "requests"},
        }
    finally:
        real_client.close()


def write_summary(
    output_dir: Path,
    episode: Dict[str, Any],
    variants: List[Dict[str, Any]],
    baseline_seconds: Optional[float],
) -> None:
    rows = []
    if baseline_seconds is not None:
        rows.append(
            {
                "model": "gpt-5.4-mini",
                "effort": "default (none)",
                "seconds": baseline_seconds,
                "api_seconds": "not recorded",
                "input_tokens": "not recorded",
                "output_tokens": "not recorded",
                "reasoning_tokens": "not recorded",
                "cost": "not recorded",
                "source_turns": "not recorded",
                "contradictions": "not recorded",
                "opening_speaker": "Brad Gerstner (known incorrect)",
            }
        )
    for variant in variants:
        effort_label = variant["reasoning_effort"]
        if variant.get("sample"):
            effort_label = f"{effort_label} ({variant['sample']})"
        rows.append(
            {
                "model": variant["model"],
                "effort": effort_label,
                "seconds": variant["total_seconds"],
                "api_seconds": variant["api_seconds"],
                "input_tokens": variant["input_tokens"],
                "output_tokens": variant["output_tokens"],
                "reasoning_tokens": variant["reasoning_tokens"],
                "cost": variant["estimated_cost_usd"],
                "source_turns": variant["source_turn_count"],
                "contradictions": variant["contradiction_count"],
                "opening_speaker": variant["opening_speaker"],
            }
        )

    csv_path = output_dir / "timings-and-usage.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# GPT-5.6 Luna text-pipeline comparison",
        "",
        f"- **Episode**: {episode['title']}",
        f"- **URL**: {episode['url']}",
        f"- **Video ID**: {episode['video_id']}",
        "- **Controlled input**: the same cached GPT-4o diarized ASR segments for every run",
        "- **Pipeline**: global speaker reconciliation, local voice reconciliation, translation, Russian cleanup, glossary annotations",
        "",
        "## Files",
        "",
        "- `00-gpt-5.4-mini-baseline.md/pdf`: current production result",
    ]
    for variant in variants:
        stem = Path(variant["markdown_path"]).stem
        effort = variant["reasoning_effort"]
        sample = variant.get("sample")
        sample_suffix = f" ({sample})" if sample else ""
        lines.append(f"- `{stem}.md/pdf`: Luna {effort} reasoning{sample_suffix}")
    lines.extend(
        [
            "- `diagnostics/<effort>/source-attributed-turns.md`: speaker mapping before translation",
            "",
            "## Timing and usage",
            "",
            "| Model | Effort | Total seconds | API seconds | Input tokens | Output tokens | Reasoning tokens | Estimated text cost | Source turns | Contradictions | Opening speaker |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        cost = row["cost"]
        cost_text = f"${float(cost):.4f}" if isinstance(cost, (int, float)) else str(cost)
        lines.append(
            f"| {row['model']} | {row['effort']} | {row['seconds']} | {row['api_seconds']} | "
            f"{row['input_tokens']} | {row['output_tokens']} | {row['reasoning_tokens']} | "
            f"{cost_text} | {row['source_turns']} | {row['contradictions']} | {row['opening_speaker']} |"
        )
    lines.extend(
        [
            "",
            "## Known baseline defect",
            "",
            "The production baseline labels the opening host as Brad Gerstner even though the host welcomes Brad to the program. "
            "The `opening_speaker` column is therefore a direct first-pass check, not a complete quality score.",
            "",
            "## Attribution audit",
            "",
            "Every evaluated Luna effort labels the opening host as Jason Calacanis, but manual listening found material differences later in the episode.",
        ]
    )
    efforts = {variant["reasoning_effort"] for variant in variants}
    medium_samples = [
        variant for variant in variants if variant["reasoning_effort"] == "medium"
    ]
    if {"none", "low", "medium", "high"}.issubset(efforts) and len(medium_samples) >= 2:
        lines.extend(
            [
                "",
                "The first medium sample was correct on two substantial Brad Gerstner transitions that none, low, and high misattributed to Jason. "
                "A second controlled medium sample failed both transitions, so that apparent effort-level advantage did not reproduce.",
                "",
                "The first medium sample selected a clean later Brad cluster from chunk 4 as its voice anchor and recovered both boundaries. "
                "The rerun instead created a generic `speaker_2` identity labeled Brad and anchored it to chunk-2 local speaker D, which is actually a long Friedberg passage about mixing open and premium models. "
                "Its reconciliation then merged both checked Brad answers into Jason.",
                "",
                "The instability is visible in the aggregate output: the first medium sample produced 406 source turns and two contradictions, while the rerun produced 300 turns and three contradictions. "
                "Voice reconciliation also changed 50 segments in the first sample versus 92 in the rerun.",
                "",
                "The defensible conclusion is therefore not that medium is strongest. "
                "Speaker attribution is sampling-sensitive because the text model chooses the identity and voice-anchor clusters; a lucky clean anchor can repair diarization, while a bad anchor propagates errors across the episode.",
                "",
                "High remains operationally unattractive: it took 3,850.8 seconds and cost an estimated $0.3903 without fixing either checked transition. "
                "The medium rerun took 615.4 seconds and cost an estimated $0.1363.",
            ]
        )
    elif {"none", "low", "medium", "high"}.issubset(efforts):
        lines.extend(
            [
                "",
                "Medium is correct on two substantial Brad Gerstner transitions that none, low, and high misattribute to Jason. "
                "The first is Brad's response after Friedberg's infrastructure argument (`Brad, what's your take on this?`). "
                "The second is Brad's answer after Jason asks about downward token-price pressure (`America is winning`).",
                "",
                "The difference comes from voice-anchor selection, not the raw diarizer alone. "
                "Medium anchors Brad to a later, cleaner chunk-4 cluster and uses voice similarity to repair the mixed early clusters. "
                "High anchors Brad to chunk 2, whose local speaker B contains Friedberg, a Sacks interjection, and Brad, so the contaminated anchor collapses both long Brad turns into Jason.",
                "",
                "Medium's 406 source turns versus 316 for none and high are therefore not evidence of fragmentation by themselves; the confirmed examples show that some additional turns are correctly recovered speaker boundaries. "
                "All four outputs still retain two detected self-address contradictions, so medium is better on the checked attribution cases rather than globally error-free.",
                "",
                "High took 3,850.8 seconds and cost an estimated $0.3903, versus 608.0 seconds and $0.1448 for medium. "
                "It is about 6.3 times slower and 2.7 times as expensive without fixing either checked transition.",
            ]
        )
    lines.extend(
        [
            "",
            "The diagnostics retain the complete mappings and pre-translation transcript so other speaker errors can be audited independently of Russian prose quality.",
            "",
        ]
    )
    ytranslate.write_text_file(str(output_dir / "README.md"), "\n".join(lines))

    html_rows = []
    for row in rows:
        cost = row["cost"]
        cost_text = f"${float(cost):.4f}" if isinstance(cost, (int, float)) else str(cost)
        html_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    row["model"],
                    row["effort"],
                    row["seconds"],
                    row["api_seconds"],
                    row["input_tokens"],
                    row["output_tokens"],
                    row["reasoning_tokens"],
                    cost_text,
                    row["source_turns"],
                    row["contradictions"],
                    row["opening_speaker"],
                )
            )
            + "</tr>"
        )
    file_links = []
    for path in sorted(output_dir.glob("*.pdf")) + sorted(output_dir.glob("*.md")):
        file_links.append(
            f'<li><a href="{html.escape(path.name)}">{html.escape(path.name)}</a></li>'
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPT-5.6 Luna comparison</title>
<style>
body{{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1d2329;background:#f4f5f6;margin:0}}
main{{max-width:1180px;margin:0 auto;padding:32px}} h1{{font-size:25px;margin:0 0 8px}} p{{color:#59636e}}
section{{background:#fff;border:1px solid #dfe3e7;border-radius:6px;padding:20px;margin:18px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:9px;border-bottom:1px solid #e5e8eb}}
th{{color:#59636e;font-weight:600}} a{{color:#135f9c}} ul{{columns:2;padding-left:20px}}
</style></head><body><main><h1>GPT-5.6 Luna text-pipeline comparison</h1>
<p>{html.escape(episode['title'])}</p>
<section><h2>Runs</h2><table><thead><tr><th>Model</th><th>Effort</th><th>Total s</th><th>API s</th><th>Input</th><th>Output</th><th>Reasoning</th><th>Cost</th><th>Turns</th><th>Contradictions</th><th>Opening speaker</th></tr></thead>
<tbody>{''.join(html_rows)}</tbody></table></section>
<section><h2>Comparison files</h2><ul>{''.join(file_links)}</ul></section>
<section><h2>Interpretation</h2><p>All Luna efforts fix the opening Jason/Brad error. The first medium sample recovered two substantial Brad transitions, but an identical medium rerun missed both and used a Friedberg passage as Brad's voice anchor. Speaker attribution is therefore sampling-sensitive rather than reliably improved by medium reasoning. High is about 6.3 times slower than the first medium sample and also fails both checks. Complete pre-translation evidence lives under <code>diagnostics/</code>.</p></section>
</main></body></html>"""
    ytranslate.write_text_file(str(output_dir / "index.html"), page)


def write_segment_reconciliation_summary(
    output_dir: Path,
    episode: Dict[str, Any],
    variants: List[Dict[str, Any]],
) -> None:
    lines = [
        "# GPT-5.6 Luna segment-level speaker reconciliation",
        "",
        f"- **Episode**: {episode['title']}",
        f"- **URL**: {episode['url']}",
        f"- **Video ID**: {episode['video_id']}",
        "- **Controlled input**: identical cached GPT-4o diarized ASR for every effort",
        "- **General fix under test**: final attribution per ASR segment; local diarizer and voice labels are hints rather than one-speaker constraints",
        "",
        "## Results",
        "",
        "| Effort | Total s | Segment-pass s | Changed segments | Page 5 | Pages 11-12 | All checks | Contradictions | Estimated text cost |",
        "|---|---:|---:|---:|---|---|---|---:|---:|",
    ]
    csv_rows = []
    for variant in variants:
        metrics_path = Path(variant["diagnostics_dir"]) / "metrics.json"
        metrics = ytranslate.read_json_file(str(metrics_path))
        groups = metrics.get("known_attribution_groups", {})
        segment_seconds = (
            metrics.get("phase_seconds", {}).get("segment_attribution", 0.0)
        )
        row = {
            "effort": variant["reasoning_effort"],
            "total_seconds": metrics.get("total_seconds"),
            "segment_pass_seconds": segment_seconds,
            "changed_segments": metrics.get("segment_reconciliation_changed_count"),
            "page_5": bool(groups.get("page_5")),
            "pages_11_12": bool(groups.get("pages_11_12")),
            "all_checks": bool(metrics.get("known_attribution_all_passed")),
            "contradictions": metrics.get("contradiction_count"),
            "estimated_text_cost_usd": metrics.get("estimated_cost_usd"),
        }
        csv_rows.append(row)
        lines.append(
            f"| {row['effort']} | {row['total_seconds']} | {row['segment_pass_seconds']} | "
            f"{row['changed_segments']} | {'PASS' if row['page_5'] else 'FAIL'} | "
            f"{'PASS' if row['pages_11_12'] else 'FAIL'} | "
            f"{'PASS' if row['all_checks'] else 'FAIL'} | {row['contradictions']} | "
            f"${float(row['estimated_text_cost_usd'] or 0):.4f} |"
        )

    lines.extend(["", "## Window Detail", ""])
    for variant in variants:
        effort = variant["reasoning_effort"]
        evaluation_path = Path(variant["diagnostics_dir"]) / "known-attribution-windows.json"
        evaluation = ytranslate.read_json_file(str(evaluation_path))
        lines.extend(
            [
                f"### {effort}",
                "",
                "| Window | Expected | Accuracy | Assigned speaker seconds |",
                "|---|---|---:|---|",
            ]
        )
        for window in evaluation.get("windows", []):
            assignments = ", ".join(
                f"{label}: {seconds:.1f}s"
                for label, seconds in window.get("speaker_seconds", {}).items()
            )
            lines.append(
                f"| {window['name']} | {window['expected']} | "
                f"{float(window.get('accuracy') or 0) * 100:.1f}% | {assignments} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Files",
            "",
        ]
    )
    for variant in variants:
        lines.append(f"- `{Path(variant['pdf_path']).name}`")
        lines.append(
            f"- `diagnostics/{variant['reasoning_effort']}/source-attributed-turns.md`"
        )
    lines.extend(
        [
            "",
            "The pass/fail checks cover the two handoffs manually verified against audio. They are regression probes, not a complete episode-wide ground truth.",
            "",
        ]
    )
    ytranslate.write_text_file(str(output_dir / "README.md"), "\n".join(lines))

    csv_path = output_dir / "segment-reconciliation-results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


def copy_baseline(
    output_dir: Path,
    baseline_pdf: Optional[Path],
    baseline_docx: Optional[Path],
    title: str,
) -> None:
    if baseline_pdf:
        shutil.copy2(baseline_pdf, output_dir / "00-gpt-5.4-mini-baseline.pdf")
    if baseline_docx:
        markdown = baseline_docx_to_markdown(baseline_docx, title)
        ytranslate.write_text_file(
            str(output_dir / "00-gpt-5.4-mini-baseline.md"),
            markdown,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="muRIXCDw-k0")
    parser.add_argument("--url", default="https://youtu.be/muRIXCDw-k0")
    parser.add_argument(
        "--efforts",
        nargs="+",
        default=list(DEFAULT_EFFORTS),
        choices=VALID_EFFORTS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument("--baseline-pdf", type=Path)
    parser.add_argument("--baseline-docx", type=Path)
    parser.add_argument("--baseline-seconds", type=float, default=397.2)
    parser.add_argument(
        "--segment-reconciliation",
        action="store_true",
        help="Run a segment-level speaker-attribution pass after local voice reconciliation.",
    )
    parser.add_argument(
        "--attribution-only",
        action="store_true",
        help="Stop after speaker attribution and render the source transcript without translation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ytranslate.load_project_env()
    openai_key = os.getenv("OPENAI_API_KEY")
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if not youtube_key:
        raise RuntimeError("YOUTUBE_API_KEY is not set")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = ytranslate.fetch_video_metadata(args.video_id, youtube_key)
    asr_path = (
        Path(ytranslate.get_video_cache_dir(args.video_id))
        / f"openai-asr-{ytranslate.OPENAI_ASR_MODEL}-{ytranslate.ASR_CHUNK_SECONDS}s.json"
    )
    if not asr_path.exists():
        raise RuntimeError(f"Cached ASR aggregate not found: {asr_path}")
    asr_result = ytranslate.read_json_file(str(asr_path))

    copy_baseline(
        output_dir,
        args.baseline_pdf.expanduser().resolve() if args.baseline_pdf else None,
        args.baseline_docx.expanduser().resolve() if args.baseline_docx else None,
        metadata.get("title") or "Untitled",
    )

    variants = []
    for effort in args.efforts:
        variant_number = VALID_EFFORTS.index(effort) + 1
        variants.append(
            run_variant(
                effort=effort,
                variant_number=variant_number,
                output_dir=output_dir,
                metadata=metadata,
                canonical_url=args.url,
                video_id=args.video_id,
                asr_result=asr_result,
                openai_key=openai_key,
                segment_reconciliation=args.segment_reconciliation,
                attribution_only=args.attribution_only,
            )
        )
        ytranslate.write_json_file(str(output_dir / "results-partial.json"), variants)

    episode = {
        "video_id": args.video_id,
        "url": args.url,
        "title": metadata.get("title") or "Untitled",
        "asr_path": str(asr_path),
    }
    if args.segment_reconciliation:
        write_segment_reconciliation_summary(output_dir, episode, variants)
    else:
        write_summary(output_dir, episode, variants, args.baseline_seconds)
    ytranslate.write_json_file(
        str(output_dir / "results.json"),
        {"episode": episode, "variants": variants},
    )
    partial_path = output_dir / "results-partial.json"
    partial_path.unlink(missing_ok=True)
    print(json.dumps({"output_dir": str(output_dir), "variants": variants}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
