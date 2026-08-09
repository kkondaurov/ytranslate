#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from gpt_transcribe_eval import (
    CACHE_ROOT,
    DIARIZE_MODEL,
    DIARIZE_PRICE_PER_MINUTE,
    ROOT,
    boundary_score,
    diarize_with_references_fields,
    extract_audio,
    load_dotenv,
    raw_segments,
    read_json,
    probe_duration_seconds,
    safe_request_fields,
    write_json,
    SpeakerReference,
)
from openai import OpenAI


DEFAULT_OUTPUT_DIR = (
    ROOT / "experiments" / "gpt_transcribe_eval" / "output" / "known-speaker-matrix"
)
ALL_SPEAKERS = {"Jason", "Sacks", "Friedberg", "Chamath", "Brad", "Gavin"}


@dataclass(frozen=True)
class VoiceClip:
    speaker: str
    video_id: str
    start: float
    end: float
    provenance: str


@dataclass(frozen=True)
class TruthSpan:
    start: float
    end: float
    speaker: str
    group: str


@dataclass(frozen=True)
class TargetWindow:
    label: str
    start: float
    end: float


@dataclass(frozen=True)
class EpisodeSample:
    key: str
    video_id: str
    clip_start: float
    clip_end: float
    truth_spans: tuple[TruthSpan, ...]
    target_windows: tuple[TargetWindow, ...]


@dataclass(frozen=True)
class Condition:
    key: str
    references: tuple[VoiceClip, ...]


@dataclass(frozen=True)
class RunSpec:
    sample: EpisodeSample
    condition: Condition
    repeat: int


CROSS_REFERENCES = {
    "Jason": VoiceClip("Jason", "wcV0SRPFK9s", 6.300, 12.250, "cross-episode"),
    "Sacks": VoiceClip("Sacks", "wcV0SRPFK9s", 183.796, 190.046, "cross-episode"),
    "Friedberg": VoiceClip(
        "Friedberg", "wcV0SRPFK9s", 372.180, 380.730, "cross-episode"
    ),
    "Chamath": VoiceClip(
        "Chamath", "ViqYWhLimGg", 2803.264, 2811.764, "cross-episode-user-audited"
    ),
    "Brad": VoiceClip("Brad", "PHL1j2ti420", 452.780, 460.980, "cross-episode"),
    "Gavin": VoiceClip("Gavin", "NGsi2PC4y68", 183.924, 192.474, "cross-episode"),
}


ALTERNATE_CROSS_REFERENCES = {
    "Jason": VoiceClip("Jason", "ViqYWhLimGg", 84.148, 90.648, "alternate-cross-episode"),
    "Sacks": VoiceClip("Sacks", "ViqYWhLimGg", 3010.640, 3016.840, "alternate-cross-episode"),
    "Friedberg": VoiceClip(
        "Friedberg", "ViqYWhLimGg", 438.854, 444.504, "alternate-cross-episode"
    ),
    "Chamath": VoiceClip(
        "Chamath", "ViqYWhLimGg", 2803.264, 2811.764, "alternate-cross-user-audited"
    ),
    "Brad": CROSS_REFERENCES["Brad"],
    "Gavin": CROSS_REFERENCES["Gavin"],
}


CURRENT_SAME_REFERENCES = {
    "Jason": VoiceClip("Jason", "muRIXCDw-k0", 151.670, 160.420, "same-episode"),
    "Friedberg": VoiceClip(
        "Friedberg", "muRIXCDw-k0", 257.136, 264.186, "same-episode"
    ),
    "Brad": VoiceClip("Brad", "muRIXCDw-k0", 459.152, 465.952, "same-episode"),
    "Sacks": VoiceClip("Sacks", "muRIXCDw-k0", 720.858, 728.958, "same-episode"),
}


AUDITED_SAME_REFERENCES = {
    "Jason": VoiceClip("Jason", "HGbA6ze0_3M", 91.668, 99.068, "same-episode"),
    "Chamath": VoiceClip("Chamath", "HGbA6ze0_3M", 193.252, 198.252, "same-episode"),
    "Gavin": VoiceClip("Gavin", "HGbA6ze0_3M", 414.832, 422.482, "same-episode"),
    "Friedberg": VoiceClip(
        "Friedberg", "HGbA6ze0_3M", 517.172, 523.022, "same-episode"
    ),
}


CURRENT_TRUTH = (
    TruthSpan(5.008, 29.414, "Friedberg", "page-5-handoff"),
    TruthSpan(29.414, 31.114, "Jason", "page-5-handoff"),
    TruthSpan(31.264, 136.222, "Brad", "page-5-handoff"),
    TruthSpan(136.222, 151.614, "Friedberg", "page-5-handoff"),
    TruthSpan(152.114, 194.212, "Jason", "post-page-5"),
    TruthSpan(194.212, 351.518, "Sacks", "sacks-anchor"),
    TruthSpan(680.342, 693.170, "Friedberg", "pages-11-12-handoff"),
    TruthSpan(693.370, 718.830, "Jason", "pages-11-12-handoff"),
    TruthSpan(719.744, 815.000, "Brad", "pages-11-12-handoff"),
)


CURRENT_SAMPLE = EpisodeSample(
    key="current-brad",
    video_id="muRIXCDw-k0",
    clip_start=400.0,
    clip_end=1215.0,
    truth_spans=CURRENT_TRUTH,
    target_windows=(
        TargetWindow("page-5 Friedberg-Jason-Brad", 20.0, 145.0),
        TargetWindow("pages-11-12 Friedberg-Jason-Brad", 680.0, 815.0),
    ),
)


def source_audio_path(video_id: str) -> Path:
    candidates = sorted((CACHE_ROOT / video_id / "audio").glob("source.*"))
    candidates = [path for path in candidates if not path.name.endswith(".part")]
    if not candidates:
        raise FileNotFoundError(f"No cached source audio for {video_id}")
    return candidates[0]


def load_audited_truth() -> tuple[TruthSpan, ...]:
    path = ROOT / "experiments" / "speaker_audit" / "output" / "HGbA6ze0_3M" / "audit-rows.json"
    spans = []
    for row in read_json(path):
        if int(row.get("chunk_index") or 0) != 1:
            continue
        start = float(row.get("start") or 0)
        end = float(row.get("end") or start)
        if end <= 390.0 or start >= 605.0:
            continue
        spans.append(
            TruthSpan(
                max(start, 390.0) - 390.0,
                min(end, 605.0) - 390.0,
                str(row.get("mapped_speaker_label") or "unknown"),
                "audited-panel",
            )
        )
    return tuple(spans)


AUDITED_SAMPLE = EpisodeSample(
    key="audited-gavin",
    video_id="HGbA6ze0_3M",
    clip_start=390.0,
    clip_end=605.0,
    truth_spans=load_audited_truth(),
    target_windows=(
        TargetWindow("Gavin-Jason handoff", 20.0, 100.0),
        TargetWindow("Chamath interruption", 185.0, 215.0),
    ),
)


SAMPLES = (CURRENT_SAMPLE, AUDITED_SAMPLE)


def refs(names: Iterable[str], library: dict[str, VoiceClip]) -> tuple[VoiceClip, ...]:
    return tuple(library[name] for name in names)


def conditions_for(sample: EpisodeSample) -> tuple[Condition, ...]:
    if sample.key == "current-brad":
        return (
            Condition("none", ()),
            Condition("cross-present3", refs(("Jason", "Sacks", "Friedberg"), CROSS_REFERENCES)),
            Condition(
                "cross-present3-plus-absent-chamath",
                refs(("Jason", "Sacks", "Friedberg", "Chamath"), CROSS_REFERENCES),
            ),
            Condition(
                "cross-actual4",
                refs(("Jason", "Sacks", "Friedberg", "Brad"), CROSS_REFERENCES),
            ),
            Condition(
                "alternate-cross-actual4",
                refs(
                    ("Jason", "Sacks", "Friedberg", "Brad"),
                    ALTERNATE_CROSS_REFERENCES,
                ),
            ),
            Condition(
                "same-actual4",
                refs(("Jason", "Sacks", "Friedberg", "Brad"), CURRENT_SAME_REFERENCES),
            ),
        )
    return (
        Condition("none", ()),
        Condition("cross-present3", refs(("Jason", "Chamath", "Friedberg"), CROSS_REFERENCES)),
        Condition(
            "cross-present3-plus-absent-sacks",
            refs(("Jason", "Chamath", "Friedberg", "Sacks"), CROSS_REFERENCES),
        ),
        Condition(
            "cross-actual4",
            refs(("Jason", "Chamath", "Friedberg", "Gavin"), CROSS_REFERENCES),
        ),
        Condition(
            "alternate-cross-actual4",
            refs(
                ("Jason", "Chamath", "Friedberg", "Gavin"),
                ALTERNATE_CROSS_REFERENCES,
            ),
        ),
        Condition(
            "same-actual4",
            refs(("Jason", "Chamath", "Friedberg", "Gavin"), AUDITED_SAME_REFERENCES),
        ),
    )


def overlap_seconds(left: dict[str, Any] | TruthSpan, right: dict[str, Any]) -> float:
    left_start = float(left.start if isinstance(left, TruthSpan) else left["start"])
    left_end = float(left.end if isinstance(left, TruthSpan) else left["end"])
    return max(0.0, min(left_end, float(right["end"])) - max(left_start, float(right["start"])))


def truth_speakers(spans: Sequence[TruthSpan]) -> set[str]:
    return {span.speaker for span in spans}


def score_open_set(
    truth: Sequence[TruthSpan],
    candidate: Sequence[dict[str, Any]],
    provided_names: Iterable[str],
) -> dict[str, Any]:
    provided = set(provided_names)
    confusion: dict[str, dict[str, float]] = {}
    for expected in truth:
        row = confusion.setdefault(expected.speaker, {})
        for actual in candidate:
            overlap = overlap_seconds(expected, actual)
            if overlap > 0:
                label = str(actual["speaker"])
                row[label] = row.get(label, 0.0) + overlap

    known_correct = 0.0
    known_scored = 0.0
    unreferenced_scored = 0.0
    unreferenced_false_known = 0.0
    absent_names = provided - truth_speakers(truth)
    absent_false_positive = 0.0
    oracle_correct = 0.0
    total = 0.0

    anonymous_map: dict[str, str] = {}
    anonymous_labels = {
        label
        for row in confusion.values()
        for label in row
        if label not in ALL_SPEAKERS
    }
    for label in anonymous_labels:
        choices = {
            expected: row.get(label, 0.0)
            for expected, row in confusion.items()
        }
        anonymous_map[label] = max(choices, key=choices.get)

    for expected, row in confusion.items():
        for actual, seconds in row.items():
            total += seconds
            if expected in provided:
                known_scored += seconds
                if actual == expected:
                    known_correct += seconds
            else:
                unreferenced_scored += seconds
                if actual in provided:
                    unreferenced_false_known += seconds
            if actual in absent_names:
                absent_false_positive += seconds
            mapped = actual if actual in ALL_SPEAKERS else anonymous_map.get(actual)
            if mapped == expected:
                oracle_correct += seconds

    fragmentation = {}
    dominant_share = {}
    for expected, row in confusion.items():
        material = {label: seconds for label, seconds in row.items() if seconds >= 0.5}
        fragmentation[expected] = len(material)
        scored = sum(row.values())
        dominant_share[expected] = round(max(row.values(), default=0.0) / scored, 4) if scored else None

    def ratio(numerator: float, denominator: float) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    return {
        "provided_names": sorted(provided),
        "truth_speakers": sorted(truth_speakers(truth)),
        "confusion_seconds": {
            expected: {actual: round(seconds, 3) for actual, seconds in sorted(row.items())}
            for expected, row in sorted(confusion.items())
        },
        "known_named_accuracy": ratio(known_correct, known_scored),
        "known_scored_seconds": round(known_scored, 3),
        "unreferenced_false_known_rate": ratio(unreferenced_false_known, unreferenced_scored),
        "unreferenced_scored_seconds": round(unreferenced_scored, 3),
        "absent_reference_names": sorted(absent_names),
        "absent_reference_false_positive_seconds": round(absent_false_positive, 3),
        "oracle_identity_accuracy": ratio(oracle_correct, total),
        "scored_seconds": round(total, 3),
        "anonymous_label_map": anonymous_map,
        "fragmentation_labels_per_speaker": fragmentation,
        "dominant_label_share_per_speaker": dominant_share,
    }


def clip_candidate_segments(
    segments: Sequence[dict[str, Any]], start: float, end: float
) -> list[dict[str, Any]]:
    clipped = []
    for segment in segments:
        if float(segment["end"]) <= start or float(segment["start"]) >= end:
            continue
        item = dict(segment)
        item["start"] = max(start, float(segment["start"]))
        item["end"] = min(end, float(segment["end"]))
        clipped.append(item)
    return clipped


def grouped_boundary_scores(
    truth: Sequence[TruthSpan], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    groups = sorted(
        group
        for group in {span.group for span in truth}
        if len({span.speaker for span in truth if span.group == group}) > 1
    )
    results = {}
    for group in groups:
        expected_spans = [span for span in truth if span.group == group]
        expected = [
            {"start": span.start, "end": span.end, "speaker": span.speaker}
            for span in expected_spans
        ]
        start = min(span.start for span in expected_spans)
        end = max(span.end for span in expected_spans)
        actual = clip_candidate_segments(candidate, start, end)
        results[group] = boundary_score(expected, actual, tolerance_seconds=2.0)
    f1_values = [result["f1"] for result in results.values()]
    return {
        "groups": results,
        "mean_f1": round(statistics.mean(f1_values), 4) if f1_values else None,
    }


def target_segments(
    sample: EpisodeSample, candidate: Sequence[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    return {
        target.label: clip_candidate_segments(candidate, target.start, target.end)
        for target in sample.target_windows
    }


def prepare_reference(clip: VoiceClip, output_dir: Path) -> tuple[SpeakerReference, Path]:
    key = f"{clip.speaker.lower()}-{clip.video_id}-{clip.start:.3f}-{clip.end:.3f}.wav"
    destination = output_dir / "assets" / "references" / key
    extract_audio(source_audio_path(clip.video_id), destination, clip.start, clip.end, "wav")
    return (
        SpeakerReference(clip.speaker, clip.speaker, clip.start, clip.end),
        destination,
    )


def prepare_sample(sample: EpisodeSample, output_dir: Path) -> Path:
    destination = output_dir / "assets" / f"{sample.key}.mp3"
    extract_audio(
        source_audio_path(sample.video_id),
        destination,
        sample.clip_start,
        sample.clip_end,
        "mp3",
    )
    return destination


def sdk_cached_api_call(
    output_path: Path,
    audio_path: Path,
    api_key: str,
    fields: Sequence[tuple[str, str]],
    force: bool,
) -> dict[str, Any]:
    if output_path.exists() and not force:
        return read_json(output_path)

    values: dict[str, list[str]] = {}
    for name, value in fields:
        values.setdefault(name, []).append(value)
    client = OpenAI(api_key=api_key, timeout=900.0, max_retries=5)
    started = time.monotonic()
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model=values["model"][0],
            response_format=values["response_format"][0],
            chunking_strategy=values["chunking_strategy"][0],
            known_speaker_names=values.get("known_speaker_names[]", []),
            known_speaker_references=values.get("known_speaker_references[]", []),
        )
    elapsed = time.monotonic() - started
    payload = response.model_dump(mode="json")
    duration = probe_duration_seconds(audio_path)
    result = {
        "request": {
            "audio": str(audio_path),
            "fields": safe_request_fields(fields),
            "transport": "openai-python",
        },
        "duration_seconds": round(duration, 3),
        "elapsed_seconds": round(elapsed, 3),
        "estimated_cost_usd": round(duration / 60 * DIARIZE_PRICE_PER_MINUTE, 6),
        "response": payload,
    }
    write_json(output_path, result)
    return result


def run_spec(
    spec: RunSpec,
    output_dir: Path,
    api_key: str,
    force: bool,
) -> dict[str, Any]:
    sample_audio = prepare_sample(spec.sample, output_dir)
    prepared = [prepare_reference(clip, output_dir) for clip in spec.condition.references]
    fields = diarize_with_references_fields(prepared)
    output_path = (
        output_dir
        / "api"
        / spec.sample.key
        / spec.condition.key
        / f"run-{spec.repeat:02d}.json"
    )
    call = sdk_cached_api_call(
        output_path,
        sample_audio,
        api_key,
        fields,
        force,
    )
    segments = raw_segments(call["response"])
    metrics = score_open_set(
        spec.sample.truth_spans,
        segments,
        (clip.speaker for clip in spec.condition.references),
    )
    boundaries = grouped_boundary_scores(spec.sample.truth_spans, segments)
    evaluation = {
        "sample": spec.sample.key,
        "video_id": spec.sample.video_id,
        "clip_start": spec.sample.clip_start,
        "clip_end": spec.sample.clip_end,
        "condition": spec.condition.key,
        "repeat": spec.repeat,
        "references": [asdict(clip) for clip in spec.condition.references],
        "api_call": {
            "duration_seconds": call["duration_seconds"],
            "elapsed_seconds": call["elapsed_seconds"],
            "estimated_cost_usd": call["estimated_cost_usd"],
        },
        "metrics": metrics,
        "boundary_metrics": boundaries,
        "target_segments": target_segments(spec.sample, segments),
        "segments": segments,
    }
    evaluation_path = (
        output_dir
        / "evaluations"
        / spec.sample.key
        / spec.condition.key
        / f"run-{spec.repeat:02d}.json"
    )
    write_json(evaluation_path, evaluation)
    return evaluation


def average(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(statistics.mean(present), 4) if present else None


def aggregate(evaluations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        buckets.setdefault((evaluation["sample"], evaluation["condition"]), []).append(evaluation)
    rows = []
    for (sample, condition), runs in sorted(buckets.items()):
        rows.append(
            {
                "sample": sample,
                "condition": condition,
                "references": [item["speaker"] for item in runs[0]["references"]],
                "runs": len(runs),
                "known_named_accuracy": average(
                    run["metrics"]["known_named_accuracy"] for run in runs
                ),
                "unreferenced_false_known_rate": average(
                    run["metrics"]["unreferenced_false_known_rate"] for run in runs
                ),
                "absent_reference_false_positive_seconds": average(
                    run["metrics"]["absent_reference_false_positive_seconds"] for run in runs
                ),
                "oracle_identity_accuracy": average(
                    run["metrics"]["oracle_identity_accuracy"] for run in runs
                ),
                "boundary_mean_f1": average(
                    run["boundary_metrics"]["mean_f1"] for run in runs
                ),
                "elapsed_seconds": average(run["api_call"]["elapsed_seconds"] for run in runs),
                "cost_usd": round(sum(run["api_call"]["estimated_cost_usd"] for run in runs), 6),
            }
        )
    return rows


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_report(rows: Sequence[dict[str, Any]], evaluations: Sequence[dict[str, Any]]) -> str:
    total_cost = sum(item["api_call"]["estimated_cost_usd"] for item in evaluations)
    lines = [
        "# Known-speaker reference matrix",
        "",
        f"Two episodes, {len(rows)} episode-condition cells, {len(evaluations)} API calls. Estimated incremental cost: ${total_cost:.3f}.",
        "",
        "| Episode | Condition | References | Runs | Named accuracy | Unknown false-match | Absent-ref FP seconds | Oracle identity | Boundary F1 | Avg latency |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {condition} | {references} | {runs} | {named} | {false_match} | {absent:.2f} | {oracle} | {boundary} | {latency:.1f}s |".format(
                sample=row["sample"],
                condition=row["condition"],
                references=", ".join(row["references"]) or "none",
                runs=row["runs"],
                named=percent(row["known_named_accuracy"]),
                false_match=percent(row["unreferenced_false_known_rate"]),
                absent=row["absent_reference_false_positive_seconds"] or 0.0,
                oracle=percent(row["oracle_identity_accuracy"]),
                boundary=percent(row["boundary_mean_f1"]),
                latency=row["elapsed_seconds"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "Named accuracy scores only speakers supplied as references. Unknown false-match measures how often an unreferenced real speaker was assigned one of the supplied names. Oracle identity maps anonymous labels to their best matching real speaker, while leaving supplied names fixed. Boundary F1 penalizes missed and invented speaker changes in the audited regions.",
            "",
            "Raw API responses and per-run target-window segments are stored beside this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate known-speaker reference roster strategies.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample", action="append", choices=[sample.key for sample in SAMPLES])
    parser.add_argument("--condition", action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    selected_samples = [
        sample for sample in SAMPLES if not args.sample or sample.key in args.sample
    ]
    specs = []
    for sample in selected_samples:
        conditions = conditions_for(sample)
        if args.condition:
            conditions = tuple(item for item in conditions if item.key in args.condition)
        for condition in conditions:
            for repeat in range(1, args.repeats + 1):
                specs.append(RunSpec(sample, condition, repeat))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample in selected_samples:
        prepare_sample(sample, args.output_dir)
        for condition in conditions_for(sample):
            for clip in condition.references:
                prepare_reference(clip, args.output_dir)

    print(f"Running {len(specs)} diarization calls with jobs={args.jobs}", flush=True)
    evaluations = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_spec, spec, args.output_dir, api_key, args.force): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                evaluation = future.result()
            except Exception as exc:
                failure = {
                    "sample": spec.sample.key,
                    "condition": spec.condition.key,
                    "repeat": spec.repeat,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                print(
                    f"[failed] {spec.sample.key} {spec.condition.key} "
                    f"run {spec.repeat}: {failure['error']}",
                    flush=True,
                )
                continue
            evaluations.append(evaluation)
            print(
                f"[{len(evaluations)}/{len(specs)}] {spec.sample.key} "
                f"{spec.condition.key} run {spec.repeat}: "
                f"{evaluation['api_call']['elapsed_seconds']:.1f}s",
                flush=True,
            )

    evaluations.sort(key=lambda item: (item["sample"], item["condition"], item["repeat"]))
    rows = aggregate(evaluations)
    summary = {"rows": rows, "evaluations": evaluations}
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "failures.json", failures)
    (args.output_dir / "report.md").write_text(
        render_report(rows, evaluations), encoding="utf-8"
    )
    print(f"Report: {args.output_dir / 'report.md'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
