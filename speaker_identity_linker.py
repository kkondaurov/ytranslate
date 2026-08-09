"""Conservative episode-wide acoustic speaker identity linking.

The linker is intentionally independent of transcript translation. Callers
provide anonymous boundary hints, a voice reference bank, and the existing
speaker-attributed segments. The existing attribution remains the fallback;
only sustained, high-confidence acoustic evidence can rewrite it.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


MAX_EMBEDDING_UNIT_SECONDS = 6.0
MIN_EMBEDDING_AUDIO_SECONDS = 1.5
SIL_TARGET_UNIT_SECONDS = 3.0
SIL_SELF_ENROLL_MIN_SIMILARITY = 0.84
SIL_SELF_ENROLL_MIN_MARGIN = 0.05
SIL_MAX_SELF_ENROLL_UNITS = 80
SIL_REFERENCE_WEIGHT = 1.0
SIL_EPISODE_PROTOTYPE_WEIGHT = 0.0
SIL_SWITCH_PENALTY_SAME = 0.045
SIL_SWITCH_PENALTY_UNCERTAIN = 0.018
SIL_SWITCH_PENALTY_CHANGE = -0.012
SIL_LOCAL_CHANGE_BONUS = 0.012
SIL_UNKNOWN_LABEL = "Unknown/External"
SIL_UNKNOWN_STATE_SCORE = 0.76
SIL_UNKNOWN_RUN_MAX_AVERAGE_SIMILARITY = 0.74
SIL_UNKNOWN_ACTIVATION_SECONDS = 12.0
SIL_UNKNOWN_ACTIVATION_UNITS = 2
SIL_UNIT_REPAIR_ACTIVATION_SECONDS = 20.0
SIL_UNIT_REPAIR_ACTIVATION_UNITS = 2
SIL_UNIT_REPAIR_MIN_SIMILARITY = 0.80
SIL_UNIT_REPAIR_MIN_MARGIN = 0.02
SIL_UNIT_REPAIR_MAX_GAP_SECONDS = 60.0
SIL_UNIT_REPAIR_DIRECT_MIN_MARGIN = 0.08
SIL_UNIT_REPAIR_SHORT_MAX_SECONDS = 3.0
SIL_UNIT_REPAIR_SHORT_MIN_SIMILARITY = 0.78
SIL_UNIT_REPAIR_SHORT_MIN_MARGIN = 0.02
AUTO_REFERENCE_CLIPS_PER_SPEAKER = 3
AUTO_REFERENCE_MIN_SECONDS = 2.5
AUTO_REFERENCE_MIN_SIMILARITY = 0.82
AUTO_REFERENCE_MIN_MARGIN = 0.04
AUTO_REFERENCE_MIN_SEPARATION_SECONDS = 15.0


def normalize_identity_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()


def canonical_speaker_name(value: Any) -> str:
    normalized = normalize_identity_text(value)
    aliases = (
        ("friedberg", ("friedberg", "freeberg", "freiberg")),
        ("sacks", ("sacks", "sachs", "zach")),
        ("chamath", ("chamath", "chumath", "jamath")),
        ("jason", ("jason", "j cal", "jcal")),
        ("brad", ("brad", "gerstner")),
        ("gavin", ("gavin", "baker")),
    )
    for canonical, candidates in aliases:
        if any(candidate in normalized for candidate in candidates):
            return canonical
    return normalized or "unknown"


def speaker_id_from_label(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", label or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    return f"speaker_{ascii_text or 'unknown'}"


def get_segment_local_key(segment: Dict[str, Any]) -> Tuple[int, str]:
    chunk_index = int(segment.get("chunk_index") or 0)
    local_speaker = str(
        segment.get("local_speaker") or segment.get("speaker") or "speaker"
    )
    return chunk_index, local_speaker


def normalized_embedding(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Cannot normalize a zero-length embedding")
    return vector / norm


def average_embeddings(embeddings: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not embeddings:
        return None
    averaged = np.mean(np.asarray(embeddings, dtype=np.float32), axis=0)
    return [float(value) for value in normalized_embedding(averaged)]


def robust_embedding_centroid(
    embeddings: Sequence[Sequence[float]],
) -> Optional[List[float]]:
    if len(embeddings) <= 2:
        return average_embeddings(embeddings)
    initial = average_embeddings(embeddings)
    if initial is None:
        return None
    initial_vector = np.asarray(initial, dtype=np.float32)
    ranked = sorted(
        embeddings,
        key=lambda embedding: float(
            np.asarray(embedding, dtype=np.float32) @ initial_vector
        ),
        reverse=True,
    )
    keep_count = max(2, math.ceil(len(ranked) * 0.7))
    return average_embeddings(ranked[:keep_count])


def split_span(start: float, end: float, max_seconds: float) -> List[Tuple[float, float]]:
    if end <= start:
        return [(start, start)]
    spans: List[Tuple[float, float]] = []
    cursor = start
    while cursor < end:
        next_end = min(end, cursor + max_seconds)
        spans.append((cursor, next_end))
        cursor = next_end
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < MIN_EMBEDDING_AUDIO_SECONDS:
        previous_start, _ = spans[-2]
        final_end = spans[-1][1]
        split_at = final_end - MIN_EMBEDDING_AUDIO_SECONDS
        spans[-2:] = [(previous_start, split_at), (split_at, final_end)]
    return spans


def build_sil_units(
    segments: Sequence[Dict[str, Any]],
    boundaries: Sequence[str],
) -> List[Dict[str, Any]]:
    if len(segments) != len(boundaries):
        raise ValueError("Segment and boundary counts differ")

    pieces: List[Dict[str, Any]] = []
    previous_local_key: Optional[Tuple[int, str]] = None
    for segment_id, (segment, boundary) in enumerate(zip(segments, boundaries)):
        start = float(segment.get("start") or 0)
        end = max(start, float(segment.get("end") or start))
        local_key = get_segment_local_key(segment)
        local_change = previous_local_key is not None and local_key != previous_local_key
        chunk_change = (
            previous_local_key is not None and local_key[0] != previous_local_key[0]
        )
        for split_index, (split_start, split_end) in enumerate(
            split_span(start, end, MAX_EMBEDDING_UNIT_SECONDS)
        ):
            pieces.append(
                {
                    "start": split_start,
                    "end": split_end,
                    "duration": max(0.0, split_end - split_start),
                    "segment_ids": [segment_id],
                    "chunk_index": local_key[0],
                    "local_speaker": local_key[1],
                    "boundary_before": boundary if split_index == 0 else "same",
                    "local_change_before": bool(local_change and split_index == 0),
                    "chunk_change_before": bool(chunk_change and split_index == 0),
                }
            )
        previous_local_key = local_key

    units: List[Dict[str, Any]] = []
    for piece in pieces:
        previous = units[-1] if units else None
        gap = float(piece["start"]) - float(previous["end"]) if previous else math.inf
        same_local_track = bool(
            previous
            and int(previous["chunk_index"]) == int(piece["chunk_index"])
            and str(previous["local_speaker"]) == str(piece["local_speaker"])
        )
        combined_duration = (
            float(piece["end"]) - float(previous["start"]) if previous else math.inf
        )
        should_merge = bool(
            previous
            and same_local_track
            and piece["boundary_before"] != "change"
            and gap <= 1.2
            and combined_duration <= MAX_EMBEDDING_UNIT_SECONDS
            and (
                float(previous["duration"]) < SIL_TARGET_UNIT_SECONDS
                or float(piece["duration"]) < MIN_EMBEDDING_AUDIO_SECONDS
            )
        )
        if should_merge:
            previous["end"] = max(float(previous["end"]), float(piece["end"]))
            previous["duration"] = float(previous["end"]) - float(previous["start"])
            previous["segment_ids"].extend(piece["segment_ids"])
            if piece["boundary_before"] == "uncertain":
                previous["contains_uncertain_boundary"] = True
            continue
        copied = dict(piece)
        copied["unit_id"] = len(units)
        units.append(copied)
    return units


def audio_clip(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    minimum_seconds: float = MIN_EMBEDDING_AUDIO_SECONDS,
) -> np.ndarray:
    start_sample = max(0, int(start * sample_rate))
    end_sample = min(len(audio), int(end * sample_rate))
    clip = audio[start_sample:end_sample]
    minimum_samples = int(minimum_seconds * sample_rate)
    if len(clip) < minimum_samples:
        clip = np.pad(clip, (0, minimum_samples - len(clip)))
    return np.asarray(clip, dtype=np.float32)


def embed_units(
    audio: np.ndarray,
    sample_rate: int,
    units: Sequence[Dict[str, Any]],
    encoder: Any,
) -> np.ndarray:
    embeddings = [
        normalized_embedding(
            encoder.embed_utterance(
                audio_clip(audio, sample_rate, float(unit["start"]), float(unit["end"]))
            )
        )
        for unit in units
    ]
    return np.vstack(embeddings)


def speaker_score_matrix(
    embeddings: np.ndarray,
    units: Sequence[Dict[str, Any]],
    references: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    names = list(references)
    reference_matrix = np.vstack([references[name] for name in names])
    reference_scores = embeddings @ reference_matrix.T
    winners = np.argmax(reference_scores, axis=1)
    sorted_scores = np.sort(reference_scores, axis=1)
    margins = (
        sorted_scores[:, -1] - sorted_scores[:, -2]
        if len(names) > 1
        else sorted_scores[:, -1]
    )

    prototypes: Dict[str, np.ndarray] = {}
    enrollment: List[Dict[str, Any]] = []
    for speaker_index, name in enumerate(names):
        candidates = [
            index
            for index, unit in enumerate(units)
            if int(winners[index]) == speaker_index
            and float(reference_scores[index, speaker_index])
            >= SIL_SELF_ENROLL_MIN_SIMILARITY
            and float(margins[index]) >= SIL_SELF_ENROLL_MIN_MARGIN
            and float(unit.get("duration") or 0) >= MIN_EMBEDDING_AUDIO_SECONDS
        ]
        candidates.sort(
            key=lambda index: (
                float(margins[index]),
                float(reference_scores[index, speaker_index]),
                float(units[index].get("duration") or 0),
            ),
            reverse=True,
        )
        candidates = candidates[:SIL_MAX_SELF_ENROLL_UNITS]
        centroid = robust_embedding_centroid(
            [[float(value) for value in embeddings[index]] for index in candidates]
        )
        prototype = normalized_embedding(centroid) if centroid is not None else references[name]
        prototypes[name] = prototype
        enrollment.append(
            {
                "speaker": name,
                "unit_count": len(candidates),
                "unit_ids": [int(units[index]["unit_id"]) for index in candidates],
            }
        )

    prototype_matrix = np.vstack([prototypes[name] for name in names])
    prototype_scores = embeddings @ prototype_matrix.T
    blended = (
        SIL_REFERENCE_WEIGHT * reference_scores
        + SIL_EPISODE_PROTOTYPE_WEIGHT * prototype_scores
    )
    return blended, prototypes, {
        "speaker_names": names,
        "reference_weight": SIL_REFERENCE_WEIGHT,
        "episode_prototype_weight": SIL_EPISODE_PROTOTYPE_WEIGHT,
        "enrollment": enrollment,
    }


def add_unknown_speaker_state(
    scores: np.ndarray,
    speaker_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    unknown_scores = np.full(
        (len(scores), 1), SIL_UNKNOWN_STATE_SCORE, dtype=np.float32
    )
    return np.column_stack([scores, unknown_scores]), [
        *speaker_names,
        SIL_UNKNOWN_LABEL,
    ]


def decode_sil_identities(
    scores: np.ndarray,
    units: Sequence[Dict[str, Any]],
    speaker_names: Sequence[str],
    known_speaker_count: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(scores) != len(units):
        raise ValueError("Score and unit counts differ")
    if not len(units):
        return np.asarray([], dtype=int), {"assignments": []}

    state_count = len(speaker_names)
    dynamic = np.full((len(units), state_count), -np.inf, dtype=np.float64)
    backpointers = np.zeros((len(units), state_count), dtype=int)
    first_weight = max(0.5, min(1.5, float(units[0]["duration"]) / 3.0))
    dynamic[0] = scores[0] * first_weight

    for index in range(1, len(units)):
        unit = units[index]
        boundary = str(unit.get("boundary_before") or "uncertain")
        if bool(unit.get("chunk_change_before")):
            switch_penalty = 0.0
        elif boundary == "change":
            switch_penalty = SIL_SWITCH_PENALTY_CHANGE
        elif boundary == "same":
            switch_penalty = SIL_SWITCH_PENALTY_SAME
        else:
            switch_penalty = SIL_SWITCH_PENALTY_UNCERTAIN
        if bool(unit.get("local_change_before")) and not bool(
            unit.get("chunk_change_before")
        ):
            switch_penalty -= SIL_LOCAL_CHANGE_BONUS

        duration_weight = max(0.5, min(1.5, float(unit["duration"]) / 3.0))
        for state in range(state_count):
            transition_scores = dynamic[index - 1].copy()
            for previous_state in range(state_count):
                if previous_state != state:
                    transition_scores[previous_state] -= switch_penalty
            winner = int(np.argmax(transition_scores))
            backpointers[index, state] = winner
            dynamic[index, state] = (
                float(transition_scores[winner])
                + float(scores[index, state]) * duration_weight
            )

    labels = np.zeros(len(units), dtype=int)
    labels[-1] = int(np.argmax(dynamic[-1]))
    for index in range(len(units) - 1, 0, -1):
        labels[index - 1] = backpointers[index, labels[index]]

    assignments: List[Dict[str, Any]] = []
    for index, (unit, label) in enumerate(zip(units, labels)):
        order = np.argsort(scores[index])[::-1]
        alternatives = [int(value) for value in order if int(value) != int(label)]
        runner_up = alternatives[0] if alternatives else int(label)
        assignment = {
            "unit_id": int(unit["unit_id"]),
            "start": round(float(unit["start"]), 3),
            "end": round(float(unit["end"]), 3),
            "duration": round(float(unit["duration"]), 3),
            "chunk_index": int(unit["chunk_index"]),
            "local_speaker": str(unit["local_speaker"]),
            "boundary_before": str(unit["boundary_before"]),
            "speaker": str(speaker_names[int(label)]),
            "similarity": round(float(scores[index, int(label)]), 4),
            "margin": round(
                float(scores[index, int(label)] - scores[index, runner_up]), 4
            ),
            "acoustic_winner": str(speaker_names[int(order[0])]),
            "segment_ids": [int(value) for value in unit["segment_ids"]],
        }
        if known_speaker_count:
            known_order = np.argsort(scores[index, :known_speaker_count])[::-1]
            best_known = int(known_order[0])
            second_known = int(known_order[1]) if len(known_order) > 1 else best_known
            assignment.update(
                {
                    "best_known_speaker": str(speaker_names[best_known]),
                    "best_known_similarity": round(float(scores[index, best_known]), 4),
                    "known_margin": round(
                        float(scores[index, best_known] - scores[index, second_known]),
                        4,
                    ),
                }
            )
        assignments.append(assignment)
    return labels, {
        "assignments": assignments,
        "speaker_unit_counts": dict(Counter(item["speaker"] for item in assignments)),
    }


def overlap_seconds(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _baseline_identity_for_assignment(
    assignment: Dict[str, Any],
    baseline_segments: Sequence[Dict[str, Any]],
) -> str:
    start = float(assignment["start"])
    end = float(assignment["end"])
    votes: Dict[str, float] = defaultdict(float)
    for segment_id in assignment.get("segment_ids", []):
        segment = baseline_segments[int(segment_id)]
        identity = canonical_speaker_name(
            segment.get("speaker_label") or segment.get("speaker_id")
        )
        votes[identity] += max(
            0.001,
            overlap_seconds(
                start,
                end,
                float(segment.get("start") or 0),
                float(segment.get("end") or 0),
            ),
        )
    return max(votes, key=votes.get) if votes else "unknown"


def _assignment_runs(
    assignments: Sequence[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    runs: List[List[Dict[str, Any]]] = []
    for assignment in assignments:
        if (
            runs
            and canonical_speaker_name(runs[-1][-1].get("speaker"))
            == canonical_speaker_name(assignment.get("speaker"))
            and float(assignment["start"]) - float(runs[-1][-1]["end"]) <= 1.5
        ):
            runs[-1].append(assignment)
        else:
            runs.append([assignment])
    return runs


def apply_episode_scale_unit_repairs(
    baseline_segments: Sequence[Dict[str, Any]],
    assignments: Sequence[Dict[str, Any]],
    boundaries: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(baseline_segments) != len(boundaries):
        raise ValueError("Baseline segment and boundary counts differ")

    unknown_identity = canonical_speaker_name(SIL_UNKNOWN_LABEL)
    enriched: List[Dict[str, Any]] = []
    strong_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        copied = dict(assignment)
        baseline_identity = _baseline_identity_for_assignment(copied, baseline_segments)
        candidate_identity = canonical_speaker_name(copied.get("speaker"))
        copied["baseline_identity"] = baseline_identity
        copied["candidate_identity"] = candidate_identity
        if (
            candidate_identity not in {baseline_identity, unknown_identity}
            and float(copied.get("similarity") or 0) >= SIL_UNIT_REPAIR_MIN_SIMILARITY
            and float(copied.get("margin") or 0) >= SIL_UNIT_REPAIR_MIN_MARGIN
        ):
            strong_by_pair[(baseline_identity, candidate_identity)].append(copied)
        enriched.append(copied)

    activated_pairs: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for pair, pair_assignments in strong_by_pair.items():
        seconds = sum(float(item.get("duration") or 0) for item in pair_assignments)
        if (
            len(pair_assignments) >= SIL_UNIT_REPAIR_ACTIVATION_UNITS
            and seconds >= SIL_UNIT_REPAIR_ACTIVATION_SECONDS
        ):
            activated_pairs[pair] = pair_assignments

    repair_candidates: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for assignment in enriched:
        baseline_identity = str(assignment["baseline_identity"])
        candidate_identity = str(assignment["candidate_identity"])
        if candidate_identity in {baseline_identity, unknown_identity}:
            continue
        scope = (
            str(assignment.get("local_speaker") or "__unscoped__"),
            baseline_identity,
            candidate_identity,
        )
        repair_candidates[scope].append(assignment)

    active_known_unit_pairs: set[Tuple[int, str, str]] = set()
    isolated_short_unit_ids: set[int] = set()
    activated_scopes: List[Dict[str, Any]] = []
    for scope, scoped_assignments in repair_candidates.items():
        local_speaker, baseline_identity, candidate_identity = scope
        scoped_assignments.sort(
            key=lambda item: (float(item["start"]), int(item["unit_id"]))
        )
        scoped_runs: List[List[Dict[str, Any]]] = []
        for assignment in scoped_assignments:
            if (
                scoped_runs
                and float(assignment["start"]) - float(scoped_runs[-1][-1]["end"])
                <= SIL_UNIT_REPAIR_MAX_GAP_SECONDS
            ):
                scoped_runs[-1].append(assignment)
            else:
                scoped_runs.append([assignment])
        for run in scoped_runs:
            run_seconds = sum(float(item.get("duration") or 0) for item in run)
            if run_seconds <= SIL_UNIT_REPAIR_SHORT_MAX_SECONDS:
                isolated_short_unit_ids.update(int(item["unit_id"]) for item in run)
            strong = [
                item
                for item in run
                if float(item.get("similarity") or 0) >= SIL_UNIT_REPAIR_MIN_SIMILARITY
                and float(item.get("margin") or 0) >= SIL_UNIT_REPAIR_MIN_MARGIN
            ]
            strong_seconds = sum(float(item.get("duration") or 0) for item in strong)
            if (
                len(strong) < SIL_UNIT_REPAIR_ACTIVATION_UNITS
                or strong_seconds < SIL_UNIT_REPAIR_ACTIVATION_SECONDS
            ):
                continue
            for item in run:
                active_known_unit_pairs.add(
                    (int(item["unit_id"]), baseline_identity, candidate_identity)
                )
            activated_scopes.append(
                {
                    "chunk_indexes": sorted(
                        {int(item.get("chunk_index", -1)) for item in run}
                    ),
                    "local_speaker": local_speaker,
                    "pair": f"{baseline_identity}->{candidate_identity}",
                    "start": round(float(run[0]["start"]), 3),
                    "end": round(float(run[-1]["end"]), 3),
                    "unit_count": len(run),
                    "seconds": round(run_seconds, 3),
                    "strong_unit_count": len(strong),
                    "strong_seconds": round(strong_seconds, 3),
                }
            )

    active_unknown_unit_ids: set[int] = set()
    unknown_runs: List[Dict[str, Any]] = []
    for run in _assignment_runs(enriched):
        if canonical_speaker_name(run[0].get("speaker")) != unknown_identity:
            continue
        seconds = sum(float(item.get("duration") or 0) for item in run)
        total_weight = sum(
            max(0.001, float(item.get("duration") or 0)) for item in run
        )
        best_known_average = sum(
            float(item.get("best_known_similarity") or 0)
            * max(0.001, float(item.get("duration") or 0))
            for item in run
        ) / total_weight
        best_known_peak = max(
            float(item.get("best_known_similarity") or 0) for item in run
        )
        accepted = bool(
            len(run) >= SIL_UNKNOWN_ACTIVATION_UNITS
            and seconds >= SIL_UNKNOWN_ACTIVATION_SECONDS
            and best_known_average < SIL_UNKNOWN_RUN_MAX_AVERAGE_SIMILARITY
        )
        if accepted:
            active_unknown_unit_ids.update(int(item["unit_id"]) for item in run)
        unknown_runs.append(
            {
                "start": round(float(run[0]["start"]), 3),
                "end": round(float(run[-1]["end"]), 3),
                "seconds": round(seconds, 3),
                "unit_count": len(run),
                "accepted": accepted,
                "best_known_average_similarity": round(best_known_average, 4),
                "best_known_similarity": round(best_known_peak, 4),
            }
        )

    votes: List[Dict[str, float]] = [defaultdict(float) for _ in baseline_segments]
    display_labels: Dict[str, str] = {}
    for segment in baseline_segments:
        label = str(
            segment.get("speaker_label") or segment.get("speaker_id") or "Unknown"
        )
        display_labels[canonical_speaker_name(label)] = label
    display_labels[unknown_identity] = SIL_UNKNOWN_LABEL
    direct_repair_unit_ids: set[int] = set()
    for assignment in enriched:
        candidate_identity = str(assignment["candidate_identity"])
        baseline_identity = str(assignment["baseline_identity"])
        unit_id = int(assignment["unit_id"])
        direct_high_confidence = bool(
            float(assignment.get("similarity") or 0) >= SIL_UNIT_REPAIR_MIN_SIMILARITY
            and float(assignment.get("margin") or 0) >= SIL_UNIT_REPAIR_DIRECT_MIN_MARGIN
        )
        direct_short = bool(
            unit_id in isolated_short_unit_ids
            and float(assignment.get("similarity") or 0)
            >= SIL_UNIT_REPAIR_SHORT_MIN_SIMILARITY
            and float(assignment.get("margin") or 0) >= SIL_UNIT_REPAIR_SHORT_MIN_MARGIN
        )
        direct_repair = bool(
            (baseline_identity, candidate_identity) in activated_pairs
            and (direct_high_confidence or direct_short)
        )
        if direct_repair:
            direct_repair_unit_ids.add(unit_id)
        for segment_id in assignment.get("segment_ids", []):
            segment = baseline_segments[int(segment_id)]
            segment_baseline_identity = canonical_speaker_name(
                segment.get("speaker_label") or segment.get("speaker_id")
            )
            if unit_id in active_unknown_unit_ids:
                effective_identity = unknown_identity
            elif (
                (
                    direct_repair
                    and (segment_baseline_identity, candidate_identity) in activated_pairs
                )
                or (
                    unit_id,
                    segment_baseline_identity,
                    candidate_identity,
                )
                in active_known_unit_pairs
            ):
                effective_identity = candidate_identity
                display_labels.setdefault(candidate_identity, str(assignment["speaker"]))
            else:
                effective_identity = segment_baseline_identity
            overlap = overlap_seconds(
                float(assignment["start"]),
                float(assignment["end"]),
                float(segment.get("start") or 0),
                float(segment.get("end") or 0),
            )
            votes[int(segment_id)][effective_identity] += max(overlap, 0.001)

    resolved: List[Dict[str, Any]] = []
    repaired_segment_ids: set[int] = set()
    for segment_id, segment in enumerate(baseline_segments):
        baseline_identity = canonical_speaker_name(
            segment.get("speaker_label") or segment.get("speaker_id")
        )
        effective_identity = (
            max(votes[segment_id].items(), key=lambda item: item[1])[0]
            if votes[segment_id]
            else baseline_identity
        )
        copied = dict(segment)
        copied["boundary_before"] = boundaries[segment_id]
        if effective_identity != baseline_identity:
            label = display_labels[effective_identity]
            copied.update(
                {
                    "speaker_id": speaker_id_from_label(label),
                    "speaker_label": label,
                    "speaker_id_source": "sil-unit-repair",
                }
            )
            repaired_segment_ids.add(segment_id)
        resolved.append(copied)

    return resolved, {
        "thresholds": {
            "activation_seconds": SIL_UNIT_REPAIR_ACTIVATION_SECONDS,
            "activation_units": SIL_UNIT_REPAIR_ACTIVATION_UNITS,
            "minimum_similarity": SIL_UNIT_REPAIR_MIN_SIMILARITY,
            "minimum_margin": SIL_UNIT_REPAIR_MIN_MARGIN,
            "maximum_gap_seconds": SIL_UNIT_REPAIR_MAX_GAP_SECONDS,
            "direct_minimum_margin": SIL_UNIT_REPAIR_DIRECT_MIN_MARGIN,
            "short_maximum_seconds": SIL_UNIT_REPAIR_SHORT_MAX_SECONDS,
            "short_minimum_similarity": SIL_UNIT_REPAIR_SHORT_MIN_SIMILARITY,
            "short_minimum_margin": SIL_UNIT_REPAIR_SHORT_MIN_MARGIN,
            "unknown_state_score": SIL_UNKNOWN_STATE_SCORE,
            "unknown_run_max_average_similarity": SIL_UNKNOWN_RUN_MAX_AVERAGE_SIMILARITY,
            "unknown_activation_seconds": SIL_UNKNOWN_ACTIVATION_SECONDS,
            "unknown_activation_units": SIL_UNKNOWN_ACTIVATION_UNITS,
        },
        "activated_pairs": {
            f"{left}->{right}": {
                "unit_count": len(items),
                "seconds": round(
                    sum(float(item.get("duration") or 0) for item in items), 3
                ),
            }
            for (left, right), items in activated_pairs.items()
        },
        "activated_scopes": activated_scopes,
        "direct_repair_unit_count": len(direct_repair_unit_ids),
        "unknown_runs": unknown_runs,
        "repaired_segment_count": len(repaired_segment_ids),
    }


def build_episode_reference_centroids(
    baseline_segments: Sequence[Dict[str, Any]],
    audio: np.ndarray,
    sample_rate: int,
    encoder: Any,
    frozen_reference_centroids: Dict[str, Sequence[float]],
    active_speaker_labels: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Resolve the active roster from the baseline without trusting it blindly.

    Reviewed recurring voices use frozen cross-episode centroids. Speakers not
    present in that bank can enroll from separated, high-confidence baseline
    segments. A frozen voice is only activated when that identity is actually
    present in the episode baseline, which prevents absent hosts from competing
    with guests.
    """
    active_labels: Dict[str, str] = {}
    for label_value in active_speaker_labels or ():
        label = str(label_value or "").strip()
        identity = canonical_speaker_name(label)
        if (
            label
            and not identity.startswith("unknown")
            and identity != canonical_speaker_name(SIL_UNKNOWN_LABEL)
        ):
            active_labels.setdefault(identity, label)
    for segment in baseline_segments:
        label = str(
            segment.get("speaker_label") or segment.get("speaker_id") or ""
        ).strip()
        identity = canonical_speaker_name(label)
        if (
            label
            and not identity.startswith("unknown")
            and identity != canonical_speaker_name(SIL_UNKNOWN_LABEL)
        ):
            active_labels.setdefault(identity, label)

    frozen_by_identity = {
        canonical_speaker_name(name): (name, normalized_embedding(values))
        for name, values in frozen_reference_centroids.items()
    }
    references: Dict[str, np.ndarray] = {}
    debug: Dict[str, Any] = {"frozen": [], "automatic": [], "skipped": []}
    for identity, label in active_labels.items():
        frozen = frozen_by_identity.get(identity)
        if frozen is not None:
            frozen_name, centroid = frozen
            references[frozen_name] = centroid
            debug["frozen"].append({"speaker": frozen_name})
            continue

        candidates = [
            segment
            for segment in baseline_segments
            if canonical_speaker_name(
                segment.get("speaker_label") or segment.get("speaker_id")
            )
            == identity
            and float(segment.get("end") or 0) - float(segment.get("start") or 0)
            >= AUTO_REFERENCE_MIN_SECONDS
            and float(segment.get("voice_similarity") or 0)
            >= AUTO_REFERENCE_MIN_SIMILARITY
            and float(segment.get("voice_similarity_margin") or 0)
            >= AUTO_REFERENCE_MIN_MARGIN
        ]
        if not candidates:
            candidates = [
                segment
                for segment in baseline_segments
                if canonical_speaker_name(
                    segment.get("speaker_label") or segment.get("speaker_id")
                )
                == identity
                and float(segment.get("end") or 0)
                - float(segment.get("start") or 0)
                >= AUTO_REFERENCE_MIN_SECONDS
            ]
        candidates.sort(
            key=lambda segment: (
                float(segment.get("voice_similarity_margin") or 0),
                float(segment.get("voice_similarity") or 0),
                float(segment.get("end") or 0) - float(segment.get("start") or 0),
            ),
            reverse=True,
        )
        selected: List[Dict[str, Any]] = []
        for candidate in candidates:
            start = float(candidate.get("start") or 0)
            if any(
                abs(start - float(previous.get("start") or 0))
                < AUTO_REFERENCE_MIN_SEPARATION_SECONDS
                for previous in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= AUTO_REFERENCE_CLIPS_PER_SPEAKER:
                break
        if not selected:
            debug["skipped"].append({"speaker": label, "reason": "no-reference-clips"})
            continue
        embeddings = [
            normalized_embedding(
                encoder.embed_utterance(
                    audio_clip(
                        audio,
                        sample_rate,
                        float(segment.get("start") or 0),
                        float(segment.get("end") or 0),
                        minimum_seconds=0.1,
                    )
                )
            )
            for segment in selected
        ]
        centroid = robust_embedding_centroid(embeddings)
        if centroid is None:
            debug["skipped"].append({"speaker": label, "reason": "no-centroid"})
            continue
        references[label] = normalized_embedding(centroid)
        debug["automatic"].append(
            {
                "speaker": label,
                "clips": [
                    {
                        "start": round(float(segment.get("start") or 0), 3),
                        "end": round(float(segment.get("end") or 0), 3),
                    }
                    for segment in selected
                ],
            }
        )
    return references, debug


def link_speaker_identities(
    baseline_segments: Sequence[Dict[str, Any]],
    boundaries: Sequence[str],
    audio: np.ndarray,
    sample_rate: int,
    encoder: Any,
    reference_centroids: Dict[str, Sequence[float]],
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not baseline_segments:
        return [], {"status": "skipped", "reason": "no-segments"}
    if len(baseline_segments) != len(boundaries):
        raise ValueError("Baseline segment and boundary counts differ")
    if len(reference_centroids) < 2:
        raise ValueError("At least two speaker reference centroids are required")

    units = build_sil_units(baseline_segments, boundaries)
    if log:
        log(f"Speaker identity linker embedding {len(units)} acoustic units...")
    embeddings = embed_units(audio, sample_rate, units, encoder)
    references = {
        name: normalized_embedding(values)
        for name, values in reference_centroids.items()
    }
    known_scores, _, prototype_debug = speaker_score_matrix(
        embeddings, units, references
    )
    scores, speaker_names = add_unknown_speaker_state(
        known_scores, list(references)
    )
    _, linker_debug = decode_sil_identities(
        scores,
        units,
        speaker_names,
        known_speaker_count=len(references),
    )
    segments, repair_debug = apply_episode_scale_unit_repairs(
        baseline_segments,
        linker_debug["assignments"],
        boundaries,
    )
    return segments, {
        "status": "ok",
        "embedding_unit_count": len(units),
        "speaker_names": list(references),
        "prototype_training": prototype_debug,
        "linker": linker_debug,
        "episode_scale_repairs": repair_debug,
    }
