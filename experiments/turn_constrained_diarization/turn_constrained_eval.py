#!/usr/bin/env python3
"""Evaluate episode-wide acoustic speaker identity linking on cached episodes.

The experiment keeps production untouched. Luna supplies weak anonymous turn
evidence, while SIL links short acoustic units across the full episode and uses
clean in-episode references only to attach names to stable voice identities.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import numpy as np
import soundfile as sf
from openai import OpenAI
from resemblyzer import VoiceEncoder
from scipy.optimize import linear_sum_assignment
from spectralcluster import configs
from spectralcluster.constraint import ConstraintMatrix


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ytranslate  # noqa: E402


MODEL = "gpt-5.6-luna"
EVALUATION_REQUEST_TIMEOUT_SECONDS = 180
DEFAULT_EFFORTS = ("low", "medium")
VALID_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
CACHE_ROOT = Path.home() / "Library" / "Caches" / "ytranslate"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
BOUNDARY_BATCH_SEGMENTS = 300
BOUNDARY_CONTEXT_SEGMENTS = 12
MAX_EMBEDDING_UNIT_SECONDS = 6.0
MIN_EMBEDDING_AUDIO_SECONDS = 1.5
BOUNDARY_SCORE = {"same": 0.0, "uncertain": 1.0, "change": 2.0}
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
ATTRIBUTION_WINDOW_PASS_ACCURACY = 0.90
SIL_MISSING_IDENTITY_ACTIVATION_SECONDS = 20.0
SIL_MISSING_IDENTITY_ACTIVATION_TURNS = 2
SIL_MISSING_IDENTITY_MIN_SIMILARITY = 0.82
SIL_MISSING_IDENTITY_MIN_MARGIN = 0.04
SIL_MISSING_IDENTITY_MIN_TURN_SECONDS = 2.0
SIL_CONFUSION_PAIR_ACTIVATION_SECONDS = 60.0
SIL_CONFUSION_PAIR_ACTIVATION_TURNS = 2
SIL_TARGET_ANCHOR_SECONDS = 60.0
SIL_TARGET_ANCHOR_TURNS = 2
AUTO_REFERENCE_CLIPS_PER_SPEAKER = 3
AUTO_REFERENCE_MIN_SECONDS = 2.5
AUTO_REFERENCE_MIN_SIMILARITY = 0.82
AUTO_REFERENCE_MIN_MARGIN = 0.04
AUTO_REFERENCE_MIN_SEPARATION_SECONDS = 15.0
REPAIR_MIN_SIMILARITY = 0.82
REPAIR_MIN_MARGIN = 0.05
REPAIR_MIN_TURN_SECONDS = 10.0
INPUT_PRICE_PER_MILLION = 0.20
CACHED_INPUT_PRICE_PER_MILLION = 0.02
OUTPUT_PRICE_PER_MILLION = 1.20


@dataclass(frozen=True)
class SpeakerReference:
    name: str
    start: float
    end: float
    source_video_id: Optional[str] = None


@dataclass(frozen=True)
class AttributionWindow:
    name: str
    start: float
    end: float
    expected: str


@dataclass(frozen=True)
class Episode:
    key: str
    video_id: str
    title: str
    context: str
    asr_filename: str
    references: Tuple[SpeakerReference, ...]
    channel_title: str = "All-In Podcast"
    tags: Tuple[str, ...] = ()
    cached_baseline_filename: Optional[str] = None
    auto_reference_seed: bool = False
    supplement_auto_references: bool = False
    audit_path: Optional[Path] = None
    windows: Tuple[AttributionWindow, ...] = ()

    @property
    def cache_dir(self) -> Path:
        return CACHE_ROOT / self.video_id

    @property
    def asr_path(self) -> Path:
        return self.cache_dir / self.asr_filename

    @property
    def audio_path(self) -> Path:
        candidates = [
            path
            for path in sorted((self.cache_dir / "audio").glob("source.*"))
            if not path.name.endswith(".part")
        ]
        if not candidates:
            raise FileNotFoundError(f"No cached source audio for {self.video_id}")
        return candidates[0]

    @property
    def wav_path(self) -> Path:
        return self.cache_dir / "voice-speaker-reconciliation" / "source-16k.wav"


TRUSTED_HOST_REFERENCES = {
    "jason": (
        SpeakerReference("Jason Calacanis", 91.668, 99.068, "HGbA6ze0_3M"),
        SpeakerReference("Jason Calacanis", 2205.0, 2213.0, "HGbA6ze0_3M"),
        SpeakerReference("Jason Calacanis", 4915.0, 4923.0, "HGbA6ze0_3M"),
    ),
    "chamath": (
        SpeakerReference("Chamath Palihapitiya", 193.252, 201.0, "HGbA6ze0_3M"),
        SpeakerReference("Chamath Palihapitiya", 824.0, 832.0, "HGbA6ze0_3M"),
        SpeakerReference("Chamath Palihapitiya", 3595.0, 3603.0, "HGbA6ze0_3M"),
    ),
    "friedberg": (
        SpeakerReference("David Friedberg", 517.172, 523.022, "HGbA6ze0_3M"),
        SpeakerReference("David Friedberg", 1148.0, 1156.0, "HGbA6ze0_3M"),
        SpeakerReference("David Friedberg", 3360.0, 3368.0, "HGbA6ze0_3M"),
    ),
    "sacks": (
        SpeakerReference("David Sacks", 183.796, 190.046, "wcV0SRPFK9s"),
        SpeakerReference("David Sacks", 2397.0, 2405.0, "muRIXCDw-k0"),
        SpeakerReference("David Sacks", 2623.0, 2631.0, "muRIXCDw-k0"),
    ),
}

TRUSTED_GUEST_REFERENCES = {
    "brad": (
        SpeakerReference("Brad Gerstner", 440.0, 448.0, "muRIXCDw-k0"),
        SpeakerReference("Brad Gerstner", 1140.0, 1148.0, "muRIXCDw-k0"),
        SpeakerReference("Brad Gerstner", 2030.0, 2038.0, "muRIXCDw-k0"),
    ),
    "gavin": (
        SpeakerReference("Gavin Baker", 370.0, 378.0, "HGbA6ze0_3M"),
        SpeakerReference("Gavin Baker", 1000.0, 1008.0, "HGbA6ze0_3M"),
        SpeakerReference("Gavin Baker", 2990.0, 2998.0, "HGbA6ze0_3M"),
    ),
}


def trusted_references(*identities: str) -> Tuple[SpeakerReference, ...]:
    references: List[SpeakerReference] = []
    for identity in identities:
        bank = TRUSTED_HOST_REFERENCES.get(identity) or TRUSTED_GUEST_REFERENCES.get(
            identity
        )
        if bank is None:
            raise KeyError(f"No trusted reference bank for {identity}")
        references.extend(bank)
    return tuple(references)


EPISODES = (
    Episode(
        key="brad-guest",
        video_id="muRIXCDw-k0",
        title=(
            "Google's AI Brain Drain, SpaceX's Big Quarter, Airtable's 90% Drop, "
            "US Data Fuels China's AI"
        ),
        context=(
            "All-In Podcast panel with Jason Calacanis, David Sacks, David Friedberg, "
            "and guest Brad Gerstner. Chamath Palihapitiya is absent."
        ),
        asr_filename="openai-asr-gpt-4o-transcribe-diarize-600s.json",
        references=trusted_references("jason", "friedberg", "brad", "sacks"),
        windows=(
            AttributionWindow("Audit: Friedberg opening reply", 15.15, 16.55, "Friedberg"),
            AttributionWindow("Audit: Brad opening interjection", 28.698, 29.048, "Brad"),
            AttributionWindow("Audit: Brad opening reply", 35.548, 37.698, "Brad"),
            AttributionWindow("Page 5: Friedberg conclusion", 405.008, 429.414, "Friedberg"),
            AttributionWindow("Page 5: Jason question", 429.414, 431.114, "Jason"),
            AttributionWindow("Page 5: Brad answer", 431.264, 536.222, "Brad"),
            AttributionWindow("Audit: Brad asks Jason", 751.818, 754.068, "Brad"),
            AttributionWindow("Audit: Friedberg model-mix answer", 971.318, 1093.17, "Friedberg"),
            AttributionWindow("Pages 11-12: Friedberg conclusion", 1080.342, 1093.170, "Friedberg"),
            AttributionWindow("Pages 11-12: Jason question", 1093.370, 1118.830, "Jason"),
            AttributionWindow("Pages 11-12: Brad answer", 1119.744, 1215.000, "Brad"),
            AttributionWindow("Audit: Brad SpaceX answer", 1311.692, 1360.0, "Brad"),
            AttributionWindow("Audit: Brad follows Friedberg", 1794.494, 1810.47, "Brad"),
            AttributionWindow("Audit: Brad valuation answer", 2004.456, 2112.316, "Brad"),
            AttributionWindow("Audit: Friedberg physical-sites answer", 2124.31, 2157.0, "Friedberg"),
            AttributionWindow("Audit: Brad short interjection", 2394.316, 2394.816, "Brad"),
            AttributionWindow("Audit: Sacks financing question", 2394.966, 2421.68, "Sacks"),
            AttributionWindow("Audit: Brad financing answer", 2421.68, 2516.274, "Brad"),
            AttributionWindow("Audit: Jason interruption", 2516.352, 2518.952, "Jason"),
            AttributionWindow("Audit: Brad resumes", 2518.952, 2611.816, "Brad"),
            AttributionWindow("Audit: Sacks response", 2613.016, 2620.716, "Sacks"),
        ),
    ),
    Episode(
        key="gavin-guest-audited",
        video_id="HGbA6ze0_3M",
        title=(
            "SpaceX's $2T Case, Nvidia's Shock Selloff, America Turns on AI, "
            "Trump Pulls AI Order, Bond Crisis?"
        ),
        context=(
            "All-In Podcast panel with Jason Calacanis, Chamath Palihapitiya, "
            "David Friedberg, and guest Gavin Baker. David Sacks is absent."
        ),
        asr_filename="openai-asr-gpt-4o-transcribe-diarize-1200s.json",
        references=trusted_references("jason", "chamath", "gavin", "friedberg"),
        audit_path=(
            REPO_ROOT
            / "experiments"
            / "speaker_audit"
            / "output"
            / "HGbA6ze0_3M"
            / "audit-rows.json"
        ),
        windows=(
            AttributionWindow(
                "External clip: Zuckerberg interview", 2321.786, 2358.658, SIL_UNKNOWN_LABEL
            ),
            AttributionWindow(
                "External clip: frontline AI quote", 2462.418, 2480.210, SIL_UNKNOWN_LABEL
            ),
        ),
    ),
    Episode(
        key="regular-four",
        video_id="wcV0SRPFK9s",
        title=(
            "The Fight Over Open Source AI, Anthropic's $1.5B Payout, "
            "NYC Socialists: Evictions = Violence?"
        ),
        context=(
            "All-In Podcast panel with Jason Calacanis, Chamath Palihapitiya, "
            "David Sacks, and David Friedberg."
        ),
        asr_filename="openai-asr-gpt-4o-transcribe-diarize-600s.json",
        references=trusted_references("jason", "sacks", "friedberg", "chamath"),
        windows=(
            AttributionWindow(
                "Audit: Sacks public-domain argument", 281.384, 313.608, "Sacks"
            ),
            AttributionWindow(
                "Audit: Sacks distillation argument", 873.878, 885.428, "Sacks"
            ),
            AttributionWindow(
                "Audit: Friedberg copyright answer", 984.598, 993.398, "Friedberg"
            ),
            AttributionWindow(
                "Audit: Sacks open-source argument", 1176.664, 1204.590, "Sacks"
            ),
        ),
    ),
    Episode(
        key="core-four-holdout",
        video_id="ViqYWhLimGg",
        title=(
            "Chip Stocks Crash, $20B Fund Margin Called, Frontier Labs: "
            "SLOW DOWN AI, Mamdani's Grocery Stores"
        ),
        context=(
            "All-In Podcast panel with Jason Calacanis, Chamath Palihapitiya, "
            "David Sacks, and David Friedberg. The first 45 minutes have manual "
            "review windows; the remainder is held out from tuning."
        ),
        asr_filename="openai-asr-gpt-4o-transcribe-diarize-600s.json",
        references=trusted_references("jason", "sacks", "friedberg", "chamath"),
        windows=(
            AttributionWindow("Review: Sacks leverage reply", 51.462, 54.212, "Sacks"),
            AttributionWindow("Review: Sacks market monologue", 365.430, 514.898, "Sacks"),
            AttributionWindow("Review: Friedberg conviction answer", 775.620, 822.632, "Friedberg"),
            AttributionWindow("Review: Friedberg China argument", 1195.380, 1308.0, "Friedberg"),
            AttributionWindow("Review: Jason takes over", 1308.0, 1340.382, "Jason"),
            AttributionWindow("Review: Chamath risk-parity answer", 1386.042, 1416.318, "Chamath"),
            AttributionWindow("Review: Friedberg fusion setup", 1696.966, 1756.116, "Friedberg"),
            AttributionWindow("Review: Friedberg fusion explanation", 1759.016, 1828.578, "Friedberg"),
            AttributionWindow("Review: Chamath solar objection", 1828.578, 1835.278, "Chamath"),
            AttributionWindow("Review: Friedberg debate reply", 1852.128, 1872.128, "Friedberg"),
            AttributionWindow("Review: Chamath electron argument", 1875.638, 1897.388, "Chamath"),
            AttributionWindow("Review: Friedberg nonlinear reply", 1898.038, 1922.988, "Friedberg"),
            AttributionWindow("Review: Chamath sun argument", 1922.934, 1939.134, "Chamath"),
            AttributionWindow("Review: Friedberg closing reply", 1943.984, 1948.734, "Friedberg"),
            AttributionWindow("Review: Jason enters discussion", 1949.942, 1963.342, "Jason"),
            AttributionWindow("Review: Chamath electricity deficit", 1993.884, 2018.412, "Chamath"),
            AttributionWindow("Review: Friedberg robot response", 2018.412, 2023.162, "Friedberg"),
            AttributionWindow("Review: Chamath long-electrons reply", 2023.512, 2030.662, "Chamath"),
            AttributionWindow("Review: external Sam Altman clip", 2150.164, 2187.968, SIL_UNKNOWN_LABEL),
            AttributionWindow("Review: Sacks frontier-labs argument A", 2293.556, 2521.446, "Sacks"),
            AttributionWindow("Review: Sacks frontier-labs argument B", 2538.146, 2601.272, "Sacks"),
            AttributionWindow("Review: Sacks frontier-labs argument C", 2605.322, 2699.098, "Sacks"),
            AttributionWindow("Review: Jason takes the other side", 2699.948, 2724.218, "Jason"),
        ),
        cached_baseline_filename="openai-asr-resolved-segments.json",
    ),
    Episode(
        key="big-technology-three-speaker",
        video_id="eAPyqzAAeWU",
        title=(
            "Anthropic's Labs Lead On Fable's Capabilities + Building "
            "AI-Native Products - With Mike Krieger"
        ),
        context=(
            "Big Technology Podcast live interview with host Alex Kantrowitz, "
            "co-interviewer Lauren Goode, and guest Mike Krieger. Brief audience "
            "questions remain a separate Audience/Other identity."
        ),
        asr_filename="openai-asr-gpt-4o-transcribe-diarize-600s.json",
        references=(),
        channel_title="Big Technology Podcast",
        tags=("Big Technology Podcast", "Mike Krieger", "Alex Kantrowitz", "Lauren Goode"),
        cached_baseline_filename="openai-asr-resolved-segments.json",
        auto_reference_seed=True,
    ),
    Episode(
        key="all-in-278-guest-panel",
        video_id="w8ah_tA0yfg",
        title=(
            "All-In Podcast Episode 278: Friedberg Absent; "
            "Travis Kalanick and Gavin Baker Join"
        ),
        context=(
            "All-In Podcast panel with Jason Calacanis, Chamath Palihapitiya, "
            "David Sacks, guest Travis Kalanick, and guest Gavin Baker. "
            "David Friedberg is explicitly absent."
        ),
        asr_filename="openai-asr-gpt-4o-transcribe-diarize-600s.json",
        references=trusted_references("jason", "chamath", "sacks", "gavin"),
        cached_baseline_filename="openai-asr-resolved-segments.json",
        auto_reference_seed=True,
        supplement_auto_references=True,
    ),
    Episode(
        key="all-in-279-external-clip",
        video_id="wgdxSCsmS-Q",
        title="All-In Podcast Episode 279: Core Four with External Alex Karp Clip",
        context=(
            "All-In Podcast panel with Jason Calacanis, Chamath Palihapitiya, "
            "David Sacks, and David Friedberg. The episode opens with a played "
            "clip of Alex Karp, who is not one of the four hosts."
        ),
        asr_filename="openai-asr-gpt-4o-transcribe-diarize-600s.json",
        references=trusted_references("jason", "chamath", "sacks", "friedberg"),
        cached_baseline_filename="openai-asr-resolved-segments.json",
        windows=(
            AttributionWindow(
                "External clip: Alex Karp interview",
                83.366,
                132.916,
                SIL_UNKNOWN_LABEL,
            ),
        ),
    ),
    Episode(
        key="uncapped-founders-fund-panel",
        video_id="NpUcRftC3k0",
        title=(
            "Founders Fund on Truth-Seeking, Taiwan, and Whether You Can Still "
            "Beat the S&P"
        ),
        context=(
            "Uncapped panel hosted by Jack Altman with Ev Randle, Trae Stephens, "
            "and Delian Asparouhov. Four active speakers participate."
        ),
        asr_filename="openai-asr-gpt-4o-transcribe-diarize-600s.json",
        references=(
            SpeakerReference("Jack Altman", 27.694, 33.494),
            SpeakerReference("Ev Randle", 90.0, 98.0),
            SpeakerReference("Ev Randle", 120.0, 128.0),
            SpeakerReference("Ev Randle", 160.0, 168.0),
            SpeakerReference("Trae Stephens", 193.0, 200.0),
            SpeakerReference("Trae Stephens", 204.0, 211.0),
            SpeakerReference("Trae Stephens", 213.0, 220.0),
            SpeakerReference("Delian Asparouhov", 221.5, 225.0),
            SpeakerReference("Delian Asparouhov", 225.0, 228.5),
            SpeakerReference("Delian Asparouhov", 228.5, 232.0),
        ),
        channel_title="Uncapped with Jack Altman",
        tags=(
            "Uncapped with Jack Altman",
            "Jack Altman",
            "Ev Randle",
            "Trae Stephens",
            "Delian Asparouhov",
        ),
        windows=(
            AttributionWindow("Transcript: Jack opens panel", 27.694, 33.494, "Jack Altman"),
            AttributionWindow("Transcript: Ev opening answer", 84.020, 189.278, "Ev Randle"),
            AttributionWindow("Transcript: Trae 1% answer", 192.372, 220.622, "Trae Stephens"),
            AttributionWindow(
                "Transcript: Delian comparative-basis reply",
                221.442,
                232.092,
                "Delian Asparouhov",
            ),
        ),
    ),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


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


class RequestRecorder:
    def __init__(self, effort: str) -> None:
        self.effort = effort
        self.stage = "unknown"
        self.records: List[Dict[str, Any]] = []
        self.phase_seconds: Dict[str, float] = defaultdict(float)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self.stage
        self.stage = name
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phase_seconds[name] += time.perf_counter() - started
            self.stage = previous

    def summary(self) -> Dict[str, Any]:
        successful = [record for record in self.records if record.get("status") == "ok"]
        input_tokens = sum(int(item.get("input_tokens") or 0) for item in successful)
        cached_tokens = sum(int(item.get("cached_input_tokens") or 0) for item in successful)
        output_tokens = sum(int(item.get("output_tokens") or 0) for item in successful)
        estimated_cost = (
            max(0, input_tokens - cached_tokens) * INPUT_PRICE_PER_MILLION
            + cached_tokens * CACHED_INPUT_PRICE_PER_MILLION
            + output_tokens * OUTPUT_PRICE_PER_MILLION
        ) / 1_000_000
        return {
            "effort": self.effort,
            "request_count": len(self.records),
            "api_seconds": round(sum(float(item.get("seconds") or 0) for item in self.records), 3),
            "phase_seconds": {key: round(value, 3) for key, value in self.phase_seconds.items()},
            **{
                key: sum(int(item.get(key) or 0) for item in successful)
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                )
            },
            "estimated_cost_usd": round(estimated_cost, 6),
            "requests": self.records,
        }


class InstrumentedResponses:
    def __init__(self, responses: Any, recorder: RequestRecorder) -> None:
        self.responses = responses
        self.recorder = recorder

    def create(self, **kwargs: Any) -> Any:
        request = dict(kwargs)
        request.pop("temperature", None)
        request["reasoning"] = {"effort": self.recorder.effort}
        started = time.perf_counter()
        try:
            response = self.responses.create(**request)
        except Exception as exc:
            self.recorder.records.append(
                {
                    "stage": self.recorder.stage,
                    "status": "error",
                    "seconds": round(time.perf_counter() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        self.recorder.records.append(
            {
                "stage": self.recorder.stage,
                "status": "ok",
                "seconds": round(time.perf_counter() - started, 3),
                **usage_record(response),
            }
        )
        return response


class InstrumentedClient:
    def __init__(self, client: OpenAI, recorder: RequestRecorder) -> None:
        self.responses = InstrumentedResponses(client.responses, recorder)


def boundary_schema(segment_count: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "boundaries": {
                "type": "array",
                "minItems": segment_count,
                "maxItems": segment_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "segment_id": {"type": "integer"},
                        "boundary_before": {
                            "type": "string",
                            "enum": ["same", "uncertain", "change"],
                        },
                    },
                    "required": ["segment_id", "boundary_before"],
                },
            }
        },
        "required": ["boundaries"],
    }


def boundary_system_prompt() -> str:
    return (
        "Detect anonymous speaker-turn boundaries in an ASR transcript. Do not identify or name speakers. "
        "For every TARGET segment, classify whether the same real person continues from the immediately "
        "preceding transcript segment: 'same' means confidently the same person, 'change' means confidently "
        "a different person starts, and 'uncertain' means the evidence is insufficient. Local diarizer labels "
        "are weak hints and can both merge two people and split one uninterrupted speaker. Use grammar, sentence "
        "continuity, overlapping timestamps, direct address, question-answer flow, interruptions, and discourse. "
        "Use 'same' or 'change' only when confident because they become clustering constraints; otherwise choose "
        "'uncertain'. Return each TARGET segment exactly once with its exact segment_id."
    )


def boundary_batches(
    segment_count: int,
    batch_size: int = BOUNDARY_BATCH_SEGMENTS,
    context_size: int = BOUNDARY_CONTEXT_SEGMENTS,
) -> List[Tuple[range, range]]:
    batches = []
    for start in range(0, segment_count, batch_size):
        stop = min(segment_count, start + batch_size)
        context_start = max(0, start - context_size)
        context_stop = min(segment_count, stop + context_size)
        batches.append((range(start, stop), range(context_start, context_stop)))
    return batches


def boundary_user_prompt(
    episode: Episode,
    segments: Sequence[Dict[str, Any]],
    target_ids: range,
    context_ids: range,
) -> str:
    target_set = set(target_ids)
    lines = [
        f"Title: {episode.title}",
        f"Context: {episode.context}",
        "",
        "Transcript. Classify TARGET lines only; BEFORE and AFTER are context:",
    ]
    for segment_id in context_ids:
        segment = segments[segment_id]
        marker = "TARGET" if segment_id in target_set else ("BEFORE" if segment_id < target_ids.start else "AFTER")
        chunk_index, local_speaker = ytranslate.get_segment_local_key(segment)
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        lines.append(
            f"{marker} {segment_id} | {start:.3f}-{end:.3f} | "
            f"local={chunk_index}/{local_speaker} | text={clean_text(segment.get('text'))}"
        )
    return "\n".join(lines)


def validate_boundary_assignments(
    assignments: Sequence[Dict[str, Any]],
    target_ids: Iterable[int],
) -> Dict[int, str]:
    expected = set(target_ids)
    result: Dict[int, str] = {}
    for assignment in assignments:
        segment_id = int(assignment.get("segment_id"))
        boundary = str(assignment.get("boundary_before") or "")
        if segment_id in result:
            raise RuntimeError(f"Duplicate boundary result for segment {segment_id}")
        if boundary not in BOUNDARY_SCORE:
            raise RuntimeError(f"Invalid boundary value {boundary!r} for segment {segment_id}")
        result[segment_id] = boundary
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise RuntimeError(f"Boundary IDs mismatch; missing={missing[:10]}, extra={extra[:10]}")
    return result


def infer_boundaries(
    client: InstrumentedClient,
    episode: Episode,
    segments: List[Dict[str, Any]],
    batch_dir: Path,
    log: Any,
) -> List[str]:
    boundaries = ["uncertain"] * len(segments)
    boundaries[0] = "change"
    batches = boundary_batches(len(segments))
    for batch_number, (target_ids, context_ids) in enumerate(batches, 1):
        batch_path = batch_dir / f"batch-{batch_number:02d}.json"
        if batch_path.exists():
            result = read_json(batch_path)
        else:
            log(f"Boundary batch {batch_number}/{len(batches)} ({len(target_ids)} segments)")
            result = ytranslate.call_openai_with_retry(
                client,
                MODEL,
                boundary_system_prompt(),
                boundary_user_prompt(episode, segments, target_ids, context_ids),
                schema_name=f"anonymous_turn_boundaries_{batch_number}",
                schema=boundary_schema(len(target_ids)),
                temperature=0.0,
            )
            write_json(batch_path, result)
        validated = validate_boundary_assignments(result.get("boundaries", []), target_ids)
        for segment_id, boundary in validated.items():
            boundaries[segment_id] = boundary
    boundaries[0] = "change"
    return boundaries


def normalized_embedding(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Cannot normalize a zero-length embedding")
    return vector / norm


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


def build_embedding_units(
    segments: Sequence[Dict[str, Any]],
    boundaries: Sequence[str],
    respect_local_changes: bool = False,
) -> List[Dict[str, Any]]:
    if len(segments) != len(boundaries):
        raise ValueError("Segment and boundary counts differ")
    if not segments:
        return []

    anonymous_turns: List[Dict[str, Any]] = []
    active: Optional[Dict[str, Any]] = None
    for segment_id, (segment, boundary) in enumerate(zip(segments, boundaries)):
        start = float(segment.get("start") or 0)
        end = max(start, float(segment.get("end") or start))
        previous_end = float(segments[segment_id - 1].get("end") or start) if segment_id else start
        local_changed = bool(
            respect_local_changes
            and segment_id
            and ytranslate.get_segment_local_key(segment)
            != ytranslate.get_segment_local_key(segments[segment_id - 1])
        )
        starts_new_turn = (
            active is None
            or boundary != "same"
            or local_changed
            or start - previous_end > 1.5
        )
        if starts_new_turn:
            active = {
                "turn_id": len(anonymous_turns),
                "start": start,
                "end": end,
                "boundary_before": "change" if segment_id == 0 else boundary,
                "segment_ids": [segment_id],
            }
            anonymous_turns.append(active)
        else:
            active["end"] = max(float(active["end"]), end)
            active["segment_ids"].append(segment_id)

    units: List[Dict[str, Any]] = []
    for turn in anonymous_turns:
        for split_index, (start, end) in enumerate(
            split_span(float(turn["start"]), float(turn["end"]), MAX_EMBEDDING_UNIT_SECONDS)
        ):
            overlapping_segment_ids = [
                segment_id
                for segment_id in turn["segment_ids"]
                if max(
                    0.0,
                    min(end, float(segments[segment_id].get("end") or end))
                    - max(start, float(segments[segment_id].get("start") or start)),
                )
                > 0
            ]
            units.append(
                {
                    "unit_id": len(units),
                    "start": start,
                    "end": end,
                    "duration": max(0.0, end - start),
                    "anonymous_turn_id": int(turn["turn_id"]),
                    "boundary_before": (
                        str(turn["boundary_before"]) if split_index == 0 else "same"
                    ),
                    "boundary_score": (
                        BOUNDARY_SCORE[str(turn["boundary_before"])] if split_index == 0 else 0.0
                    ),
                    "segment_ids": overlapping_segment_ids,
                }
            )
    return units


def build_sil_units(
    segments: Sequence[Dict[str, Any]],
    boundaries: Sequence[str],
) -> List[Dict[str, Any]]:
    """Build acoustic units without assuming a predicted turn is one identity.

    Local diarizer changes and Luna changes both open a fresh unit, but neither
    becomes a hard identity constraint. Long spans remain split so a missed
    boundary can be recovered from sustained acoustic evidence.
    """
    if len(segments) != len(boundaries):
        raise ValueError("Segment and boundary counts differ")

    pieces: List[Dict[str, Any]] = []
    previous_local_key: Optional[Tuple[int, str]] = None
    for segment_id, (segment, boundary) in enumerate(zip(segments, boundaries)):
        start = float(segment.get("start") or 0)
        end = max(start, float(segment.get("end") or start))
        chunk_index, local_speaker = ytranslate.get_segment_local_key(segment)
        local_key = (int(chunk_index), str(local_speaker))
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


def speaker_score_matrix(
    embeddings: np.ndarray,
    units: Sequence[Dict[str, Any]],
    references: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Blend fixed reference voices with conservative episode self-enrollment."""
    names = list(references)
    reference_matrix = np.vstack([references[name] for name in names])
    reference_scores = embeddings @ reference_matrix.T
    winners = np.argmax(reference_scores, axis=1)
    sorted_scores = np.sort(reference_scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2] if len(names) > 1 else sorted_scores[:, -1]

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
        centroid = ytranslate.robust_embedding_centroid(
            [[float(value) for value in embeddings[index]] for index in candidates]
        )
        prototype = (
            normalized_embedding(centroid)
            if centroid is not None
            else references[name]
        )
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
    """Add an open-set state that wins only across sustained weak matches."""
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
    """Decode one episode-wide identity path with weak boundary priors."""
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
                float(transition_scores[winner]) + float(scores[index, state]) * duration_weight
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
                    "best_known_similarity": round(
                        float(scores[index, best_known]), 4
                    ),
                    "known_margin": round(
                        float(
                            scores[index, best_known]
                            - scores[index, second_known]
                        ),
                        4,
                    ),
                }
            )
        assignments.append(assignment)
    return labels, {
        "assignments": assignments,
        "speaker_unit_counts": dict(Counter(item["speaker"] for item in assignments)),
    }


def pool_sil_units(
    scores: np.ndarray,
    units: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[List[int]]]:
    """Pool adjacent evidence while retaining every plausible handoff."""
    groups: List[Dict[str, Any]] = []
    group_unit_indexes: List[List[int]] = []
    for unit_index, unit in enumerate(units):
        previous = units[unit_index - 1] if unit_index else None
        gap = float(unit["start"]) - float(previous["end"]) if previous else math.inf
        starts_group = bool(
            previous is None
            or bool(unit.get("local_change_before"))
            or bool(unit.get("chunk_change_before"))
            or str(unit.get("boundary_before") or "uncertain") == "change"
            or gap > 1.5
        )
        if starts_group:
            copied = dict(unit)
            copied["unit_id"] = len(groups)
            copied["source_unit_ids"] = [int(unit["unit_id"])]
            copied["duration"] = max(0.001, float(unit.get("duration") or 0))
            copied["segment_ids"] = list(unit["segment_ids"])
            groups.append(copied)
            group_unit_indexes.append([unit_index])
        else:
            group = groups[-1]
            group["end"] = max(float(group["end"]), float(unit["end"]))
            group["duration"] += max(0.001, float(unit.get("duration") or 0))
            group["source_unit_ids"].append(int(unit["unit_id"]))
            group["segment_ids"].extend(
                segment_id
                for segment_id in unit["segment_ids"]
                if segment_id not in group["segment_ids"]
            )
            group_unit_indexes[-1].append(unit_index)

    pooled_scores: List[np.ndarray] = []
    for indexes in group_unit_indexes:
        weights = np.asarray(
            [max(0.001, float(units[index]["duration"])) for index in indexes],
            dtype=np.float32,
        )
        pooled_scores.append(np.average(scores[indexes], axis=0, weights=weights))
    return np.vstack(pooled_scores), groups, group_unit_indexes


def cached_audio_path(video_id: str) -> Path:
    candidates = [
        path
        for path in sorted((CACHE_ROOT / video_id / "audio").glob("source.*"))
        if not path.name.endswith(".part")
    ]
    if not candidates:
        raise FileNotFoundError(f"No cached source audio for {video_id}")
    return candidates[0]


def load_cached_audio(video_id: str) -> Tuple[np.ndarray, int]:
    wav_path = CACHE_ROOT / video_id / "voice-speaker-reconciliation" / "source-16k.wav"
    if not wav_path.exists():
        ytranslate.transcode_audio_for_voice_reconciliation(
            str(cached_audio_path(video_id)), video_id, print
        )
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def load_episode_audio(episode: Episode) -> Tuple[np.ndarray, int]:
    return load_cached_audio(episode.video_id)


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
    encoder: VoiceEncoder,
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


def turn_constraint_matrix(units: Sequence[Dict[str, Any]]) -> np.ndarray:
    return ConstraintMatrix(
        [float(unit["boundary_score"]) for unit in units], threshold=1.0
    ).compute_diagonals()


def consolidate_anonymous_turn_clusters(
    labels: Sequence[int],
    units: Sequence[Dict[str, Any]],
) -> np.ndarray:
    if len(labels) != len(units):
        raise ValueError("Cluster label and unit counts differ")
    resolved = np.asarray(labels, dtype=int).copy()
    by_turn: Dict[int, List[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        by_turn[int(unit["anonymous_turn_id"])].append(index)
    for indexes in by_turn.values():
        duration_by_cluster: Dict[int, float] = defaultdict(float)
        for index in indexes:
            duration_by_cluster[int(resolved[index])] += max(
                0.001, float(units[index].get("duration") or 0)
            )
        winner = max(duration_by_cluster.items(), key=lambda item: (item[1], -item[0]))[0]
        resolved[indexes] = winner
    return resolved


def cluster_embeddings(
    embeddings: np.ndarray,
    units: Sequence[Dict[str, Any]],
    speaker_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    clusterer = copy.deepcopy(configs.turntodiarize_clusterer)
    clusterer.min_clusters = speaker_count
    clusterer.max_clusters = speaker_count
    constraints = turn_constraint_matrix(units)
    raw_labels = np.asarray(clusterer.predict(embeddings, constraints), dtype=int)
    return raw_labels, consolidate_anonymous_turn_clusters(raw_labels, units)


def select_cached_speaker_references(
    episode: Episode,
    baseline_segments: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[SpeakerReference, ...]:
    mapping = read_json(episode.cache_dir / "speaker-mapping-effective.json")
    if baseline_segments is None:
        segments = read_json(episode.cache_dir / "openai-asr-resolved-segments.json")
        speakers = list(mapping.get("speakers", []))
    else:
        segments = [dict(segment) for segment in baseline_segments]
        speakers_by_identity: Dict[str, Dict[str, str]] = {}
        for segment in segments:
            speaker_id = str(segment.get("speaker_id") or "")
            label = str(segment.get("speaker_label") or speaker_id)
            identity = canonical_speaker_name(label)
            if not speaker_id or identity in speakers_by_identity:
                continue
            speakers_by_identity[identity] = {
                "id": speaker_id,
                "label_short": label,
                "label_full": label,
            }
        speakers = list(speakers_by_identity.values())
    selected: List[SpeakerReference] = []
    for speaker in speakers:
        speaker_id = str(speaker.get("id") or "")
        label = str(
            speaker.get("label_short") or speaker.get("label_full") or speaker_id
        )
        candidates = [
            segment
            for segment in segments
            if (
                str(segment.get("speaker_id") or "") == speaker_id
                or canonical_speaker_name(segment.get("speaker_label"))
                == canonical_speaker_name(label)
            )
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
                for segment in segments
                if (
                    str(segment.get("speaker_id") or "") == speaker_id
                    or canonical_speaker_name(segment.get("speaker_label"))
                    == canonical_speaker_name(label)
                )
                and float(segment.get("end") or 0) - float(segment.get("start") or 0)
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
        speaker_references: List[SpeakerReference] = []
        for candidate in candidates:
            start = float(candidate.get("start") or 0)
            if any(
                abs(start - reference.start) < AUTO_REFERENCE_MIN_SEPARATION_SECONDS
                for reference in speaker_references
            ):
                continue
            speaker_references.append(
                SpeakerReference(
                    label,
                    start,
                    float(candidate.get("end") or start),
                    episode.video_id,
                )
            )
            if len(speaker_references) >= AUTO_REFERENCE_CLIPS_PER_SPEAKER:
                break
        if not speaker_references:
            continue
        selected.extend(speaker_references)
    if not selected:
        raise RuntimeError(f"No automatic speaker references for {episode.key}")
    return tuple(selected)


def resolve_speaker_references(
    episode: Episode,
    baseline_segments: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[SpeakerReference, ...]:
    if episode.references and not episode.supplement_auto_references:
        return episode.references
    if episode.auto_reference_seed:
        automatic = select_cached_speaker_references(episode, baseline_segments)
        explicit_identities = {
            canonical_speaker_name(reference.name) for reference in episode.references
        }
        supplemental = tuple(
            reference
            for reference in automatic
            if canonical_speaker_name(reference.name) not in explicit_identities
        )
        return episode.references + supplemental
    raise RuntimeError(f"No speaker references configured for {episode.key}")


def embed_references(
    episode: Episode,
    audio: np.ndarray,
    sample_rate: int,
    encoder: VoiceEncoder,
    speaker_references: Optional[Sequence[SpeakerReference]] = None,
) -> Dict[str, np.ndarray]:
    references = tuple(speaker_references or resolve_speaker_references(episode))
    audio_by_video_id: Dict[str, Tuple[np.ndarray, int]] = {
        episode.video_id: (audio, sample_rate)
    }
    embeddings_by_name: Dict[str, List[List[float]]] = defaultdict(list)
    for reference in references:
        source_video_id = reference.source_video_id or episode.video_id
        if source_video_id not in audio_by_video_id:
            audio_by_video_id[source_video_id] = load_cached_audio(source_video_id)
        source_audio, source_sample_rate = audio_by_video_id[source_video_id]
        embedding = normalized_embedding(
            encoder.embed_utterance(
                audio_clip(
                    source_audio,
                    source_sample_rate,
                    reference.start,
                    reference.end,
                    minimum_seconds=0.1,
                )
            )
        )
        embeddings_by_name[reference.name].append(
            [float(value) for value in embedding]
        )
    resolved: Dict[str, np.ndarray] = {}
    for name, embeddings in embeddings_by_name.items():
        centroid = ytranslate.robust_embedding_centroid(embeddings)
        if centroid is None:
            raise RuntimeError(f"No reference centroid for {name} in {episode.key}")
        resolved[name] = normalized_embedding(centroid)
    return resolved


def cluster_centroids(
    embeddings: np.ndarray,
    labels: Sequence[int],
) -> Dict[int, np.ndarray]:
    centroids: Dict[int, np.ndarray] = {}
    for cluster_id in sorted(set(int(label) for label in labels)):
        members = [
            [float(value) for value in embedding]
            for embedding, label in zip(embeddings, labels)
            if int(label) == cluster_id
        ]
        centroid = ytranslate.robust_embedding_centroid(members)
        if centroid is None:
            raise RuntimeError(f"No centroid for cluster {cluster_id}")
        centroids[cluster_id] = normalized_embedding(centroid)
    return centroids


def map_clusters_to_references(
    centroids: Dict[int, np.ndarray],
    references: Dict[str, np.ndarray],
    minimum_similarity: float = 0.62,
    minimum_margin: float = 0.015,
) -> Tuple[Dict[int, str], Dict[str, Any]]:
    cluster_ids = sorted(centroids)
    reference_names = list(references)
    if len(cluster_ids) != len(reference_names):
        raise ValueError("This controlled experiment requires one reference per cluster")
    scores = np.asarray(
        [
            [float(np.dot(centroids[cluster_id], references[name])) for name in reference_names]
            for cluster_id in cluster_ids
        ],
        dtype=np.float32,
    )
    row_indexes, column_indexes = linear_sum_assignment(-scores)
    mapping: Dict[int, str] = {}
    assignments: List[Dict[str, Any]] = []
    for row_index, column_index in zip(row_indexes, column_indexes):
        cluster_id = cluster_ids[int(row_index)]
        name = reference_names[int(column_index)]
        score = float(scores[row_index, column_index])
        alternatives = [
            float(value)
            for index, value in enumerate(scores[row_index])
            if index != int(column_index)
        ]
        margin = score - max(alternatives, default=-1.0)
        accepted = score >= minimum_similarity and margin >= minimum_margin
        mapping[cluster_id] = name if accepted else f"Speaker Cluster {cluster_id + 1}"
        assignments.append(
            {
                "cluster_id": cluster_id,
                "speaker": name,
                "similarity": round(score, 4),
                "margin": round(margin, 4),
                "accepted": accepted,
                "effective_label": mapping[cluster_id],
            }
        )
    return mapping, {
        "cluster_ids": cluster_ids,
        "reference_names": reference_names,
        "score_matrix": [[round(float(value), 4) for value in row] for row in scores],
        "assignments": assignments,
    }


def classify_anonymous_turns_by_reference(
    embeddings: np.ndarray,
    units: Sequence[Dict[str, Any]],
    references: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[int, str], Dict[str, Any]]:
    reference_names = list(references)
    reference_matrix = np.vstack([references[name] for name in reference_names])
    labels = np.zeros(len(units), dtype=int)
    assignments: List[Dict[str, Any]] = []
    by_turn: Dict[int, List[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        by_turn[int(unit["anonymous_turn_id"])].append(index)

    for turn_id, indexes in sorted(by_turn.items()):
        weights = np.asarray(
            [max(0.001, float(units[index].get("duration") or 0)) for index in indexes],
            dtype=np.float32,
        )
        centroid = normalized_embedding(np.average(embeddings[indexes], axis=0, weights=weights))
        scores = centroid @ reference_matrix.T
        order = np.argsort(scores)[::-1]
        winner = int(order[0])
        runner_up = int(order[1]) if len(order) > 1 else winner
        labels[indexes] = winner
        assignments.append(
            {
                "anonymous_turn_id": turn_id,
                "speaker": reference_names[winner],
                "similarity": round(float(scores[winner]), 4),
                "margin": round(float(scores[winner] - scores[runner_up]), 4),
                "start": round(min(float(units[index]["start"]) for index in indexes), 3),
                "end": round(max(float(units[index]["end"]) for index in indexes), 3),
            }
        )

    similarities = [float(item["similarity"]) for item in assignments]
    margins = [float(item["margin"]) for item in assignments]
    return labels, dict(enumerate(reference_names)), {
        "reference_names": reference_names,
        "turn_count": len(assignments),
        "speaker_turn_counts": dict(Counter(item["speaker"] for item in assignments)),
        "similarity_percentiles": {
            str(percentile): round(float(np.percentile(similarities, percentile)), 4)
            for percentile in (10, 25, 50, 75, 90)
        }
        if similarities
        else {},
        "margin_percentiles": {
            str(percentile): round(float(np.percentile(margins, percentile)), 4)
            for percentile in (10, 25, 50, 75, 90)
        }
        if margins
        else {},
        "ambiguous_turn_count": sum(margin < 0.02 for margin in margins),
        "assignments": assignments,
    }


def overlap_seconds(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def assign_segments_from_units(
    segments: Sequence[Dict[str, Any]],
    units: Sequence[Dict[str, Any]],
    unit_labels: Sequence[int],
    cluster_names: Dict[int, str],
    boundaries: Sequence[str],
) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for segment_id, segment in enumerate(segments):
        start = float(segment.get("start") or 0)
        end = max(start, float(segment.get("end") or start))
        votes: Dict[int, float] = defaultdict(float)
        for unit, label in zip(units, unit_labels):
            overlap = overlap_seconds(start, end, float(unit["start"]), float(unit["end"]))
            if overlap > 0:
                votes[int(label)] += overlap
        if votes:
            cluster_id = max(votes.items(), key=lambda item: (item[1], -item[0]))[0]
        else:
            midpoint = (start + end) / 2
            nearest_index = min(
                range(len(units)),
                key=lambda index: abs(
                    midpoint - (float(units[index]["start"]) + float(units[index]["end"])) / 2
                ),
            )
            cluster_id = int(unit_labels[nearest_index])
        label = cluster_names[cluster_id]
        copied = dict(segment)
        copied.update(
            {
                "segment_id": segment_id,
                "cluster_id": cluster_id,
                "speaker_id": ytranslate.speaker_id_from_label(label),
                "speaker_label": label,
                "speaker_id_source": "turn-constrained-clustering",
                "boundary_before": boundaries[segment_id],
            }
        )
        resolved.append(copied)
    return resolved


def assign_segments_from_sil_units(
    segments: Sequence[Dict[str, Any]],
    units: Sequence[Dict[str, Any]],
    unit_labels: Sequence[int],
    speaker_names: Sequence[str],
    boundaries: Sequence[str],
) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for segment_id, segment in enumerate(segments):
        start = float(segment.get("start") or 0)
        end = max(start, float(segment.get("end") or start))
        votes: Dict[int, float] = defaultdict(float)
        for unit, label in zip(units, unit_labels):
            if segment_id not in unit["segment_ids"]:
                continue
            overlap = overlap_seconds(start, end, float(unit["start"]), float(unit["end"]))
            votes[int(label)] += max(overlap, 0.001)
        if votes:
            speaker_index = max(votes.items(), key=lambda item: (item[1], -item[0]))[0]
        else:
            midpoint = (start + end) / 2
            nearest_index = min(
                range(len(units)),
                key=lambda index: abs(
                    midpoint - (float(units[index]["start"]) + float(units[index]["end"])) / 2
                ),
            )
            speaker_index = int(unit_labels[nearest_index])
        label = str(speaker_names[speaker_index])
        copied = dict(segment)
        copied.update(
            {
                "segment_id": segment_id,
                "cluster_id": speaker_index,
                "speaker_id": ytranslate.speaker_id_from_label(label),
                "speaker_label": label,
                "speaker_id_source": "sil-global-acoustic",
                "boundary_before": boundaries[segment_id],
            }
        )
        resolved.append(copied)
    return resolved


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
    """Apply decoded identities only after sustained local evidence.

    The baseline remains the fallback for local ambiguity. A known-identity
    correction activates from repeated strong unit evidence inside one
    contiguous diarization-track run, without requiring that the flawed
    baseline first label that target correctly. Unknown is emitted only for a
    sustained decoded run.
    """
    if len(baseline_segments) != len(boundaries):
        raise ValueError("Baseline segment and boundary counts differ")

    unknown_identity = canonical_speaker_name(SIL_UNKNOWN_LABEL)
    enriched: List[Dict[str, Any]] = []
    strong_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        copied = dict(assignment)
        baseline_identity = _baseline_identity_for_assignment(
            copied, baseline_segments
        )
        candidate_identity = canonical_speaker_name(copied.get("speaker"))
        copied["baseline_identity"] = baseline_identity
        copied["candidate_identity"] = candidate_identity
        if (
            candidate_identity not in {baseline_identity, unknown_identity}
            and float(copied.get("similarity") or 0)
            >= SIL_UNIT_REPAIR_MIN_SIMILARITY
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

    repair_candidates: Dict[
        Tuple[str, str, str], List[Dict[str, Any]]
    ] = defaultdict(list)
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
                and float(assignment["start"])
                - float(scoped_runs[-1][-1]["end"])
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
                if float(item.get("similarity") or 0)
                >= SIL_UNIT_REPAIR_MIN_SIMILARITY
                and float(item.get("margin") or 0) >= SIL_UNIT_REPAIR_MIN_MARGIN
            ]
            strong_seconds = sum(
                float(item.get("duration") or 0) for item in strong
            )
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
                        {
                            int(item.get("chunk_index", -1))
                            for item in run
                        }
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
        total_weight = sum(max(0.001, float(item.get("duration") or 0)) for item in run)
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
                "best_known_similarity": round(
                    best_known_peak, 4
                ),
            }
        )

    votes: List[Dict[str, float]] = [defaultdict(float) for _ in baseline_segments]
    display_labels: Dict[str, str] = {}
    for segment in baseline_segments:
        label = str(segment.get("speaker_label") or segment.get("speaker_id") or "Unknown")
        display_labels[canonical_speaker_name(label)] = label
    display_labels[unknown_identity] = SIL_UNKNOWN_LABEL
    direct_repair_unit_ids: set[int] = set()
    for assignment in enriched:
        candidate_identity = str(assignment["candidate_identity"])
        baseline_identity = str(assignment["baseline_identity"])
        unit_id = int(assignment["unit_id"])
        direct_high_confidence = bool(
            float(assignment.get("similarity") or 0)
            >= SIL_UNIT_REPAIR_MIN_SIMILARITY
            and float(assignment.get("margin") or 0)
            >= SIL_UNIT_REPAIR_DIRECT_MIN_MARGIN
        )
        direct_short = bool(
            unit_id in isolated_short_unit_ids
            and float(assignment.get("similarity") or 0)
            >= SIL_UNIT_REPAIR_SHORT_MIN_SIMILARITY
            and float(assignment.get("margin") or 0)
            >= SIL_UNIT_REPAIR_SHORT_MIN_MARGIN
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
                    and (segment_baseline_identity, candidate_identity)
                    in activated_pairs
                )
                or (
                    unit_id,
                    segment_baseline_identity,
                    candidate_identity,
                )
                in active_known_unit_pairs
            ):
                effective_identity = candidate_identity
                display_labels.setdefault(
                    candidate_identity, str(assignment["speaker"])
                )
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
                    "speaker_id": ytranslate.speaker_id_from_label(label),
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
            "unknown_run_max_average_similarity": (
                SIL_UNKNOWN_RUN_MAX_AVERAGE_SIMILARITY
            ),
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


def apply_conservative_turn_repairs(
    baseline_segments: Sequence[Dict[str, Any]],
    reference_segments: Sequence[Dict[str, Any]],
    reference_assignments: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(baseline_segments) != len(reference_segments):
        raise ValueError("Baseline and reference-classified segment counts differ")
    assignments = [dict(assignment) for assignment in reference_assignments]
    for assignment in assignments:
        duration_by_baseline_speaker: Dict[str, float] = defaultdict(float)
        segment_ids: List[int] = []
        for segment_id, segment in enumerate(baseline_segments):
            start = float(segment.get("start") or 0)
            end = max(start, float(segment.get("end") or start))
            overlap = overlap_seconds(
                start,
                end,
                float(assignment["start"]),
                float(assignment["end"]),
            )
            if overlap <= 0:
                continue
            segment_ids.append(segment_id)
            duration_by_baseline_speaker[
                canonical_speaker_name(
                    segment.get("speaker_label") or segment.get("speaker_id")
                )
            ] += overlap
        assignment["baseline_speaker"] = (
            max(duration_by_baseline_speaker, key=duration_by_baseline_speaker.get)
            if duration_by_baseline_speaker
            else "unknown"
        )
        assignment["reference_speaker"] = canonical_speaker_name(assignment["speaker"])
        assignment["segment_ids"] = segment_ids
        assignment["repair"] = False
        assignment["repair_reason"] = None

    for index, assignment in enumerate(assignments):
        if index == 0:
            continue
        previous = assignments[index - 1]
        duration = float(assignment["end"]) - float(assignment["start"])
        current_confident = (
            float(assignment["similarity"]) >= REPAIR_MIN_SIMILARITY
            and float(assignment["margin"]) >= REPAIR_MIN_MARGIN
            and duration >= REPAIR_MIN_TURN_SECONDS
        )
        previous_confident = (
            float(previous["similarity"]) >= REPAIR_MIN_SIMILARITY
            and float(previous["margin"]) >= REPAIR_MIN_MARGIN
        )
        opens_repair = (
            current_confident
            and previous_confident
            and assignment["baseline_speaker"] == previous["baseline_speaker"]
            and assignment["reference_speaker"] != previous["reference_speaker"]
            and previous["reference_speaker"] == previous["baseline_speaker"]
            and assignment["reference_speaker"] != assignment["baseline_speaker"]
        )
        continues_repair = (
            current_confident
            and bool(previous["repair"])
            and assignment["baseline_speaker"] == previous["baseline_speaker"]
            and assignment["reference_speaker"] == previous["reference_speaker"]
        )
        if opens_repair or continues_repair:
            assignment["repair"] = True
            assignment["repair_reason"] = (
                "confident-handoff-inside-baseline-run"
                if opens_repair
                else "continuation-of-repaired-turn"
            )

    resolved = [dict(segment) for segment in baseline_segments]
    for segment_id, (segment, reference_segment) in enumerate(
        zip(resolved, reference_segments)
    ):
        segment["segment_id"] = segment_id
        if reference_segment.get("boundary_before"):
            segment["boundary_before"] = reference_segment["boundary_before"]
    repaired_segment_ids: set[int] = set()
    for assignment in assignments:
        if not assignment["repair"]:
            continue
        for segment_id in assignment["segment_ids"]:
            reference_segment = reference_segments[segment_id]
            resolved[segment_id]["speaker_id_before_turn_repair"] = resolved[segment_id].get(
                "speaker_id"
            )
            resolved[segment_id]["speaker_label_before_turn_repair"] = resolved[segment_id].get(
                "speaker_label"
            )
            resolved[segment_id]["speaker_id"] = reference_segment["speaker_id"]
            resolved[segment_id]["speaker_label"] = reference_segment["speaker_label"]
            resolved[segment_id]["speaker_id_source"] = "turn-constrained-repair"
            resolved[segment_id]["cluster_id"] = reference_segment.get("cluster_id")
            repaired_segment_ids.add(segment_id)
    return resolved, {
        "thresholds": {
            "minimum_similarity": REPAIR_MIN_SIMILARITY,
            "minimum_margin": REPAIR_MIN_MARGIN,
            "minimum_turn_seconds": REPAIR_MIN_TURN_SECONDS,
        },
        "repaired_turn_count": sum(bool(assignment["repair"]) for assignment in assignments),
        "repaired_segment_count": len(repaired_segment_ids),
        "repaired_seconds": round(
            sum(
                max(
                    0.0,
                    float(resolved[index].get("end") or 0)
                    - float(resolved[index].get("start") or 0),
                )
                for index in repaired_segment_ids
            ),
            3,
        ),
        "turns": assignments,
    }


def apply_missing_identity_repairs(
    baseline_segments: Sequence[Dict[str, Any]],
    repaired_segments: Sequence[Dict[str, Any]],
    reference_segments: Sequence[Dict[str, Any]],
    reference_assignments: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Recover an enrolled participant omitted from the baseline roster.

    Activation requires repeated strong acoustic evidence across the episode.
    This avoids forcing an unknown guest into a known host while addressing the
    catastrophic case where one real participant was globally collapsed away.
    """
    if not (
        len(baseline_segments) == len(repaired_segments) == len(reference_segments)
    ):
        raise ValueError("Missing-identity repair segment counts differ")

    baseline_identities = {
        canonical_speaker_name(
            segment.get("speaker_label") or segment.get("speaker_id")
        )
        for segment in baseline_segments
    }
    candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for assignment in reference_assignments:
        identity = canonical_speaker_name(assignment.get("speaker"))
        duration = float(assignment["end"]) - float(assignment["start"])
        strong = (
            identity not in baseline_identities
            and float(assignment["similarity"])
            >= SIL_MISSING_IDENTITY_MIN_SIMILARITY
            and float(assignment["margin"]) >= SIL_MISSING_IDENTITY_MIN_MARGIN
            and duration >= SIL_MISSING_IDENTITY_MIN_TURN_SECONDS
        )
        if strong:
            candidates[identity].append(dict(assignment))

    activated: Dict[str, List[Dict[str, Any]]] = {}
    for identity, assignments in candidates.items():
        total_seconds = sum(
            float(assignment["end"]) - float(assignment["start"])
            for assignment in assignments
        )
        if (
            len(assignments) >= SIL_MISSING_IDENTITY_ACTIVATION_TURNS
            and total_seconds >= SIL_MISSING_IDENTITY_ACTIVATION_SECONDS
        ):
            activated[identity] = assignments

    resolved = [dict(segment) for segment in repaired_segments]
    repaired_segment_ids: set[int] = set()
    for identity, assignments in activated.items():
        for assignment in assignments:
            for segment_id, segment in enumerate(baseline_segments):
                start = float(segment.get("start") or 0)
                end = max(start, float(segment.get("end") or start))
                midpoint = (start + end) / 2
                if not (
                    float(assignment["start"]) - 1e-6
                    <= midpoint
                    <= float(assignment["end"]) + 1e-6
                ):
                    continue
                reference_segment = reference_segments[segment_id]
                if canonical_speaker_name(
                    reference_segment.get("speaker_label")
                    or reference_segment.get("speaker_id")
                ) != identity:
                    continue
                resolved[segment_id]["speaker_id_before_sil"] = resolved[segment_id].get(
                    "speaker_id"
                )
                resolved[segment_id]["speaker_label_before_sil"] = resolved[segment_id].get(
                    "speaker_label"
                )
                resolved[segment_id]["speaker_id"] = reference_segment["speaker_id"]
                resolved[segment_id]["speaker_label"] = reference_segment["speaker_label"]
                resolved[segment_id]["speaker_id_source"] = "sil-missing-identity"
                repaired_segment_ids.add(segment_id)

    return resolved, {
        "thresholds": {
            "activation_seconds": SIL_MISSING_IDENTITY_ACTIVATION_SECONDS,
            "activation_turns": SIL_MISSING_IDENTITY_ACTIVATION_TURNS,
            "minimum_similarity": SIL_MISSING_IDENTITY_MIN_SIMILARITY,
            "minimum_margin": SIL_MISSING_IDENTITY_MIN_MARGIN,
            "minimum_turn_seconds": SIL_MISSING_IDENTITY_MIN_TURN_SECONDS,
        },
        "baseline_identities": sorted(baseline_identities),
        "activated_identities": {
            identity: {
                "turn_count": len(assignments),
                "seconds": round(
                    sum(
                        float(assignment["end"]) - float(assignment["start"])
                        for assignment in assignments
                    ),
                    3,
                ),
            }
            for identity, assignments in activated.items()
        },
        "repaired_segment_count": len(repaired_segment_ids),
    }


def apply_systematic_confusion_repairs(
    baseline_segments: Sequence[Dict[str, Any]],
    repaired_segments: Sequence[Dict[str, Any]],
    reference_segments: Sequence[Dict[str, Any]],
    reference_assignments: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Repair recurring known-speaker confusions with an anchored target."""
    baseline_identities = {
        canonical_speaker_name(
            segment.get("speaker_label") or segment.get("speaker_id")
        )
        for segment in baseline_segments
    }
    pair_assignments: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    self_assignments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for assignment in reference_assignments:
        duration = float(assignment["end"]) - float(assignment["start"])
        if (
            float(assignment["similarity"]) < SIL_MISSING_IDENTITY_MIN_SIMILARITY
            or float(assignment["margin"]) < SIL_MISSING_IDENTITY_MIN_MARGIN
            or duration < SIL_MISSING_IDENTITY_MIN_TURN_SECONDS
        ):
            continue
        baseline_seconds: Dict[str, float] = defaultdict(float)
        for segment in baseline_segments:
            overlap = overlap_seconds(
                float(segment.get("start") or 0),
                float(segment.get("end") or segment.get("start") or 0),
                float(assignment["start"]),
                float(assignment["end"]),
            )
            if overlap > 0:
                baseline_seconds[
                    canonical_speaker_name(
                        segment.get("speaker_label") or segment.get("speaker_id")
                    )
                ] += overlap
        if not baseline_seconds:
            continue
        baseline_identity = max(baseline_seconds, key=baseline_seconds.get)
        reference_identity = canonical_speaker_name(assignment.get("speaker"))
        copied = dict(assignment)
        copied["baseline_identity"] = baseline_identity
        copied["reference_identity"] = reference_identity
        if baseline_identity == reference_identity:
            self_assignments[reference_identity].append(copied)
            continue
        if reference_identity not in baseline_identities:
            continue
        pair_assignments[(baseline_identity, reference_identity)].append(copied)

    anchored_identities: Dict[str, List[Dict[str, Any]]] = {}
    for identity, assignments in self_assignments.items():
        seconds = sum(
            float(assignment["end"]) - float(assignment["start"])
            for assignment in assignments
        )
        if (
            len(assignments) >= SIL_TARGET_ANCHOR_TURNS
            and seconds >= SIL_TARGET_ANCHOR_SECONDS
        ):
            anchored_identities[identity] = assignments

    activated: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    rejected_unanchored: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for pair, assignments in pair_assignments.items():
        seconds = sum(
            float(assignment["end"]) - float(assignment["start"])
            for assignment in assignments
        )
        if (
            len(assignments) >= SIL_CONFUSION_PAIR_ACTIVATION_TURNS
            and seconds >= SIL_CONFUSION_PAIR_ACTIVATION_SECONDS
        ):
            if pair[1] in anchored_identities:
                activated[pair] = assignments
            else:
                rejected_unanchored[pair] = assignments

    resolved = [dict(segment) for segment in repaired_segments]
    repaired_segment_ids: set[int] = set()
    for (baseline_identity, reference_identity), _ in activated.items():
        matching_assignments = [
            assignment
            for assignment in reference_assignments
            if canonical_speaker_name(assignment.get("speaker")) == reference_identity
            and float(assignment["similarity"])
            >= SIL_MISSING_IDENTITY_MIN_SIMILARITY
            and float(assignment["margin"]) >= SIL_MISSING_IDENTITY_MIN_MARGIN
            and float(assignment["end"]) - float(assignment["start"])
            >= SIL_MISSING_IDENTITY_MIN_TURN_SECONDS
        ]
        for assignment in matching_assignments:
            for segment_id, baseline_segment in enumerate(baseline_segments):
                if canonical_speaker_name(
                    baseline_segment.get("speaker_label")
                    or baseline_segment.get("speaker_id")
                ) != baseline_identity:
                    continue
                start = float(baseline_segment.get("start") or 0)
                end = max(start, float(baseline_segment.get("end") or start))
                midpoint = (start + end) / 2
                if not (
                    float(assignment["start"]) - 1e-6
                    <= midpoint
                    <= float(assignment["end"]) + 1e-6
                ):
                    continue
                reference_segment = reference_segments[segment_id]
                if canonical_speaker_name(
                    reference_segment.get("speaker_label")
                    or reference_segment.get("speaker_id")
                ) != reference_identity:
                    continue
                resolved[segment_id]["speaker_id_before_sil"] = resolved[segment_id].get(
                    "speaker_id"
                )
                resolved[segment_id]["speaker_label_before_sil"] = resolved[segment_id].get(
                    "speaker_label"
                )
                resolved[segment_id]["speaker_id"] = reference_segment["speaker_id"]
                resolved[segment_id]["speaker_label"] = reference_segment["speaker_label"]
                resolved[segment_id]["speaker_id_source"] = "sil-systematic-confusion"
                repaired_segment_ids.add(segment_id)

    return resolved, {
        "thresholds": {
            "activation_seconds": SIL_CONFUSION_PAIR_ACTIVATION_SECONDS,
            "activation_turns": SIL_CONFUSION_PAIR_ACTIVATION_TURNS,
            "minimum_similarity": SIL_MISSING_IDENTITY_MIN_SIMILARITY,
            "minimum_margin": SIL_MISSING_IDENTITY_MIN_MARGIN,
            "minimum_turn_seconds": SIL_MISSING_IDENTITY_MIN_TURN_SECONDS,
        },
        "target_anchor_thresholds": {
            "seconds": SIL_TARGET_ANCHOR_SECONDS,
            "turns": SIL_TARGET_ANCHOR_TURNS,
        },
        "anchored_identities": {
            identity: {
                "turn_count": len(assignments),
                "seconds": round(
                    sum(
                        float(assignment["end"]) - float(assignment["start"])
                        for assignment in assignments
                    ),
                    3,
                ),
            }
            for identity, assignments in anchored_identities.items()
        },
        "activated_pairs": {
            f"{left}->{right}": {
                "turn_count": len(assignments),
                "seconds": round(
                    sum(
                        float(assignment["end"]) - float(assignment["start"])
                        for assignment in assignments
                    ),
                    3,
                ),
            }
            for (left, right), assignments in activated.items()
        },
        "rejected_unanchored_pairs": {
            f"{left}->{right}": {
                "turn_count": len(assignments),
                "seconds": round(
                    sum(
                        float(assignment["end"]) - float(assignment["start"])
                        for assignment in assignments
                    ),
                    3,
                ),
            }
            for (left, right), assignments in rejected_unanchored.items()
        },
        "repaired_segment_count": len(repaired_segment_ids),
    }


def turns_from_segments(
    segments: Sequence[Dict[str, Any]],
    max_gap_seconds: float = 1.5,
    max_turn_seconds: float = 25.0,
) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        text = clean_text(segment.get("text"))
        if not text:
            continue
        start = float(segment.get("start") or 0)
        end = max(start, float(segment.get("end") or start))
        speaker_id = str(segment.get("speaker_id") or "speaker")
        speaker_label = str(segment.get("speaker_label") or speaker_id)
        can_merge = (
            bool(turns)
            and turns[-1]["speaker_id"] == speaker_id
            and start - float(turns[-1]["end"]) <= max_gap_seconds
            and end - float(turns[-1]["start"]) <= max_turn_seconds
        )
        if can_merge:
            turns[-1]["end"] = max(float(turns[-1]["end"]), end)
            turns[-1]["text"] = clean_text(f"{turns[-1]['text']} {text}")
            turns[-1]["segment_ids"].append(segment_index)
        else:
            turns.append(
                {
                    "turn_id": len(turns),
                    "start": start,
                    "end": end,
                    "speaker_id": speaker_id,
                    "speaker_label": speaker_label,
                    "text": text,
                    "segment_ids": [segment_index],
                }
            )
    return turns


def canonical_speaker_name(value: Any) -> str:
    normalized = ytranslate.normalize_identity_text(str(value or ""))
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


def normalize_speaker_display_labels(
    segments: Sequence[Dict[str, Any]],
    baseline_segments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_labels: Dict[str, str] = {}
    for segment in baseline_segments:
        label = str(segment.get("speaker_label") or segment.get("speaker_id") or "speaker")
        baseline_labels.setdefault(canonical_speaker_name(label), label)
    normalized: List[Dict[str, Any]] = []
    for segment in segments:
        copied = dict(segment)
        label = str(copied.get("speaker_label") or copied.get("speaker_id") or "speaker")
        copied["speaker_label"] = baseline_labels.get(canonical_speaker_name(label), label)
        normalized.append(copied)
    return normalized


def public_speakers_from_segments(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    speakers: Dict[str, Dict[str, str]] = {}
    for segment in segments:
        speaker_id = str(segment.get("speaker_id") or "speaker")
        label = str(segment.get("speaker_label") or speaker_id)
        speakers.setdefault(
            speaker_id,
            {"id": speaker_id, "label_short": label, "label_full": label},
        )
    return list(speakers.values())


def evaluate_windows(
    segments: Sequence[Dict[str, Any]],
    windows: Sequence[AttributionWindow],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for window in windows:
        total = 0.0
        correct = 0.0
        seconds_by_speaker: Dict[str, float] = defaultdict(float)
        for segment in segments:
            start = float(segment.get("start") or 0)
            end = max(start, float(segment.get("end") or start))
            overlap = overlap_seconds(start, end, window.start, window.end)
            if overlap <= 0:
                continue
            label = str(segment.get("speaker_label") or segment.get("speaker_id") or "unknown")
            seconds_by_speaker[label] += overlap
            total += overlap
            if canonical_speaker_name(label) == canonical_speaker_name(window.expected):
                correct += overlap
        accuracy = correct / total if total else 0.0
        results.append(
            {
                "name": window.name,
                "start": window.start,
                "end": window.end,
                "expected": window.expected,
                "accuracy": round(accuracy, 4),
                "passed": accuracy >= ATTRIBUTION_WINDOW_PASS_ACCURACY,
                "speaker_seconds": {
                    label: round(seconds, 3)
                    for label, seconds in sorted(
                        seconds_by_speaker.items(), key=lambda item: item[1], reverse=True
                    )
                },
            }
        )
    return {
        "windows": results,
        "passed": sum(int(result["passed"]) for result in results),
        "total": len(results),
        "all_passed": bool(results) and all(result["passed"] for result in results),
    }


def evaluate_audit(
    segments: Sequence[Dict[str, Any]],
    audit_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(segments) != len(audit_rows):
        raise ValueError(
            f"Audit and segment counts differ ({len(audit_rows)} vs {len(segments)})"
        )
    total_seconds = 0.0
    correct_seconds = 0.0
    correct_segments = 0
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    for segment, row in zip(segments, audit_rows):
        expected = canonical_speaker_name(row.get("mapped_speaker_label"))
        predicted = canonical_speaker_name(
            segment.get("speaker_label") or segment.get("speaker_id")
        )
        start = float(row.get("start") or 0)
        end = max(start, float(row.get("end") or start))
        duration = max(0.001, end - start)
        total_seconds += duration
        confusion[expected][predicted] += duration
        if expected == predicted:
            correct_segments += 1
            correct_seconds += duration
    return {
        "segment_accuracy": round(correct_segments / len(audit_rows), 4) if audit_rows else 0.0,
        "duration_accuracy": round(correct_seconds / total_seconds, 4) if total_seconds else 0.0,
        "correct_segments": correct_segments,
        "total_segments": len(audit_rows),
        "confusion_seconds": {
            expected: {
                predicted: round(float(seconds), 3)
                for predicted, seconds in counter.most_common()
            }
            for expected, counter in sorted(confusion.items())
        },
    }


def evaluate_boundaries(
    boundaries: Sequence[str],
    audit_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(boundaries) != len(audit_rows):
        raise ValueError("Boundary and audit counts differ")
    truth = ["change"]
    for previous, current in zip(audit_rows, audit_rows[1:]):
        truth.append(
            "same"
            if canonical_speaker_name(previous.get("mapped_speaker_label"))
            == canonical_speaker_name(current.get("mapped_speaker_label"))
            else "change"
        )
    evaluated = [
        (predicted, expected)
        for predicted, expected in zip(boundaries[1:], truth[1:])
        if predicted != "uncertain"
    ]
    correct = sum(predicted == expected for predicted, expected in evaluated)
    true_changes = sum(expected == "change" for expected in truth[1:])
    predicted_changes = sum(predicted == "change" for predicted, _ in evaluated)
    correct_changes = sum(
        predicted == expected == "change" for predicted, expected in evaluated
    )
    return {
        "coverage": round(len(evaluated) / max(1, len(truth) - 1), 4),
        "confident_accuracy": round(correct / len(evaluated), 4) if evaluated else 0.0,
        "change_precision": round(correct_changes / predicted_changes, 4)
        if predicted_changes
        else 0.0,
        "change_recall": round(correct_changes / true_changes, 4) if true_changes else 0.0,
        "counts": dict(Counter(boundaries)),
    }


def evaluate_result(
    episode: Episode,
    segments: Sequence[Dict[str, Any]],
    boundaries: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "windows": evaluate_windows(segments, episode.windows) if episode.windows else None,
        "speaker_seconds": dict(
            sorted(
                {
                    label: round(seconds, 3)
                    for label, seconds in speaker_duration_totals(segments).items()
                }.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    }
    if episode.audit_path:
        audit_rows = read_json(episode.audit_path)
        metrics["audit"] = evaluate_audit(segments, audit_rows)
        external_windows = [
            window
            for window in episode.windows
            if canonical_speaker_name(window.expected)
            == canonical_speaker_name(SIL_UNKNOWN_LABEL)
        ]
        if external_windows:
            known_pairs = [
                (segment, row)
                for segment, row in zip(segments, audit_rows)
                if not any(
                    overlap_seconds(
                        float(row.get("start") or 0),
                        max(
                            float(row.get("start") or 0),
                            float(row.get("end") or row.get("start") or 0),
                        ),
                        window.start,
                        window.end,
                    )
                    > 0
                    for window in external_windows
                )
            ]
            metrics["known_host_audit"] = evaluate_audit(
                [segment for segment, _ in known_pairs],
                [row for _, row in known_pairs],
            )
        if boundaries is not None:
            metrics["boundaries"] = evaluate_boundaries(boundaries, audit_rows)
    return metrics


def speaker_duration_totals(
    segments: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for segment in segments:
        label = str(segment.get("speaker_label") or segment.get("speaker_id") or "unknown")
        start = float(segment.get("start") or 0)
        end = max(start, float(segment.get("end") or start))
        totals[label] += max(0.0, end - start)
    return totals


def slim_segments(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = (
        "segment_id",
        "chunk_index",
        "local_speaker",
        "start",
        "end",
        "text",
        "speaker_id",
        "speaker_label",
        "speaker_id_source",
        "cluster_id",
        "boundary_before",
    )
    return [{key: segment.get(key) for key in keep if key in segment} for segment in segments]


def episode_metadata(episode: Episode) -> Dict[str, Any]:
    tags = list(episode.tags) or [
        "All-In Podcast",
        "Jason Calacanis",
        "Chamath Palihapitiya",
        "David Sacks",
        "David Friedberg",
    ]
    return {
        "title": episode.title,
        "description": episode.context,
        "channelTitle": episode.channel_title,
        "tags": tags,
        "defaultLanguage": "en",
        "defaultAudioLanguage": "en",
    }


def load_cached_current_baseline(episode: Episode) -> Dict[str, Any]:
    if not episode.cached_baseline_filename:
        raise ValueError(f"No cached baseline configured for {episode.key}")
    segment_path = episode.cache_dir / episode.cached_baseline_filename
    mapping_path = episode.cache_dir / "speaker-mapping-effective.json"
    cached_segments = read_json(segment_path)
    mapping = read_json(mapping_path)
    labels = {
        str(speaker["id"]): str(
            speaker.get("label_short") or speaker.get("label_full") or speaker["id"]
        )
        for speaker in mapping.get("speakers", [])
        if speaker.get("id")
    }
    segments: List[Dict[str, Any]] = []
    for segment_id, segment in enumerate(cached_segments):
        copied = dict(segment)
        speaker_id = str(copied.get("speaker_id") or "speaker")
        copied.update(
            {
                "segment_id": segment_id,
                "speaker_label": labels.get(speaker_id, speaker_id),
            }
        )
        segments.append(copied)
    turns = turns_from_segments(segments)
    speakers = public_speakers_from_segments(segments)
    return {
        "episode_key": episode.key,
        "video_id": episode.video_id,
        "title": episode.title,
        "variant_id": "current-low",
        "variant_label": "Current approach (Luna low)",
        "method": "current-cached",
        "effort": "low",
        "speakers": speakers,
        "segments": slim_segments(segments),
        "turns": turns,
        "metrics": evaluate_result(episode, segments),
        "diagnostics": {
            "source": "cached-production-artifacts",
            "segments_path": str(segment_path),
            "mapping_path": str(mapping_path),
        },
    }


def run_current_baseline(
    episode: Episode,
    raw_segments: List[Dict[str, Any]],
    openai_key: str,
    log: Any,
) -> Dict[str, Any]:
    recorder = RequestRecorder("low")
    real_client = OpenAI(
        api_key=openai_key, timeout=EVALUATION_REQUEST_TIMEOUT_SECONDS
    )
    client = InstrumentedClient(real_client, recorder)
    metadata = episode_metadata(episode)
    url = f"https://www.youtube.com/watch?v={episode.video_id}"
    debug_sink: List[Dict[str, Any]] = []
    try:
        with recorder.phase("speaker_mapping"):
            mapping_model = ytranslate.assign_global_speakers_for_diarized_segments(
                client,
                MODEL,
                url,
                episode.title,
                episode.context,
                raw_segments,
                "en",
                debug_sink=debug_sink,
                metadata=metadata,
            )
            roster = ytranslate.infer_known_speaker_roster(metadata)
            evidence = ytranslate.build_speaker_identity_evidence(raw_segments, roster)
            mapping = ytranslate.apply_speaker_identity_evidence(mapping_model, evidence)
            overrides = ytranslate.load_speaker_mapping_overrides(episode.video_id)
            if overrides:
                mapping = ytranslate.apply_speaker_mapping_overrides(mapping, overrides)

        with recorder.phase("voice_reconciliation"):
            segments, voice_debug = ytranslate.reconcile_diarized_segments_with_voice(
                url, episode.video_id, raw_segments, mapping, log
            )

        with recorder.phase("turn_attribution"):
            segments, effective_speakers, role_debug = ytranslate.collapse_role_speaker_identities(
                segments, mapping.get("speakers", [])
            )
            labels = {
                str(speaker.get("id")): str(
                    speaker.get("label_short") or speaker.get("label_full") or speaker.get("id")
                )
                for speaker in effective_speakers
                if speaker.get("id")
            }
            for segment_id, segment in enumerate(segments):
                segment["segment_id"] = segment_id
                speaker_id = str(segment.get("speaker_id") or "speaker")
                segment["speaker_label"] = labels.get(speaker_id, speaker_id)
            turns = turns_from_segments(segments)
            contradictions = ytranslate.find_speaker_identity_contradictions(
                effective_speakers,
                [
                    {"speaker_id": turn["speaker_id"], "text_source": turn["text"]}
                    for turn in turns
                ],
            )
        return {
            "episode_key": episode.key,
            "video_id": episode.video_id,
            "title": episode.title,
            "variant_id": "current-low",
            "variant_label": "Current approach (Luna low)",
            "method": "current",
            "effort": "low",
            "speakers": effective_speakers,
            "segments": slim_segments(segments),
            "turns": turns,
            "metrics": evaluate_result(episode, segments),
            "diagnostics": {
                "mapping_model": mapping_model,
                "effective_mapping": mapping,
                "identity_evidence": ytranslate.serialize_speaker_identity_evidence(evidence),
                "voice": voice_debug,
                "role_merge": role_debug,
                "contradictions": contradictions,
                "usage": recorder.summary(),
            },
        }
    finally:
        real_client.close()


def run_turn_constrained_variant(
    episode: Episode,
    raw_segments: List[Dict[str, Any]],
    baseline_segments: List[Dict[str, Any]],
    effort: str,
    openai_key: str,
    output_dir: Path,
    audio: np.ndarray,
    sample_rate: int,
    encoder: VoiceEncoder,
    log: Any,
) -> Dict[str, Any]:
    recorder = RequestRecorder(effort)
    real_client = OpenAI(
        api_key=openai_key, timeout=EVALUATION_REQUEST_TIMEOUT_SECONDS
    )
    client = InstrumentedClient(real_client, recorder)
    try:
        with recorder.phase("boundary_detection"):
            boundaries = infer_boundaries(
                client,
                episode,
                raw_segments,
                output_dir / "boundary-batches" / effort,
                log,
            )
        with recorder.phase("acoustic_clustering"):
            speaker_references = resolve_speaker_references(
                episode, baseline_segments
            )
            units = build_embedding_units(raw_segments, boundaries)
            embeddings = embed_units(audio, sample_rate, units, encoder)
            raw_spectral_labels, spectral_labels = cluster_embeddings(
                embeddings,
                units,
                len({reference.name for reference in speaker_references}),
            )
            references = embed_references(
                episode, audio, sample_rate, encoder, speaker_references
            )
            centroids = cluster_centroids(embeddings, spectral_labels)
            _, spectral_naming_debug = map_clusters_to_references(centroids, references)
            labels, cluster_names, reference_debug = classify_anonymous_turns_by_reference(
                embeddings, units, references
            )
            reference_segments = assign_segments_from_units(
                raw_segments, units, labels, cluster_names, boundaries
            )
            segments, repair_debug = apply_conservative_turn_repairs(
                baseline_segments,
                reference_segments,
                reference_debug["assignments"],
            )
            segments = normalize_speaker_display_labels(segments, baseline_segments)
            turns = turns_from_segments(segments)
            speakers = public_speakers_from_segments(segments)
            contradictions = ytranslate.find_speaker_identity_contradictions(
                speakers,
                [
                    {"speaker_id": turn["speaker_id"], "text_source": turn["text"]}
                    for turn in turns
                ],
            )
        return {
            "episode_key": episode.key,
            "video_id": episode.video_id,
            "title": episode.title,
            "variant_id": f"turn-constrained-{effort}",
            "variant_label": f"Turn repair (Luna {effort})",
            "method": "turn-repair",
            "effort": effort,
            "speakers": speakers,
            "segments": slim_segments(segments),
            "turns": turns,
            "metrics": evaluate_result(episode, segments, boundaries),
            "diagnostics": {
                "boundary_counts": dict(Counter(boundaries)),
                "embedding_unit_count": len(units),
                "anonymous_turn_count": len(
                    set(int(unit["anonymous_turn_id"]) for unit in units)
                ),
                "raw_spectral_cluster_counts": dict(
                    Counter(int(label) for label in raw_spectral_labels)
                ),
                "spectral_cluster_counts": dict(
                    Counter(int(label) for label in spectral_labels)
                ),
                "spectral_cluster_naming": spectral_naming_debug,
                "speaker_references": [
                    {
                        "speaker": reference.name,
                        "start": reference.start,
                        "end": reference.end,
                        "source_video_id": reference.source_video_id or episode.video_id,
                    }
                    for reference in speaker_references
                ],
                "reference_classification": reference_debug,
                "repairs": repair_debug,
                "contradictions": contradictions,
                "usage": recorder.summary(),
            },
        }
    finally:
        real_client.close()


def run_sil_variant(
    episode: Episode,
    raw_segments: List[Dict[str, Any]],
    baseline_segments: List[Dict[str, Any]],
    effort: str,
    openai_key: str,
    output_dir: Path,
    audio: np.ndarray,
    sample_rate: int,
    encoder: VoiceEncoder,
    log: Any,
) -> Dict[str, Any]:
    recorder = RequestRecorder(effort)
    real_client = OpenAI(
        api_key=openai_key, timeout=EVALUATION_REQUEST_TIMEOUT_SECONDS
    )
    client = InstrumentedClient(real_client, recorder)
    try:
        with recorder.phase("boundary_detection"):
            boundaries = infer_boundaries(
                client,
                episode,
                raw_segments,
                output_dir / "boundary-batches" / effort,
                log,
            )
        with recorder.phase("speaker_identity_linking"):
            speaker_references = resolve_speaker_references(
                episode, baseline_segments
            )
            units = build_sil_units(raw_segments, boundaries)
            embeddings = embed_units(audio, sample_rate, units, encoder)
            references = embed_references(
                episode, audio, sample_rate, encoder, speaker_references
            )
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
            segments = normalize_speaker_display_labels(segments, baseline_segments)
            turns = turns_from_segments(segments)
            speakers = public_speakers_from_segments(segments)
            contradictions = ytranslate.find_speaker_identity_contradictions(
                speakers,
                [
                    {"speaker_id": turn["speaker_id"], "text_source": turn["text"]}
                    for turn in turns
                ],
            )
        return {
            "episode_key": episode.key,
            "video_id": episode.video_id,
            "title": episode.title,
            "variant_id": f"sil-{effort}",
            "variant_label": f"SIL (Luna {effort})",
            "method": "speaker-identity-linker",
            "effort": effort,
            "speakers": speakers,
            "segments": slim_segments(segments),
            "turns": turns,
            "metrics": evaluate_result(episode, segments, boundaries),
            "diagnostics": {
                "boundary_counts": dict(Counter(boundaries)),
                "embedding_unit_count": len(units),
                "speaker_references": [
                    {
                        "speaker": reference.name,
                        "start": reference.start,
                        "end": reference.end,
                        "source_video_id": reference.source_video_id or episode.video_id,
                    }
                    for reference in speaker_references
                ],
                "prototype_training": prototype_debug,
                "linker": linker_debug,
                "episode_scale_repairs": repair_debug,
                "contradictions": contradictions,
                "usage": recorder.summary(),
            },
        }
    finally:
        real_client.close()


def rebuild_as_conservative_repair(
    episode: Episode,
    raw_segments: List[Dict[str, Any]],
    baseline_segments: List[Dict[str, Any]],
    reference_variant: Dict[str, Any],
) -> Dict[str, Any]:
    reference_debug = reference_variant["diagnostics"]["reference_classification"]
    if reference_variant.get("method") == "turn-repair":
        reference_segments = reference_segments_from_assignments(
            raw_segments,
            reference_debug["assignments"],
            [
                str(segment.get("boundary_before") or "uncertain")
                for segment in reference_variant["segments"]
            ],
        )
    else:
        reference_segments = reference_variant["segments"]
    segments, repair_debug = apply_conservative_turn_repairs(
        baseline_segments,
        reference_segments,
        reference_debug["assignments"],
    )
    turns = turns_from_segments(segments)
    speakers = public_speakers_from_segments(segments)
    boundaries = [
        str(segment.get("boundary_before") or ("change" if index == 0 else "uncertain"))
        for index, segment in enumerate(reference_segments)
    ]
    diagnostics = dict(reference_variant.get("diagnostics", {}))
    diagnostics["reference_only_metrics"] = reference_variant.get("metrics", {})
    diagnostics["repairs"] = repair_debug
    diagnostics["contradictions"] = ytranslate.find_speaker_identity_contradictions(
        speakers,
        [
            {"speaker_id": turn["speaker_id"], "text_source": turn["text"]}
            for turn in turns
        ],
    )
    effort = str(reference_variant.get("effort") or "unknown")
    return {
        "episode_key": episode.key,
        "video_id": episode.video_id,
        "title": episode.title,
        "variant_id": f"turn-constrained-{effort}",
        "variant_label": f"Turn repair (Luna {effort})",
        "method": "turn-repair",
        "effort": effort,
        "speakers": speakers,
        "segments": slim_segments(segments),
        "turns": turns,
        "metrics": evaluate_result(episode, segments, boundaries),
        "diagnostics": diagnostics,
    }


def reference_segments_from_assignments(
    raw_segments: Sequence[Dict[str, Any]],
    assignments: Sequence[Dict[str, Any]],
    boundaries: Sequence[str],
) -> List[Dict[str, Any]]:
    ordered = sorted(assignments, key=lambda item: (float(item["start"]), float(item["end"])))
    resolved: List[Dict[str, Any]] = []
    for segment_id, segment in enumerate(raw_segments):
        start = float(segment.get("start") or 0)
        end = max(start, float(segment.get("end") or start))
        midpoint = (start + end) / 2
        assignment = next(
            (
                item
                for item in ordered
                if float(item["start"]) - 1e-6 <= midpoint <= float(item["end"]) + 1e-6
            ),
            None,
        )
        if assignment is None:
            assignment = min(
                ordered,
                key=lambda item: abs(
                    midpoint - (float(item["start"]) + float(item["end"])) / 2
                ),
            )
        label = str(assignment["speaker"])
        copied = dict(segment)
        copied.update(
            {
                "segment_id": segment_id,
                "speaker_id": ytranslate.speaker_id_from_label(label),
                "speaker_label": label,
                "speaker_id_source": "turn-reference-classification",
                "boundary_before": boundaries[segment_id],
            }
        )
        resolved.append(copied)
    return resolved


def duration_weighted_disagreement(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
) -> float:
    if len(left) != len(right):
        raise ValueError("Variant segment counts differ")
    total = 0.0
    disagreement = 0.0
    for left_segment, right_segment in zip(left, right):
        start = float(left_segment.get("start") or 0)
        end = max(start, float(left_segment.get("end") or start))
        duration = max(0.001, end - start)
        total += duration
        if canonical_speaker_name(
            left_segment.get("speaker_label") or left_segment.get("speaker_id")
        ) != canonical_speaker_name(
            right_segment.get("speaker_label") or right_segment.get("speaker_id")
        ):
            disagreement += duration
    return round(disagreement / total, 4) if total else 0.0


def output_variant_path(output_dir: Path, episode: Episode, variant_id: str) -> Path:
    return output_dir / "episodes" / episode.key / "variants" / f"{variant_id}.json"


def write_episode_audio_link(output_dir: Path, episode: Episode) -> Path:
    episode_dir = output_dir / "episodes" / episode.key
    episode_dir.mkdir(parents=True, exist_ok=True)
    source = episode.audio_path.resolve()
    link = episode_dir / f"audio{source.suffix.lower()}"
    if link.is_symlink() and link.resolve() == source:
        return link
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(source)
    return link


def compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key in {"audit", "known_host_audit", "boundaries", "windows"}
        and value is not None
    }


def build_manifest(
    output_dir: Path,
    episode_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    episodes = []
    for episode in EPISODES:
        variants = episode_results.get(episode.key, [])
        if not variants:
            continue
        audio_link = write_episode_audio_link(output_dir, episode)
        current = next(
            (variant for variant in variants if variant["variant_id"] == "current-low"),
            None,
        )
        variant_entries = []
        for variant in variants:
            entry = {
                "id": variant["variant_id"],
                "label": variant["variant_label"],
                "method": variant["method"],
                "effort": variant["effort"],
                "path": str(
                    output_variant_path(output_dir, episode, variant["variant_id"])
                    .relative_to(output_dir)
                ),
                "metrics": compact_metrics(variant.get("metrics", {})),
            }
            if current is not None and variant is not current:
                entry["disagreement_with_current"] = duration_weighted_disagreement(
                    current["segments"], variant["segments"]
                )
            variant_entries.append(entry)
        episodes.append(
            {
                "key": episode.key,
                "video_id": episode.video_id,
                "title": episode.title,
                "context": episode.context,
                "duration": round(
                    max(float(segment.get("end") or 0) for segment in variants[0]["segments"]),
                    3,
                ),
                "audio": str(audio_link.relative_to(output_dir)),
                "variants": variant_entries,
            }
        )
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": MODEL,
        "primary_comparison": ["current-low", "sil-low"],
        "episodes": episodes,
    }


def print_summary(manifest: Dict[str, Any]) -> None:
    print("\nExperiment summary", flush=True)
    for episode in manifest["episodes"]:
        print(f"\n{episode['key']}: {episode['title']}", flush=True)
        for variant in episode["variants"]:
            metrics = variant["metrics"]
            fields = []
            comparable_audit = metrics.get("known_host_audit") or metrics.get("audit")
            if comparable_audit:
                fields.append(
                    f"audit-duration={comparable_audit['duration_accuracy']:.1%}"
                )
            if metrics.get("windows"):
                fields.append(
                    f"windows={metrics['windows']['passed']}/{metrics['windows']['total']}"
                )
            if metrics.get("boundaries"):
                fields.append(
                    f"boundary={metrics['boundaries']['confident_accuracy']:.1%}"
                    f"@{metrics['boundaries']['coverage']:.1%}"
                )
            if "disagreement_with_current" in variant:
                fields.append(f"vs-current={variant['disagreement_with_current']:.1%}")
            print(f"  {variant['label']}: {', '.join(fields) or 'manual review'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate transcript-constrained acoustic diarization on cached All-In episodes."
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        choices=[episode.key for episode in EPISODES],
        default=[episode.key for episode in EPISODES],
    )
    parser.add_argument(
        "--efforts",
        nargs="+",
        choices=VALID_EFFORTS,
        default=list(DEFAULT_EFFORTS),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-current", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-sil", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ytranslate.load_project_env()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    selected = [episode for episode in EPISODES if episode.key in set(args.episodes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder: Optional[VoiceEncoder] = None
    episode_results: Dict[str, List[Dict[str, Any]]] = {}

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    for episode in selected:
        log(f"Episode {episode.key}: {episode.title}")
        if not episode.asr_path.exists():
            raise FileNotFoundError(episode.asr_path)
        asr_result = read_json(episode.asr_path)
        raw_segments = [dict(segment) for segment in asr_result.get("segments", [])]
        if not raw_segments:
            raise RuntimeError(f"No ASR segments in {episode.asr_path}")
        results: List[Dict[str, Any]] = []

        current_path = output_variant_path(args.output_dir, episode, "current-low")
        current: Optional[Dict[str, Any]] = None
        if not args.skip_current:
            if current_path.exists() and not args.force:
                log("Loading cached current-low baseline")
                current = read_json(current_path)
                current["metrics"] = evaluate_result(episode, current["segments"])
                write_json(current_path, current)
            elif episode.cached_baseline_filename:
                log("Loading current-low baseline from cached production artifacts")
                current = load_cached_current_baseline(episode)
                write_json(current_path, current)
            else:
                log("Running current production attribution with Luna low")
                current = run_current_baseline(
                    episode, copy.deepcopy(raw_segments), openai_key, log
                )
                write_json(current_path, current)
        elif current_path.exists():
            current = read_json(current_path)
            current["metrics"] = evaluate_result(episode, current["segments"])
            write_json(current_path, current)
        else:
            raise RuntimeError(
                f"--skip-current requires an existing baseline at {current_path}"
            )
        results.append(current)

        audio: Optional[np.ndarray] = None
        sample_rate = 0
        for effort in args.efforts:
            variant_id = f"sil-{effort}"
            variant_path = output_variant_path(args.output_dir, episode, variant_id)
            if variant_path.exists() and not (args.force or args.force_sil):
                log(f"Loading cached {variant_id}")
                variant = read_json(variant_path)
                variant["segments"] = slim_segments(
                    normalize_speaker_display_labels(
                        variant["segments"], current["segments"]
                    )
                )
                variant["turns"] = turns_from_segments(variant["segments"])
                variant["speakers"] = public_speakers_from_segments(variant["segments"])
                variant["metrics"] = evaluate_result(episode, variant["segments"])
                write_json(variant_path, variant)
            else:
                if encoder is None:
                    log("Loading voice encoder")
                    encoder = VoiceEncoder()
                if audio is None:
                    audio, sample_rate = load_episode_audio(episode)
                log(f"Running SIL attribution with Luna {effort}")
                variant = run_sil_variant(
                    episode,
                    copy.deepcopy(raw_segments),
                    copy.deepcopy(current["segments"]),
                    effort,
                    openai_key,
                    args.output_dir / "episodes" / episode.key,
                    audio,
                    sample_rate,
                    encoder,
                    log,
                )
                write_json(variant_path, variant)
            results.append(variant)
        episode_results[episode.key] = results

    manifest = build_manifest(args.output_dir, episode_results)
    write_json(args.output_dir / "manifest.json", manifest)
    print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
