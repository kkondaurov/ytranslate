#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import shutil
import sys
import subprocess
import time
import unicodedata
import inspect
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
from openai import OpenAI
import openai
from docx import Document
from docx.shared import Pt
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled


YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_TARGET_LANGUAGE = "Russian"
OPENAI_TIMEOUT_SECONDS = 1800
OPENAI_TEMPERATURE = 0.2
OPENAI_CLEANUP_TEMPERATURE = 0.0
OPENAI_ANNOTATION_TEMPERATURE = 0.0
DOCX_FONT_NAME = "Arial"
DOCX_FONT_SIZE = Pt(13)
DOCX_HEADING_FONT_SIZE = Pt(16)
OUTPUT_DIR = os.path.expanduser("~/Downloads")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate a YouTube video's transcript into a target language "
            "and structure it as a conversation."
        )
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "target_language",
        nargs="?",
        help="Target language (defaults to DEFAULT_TARGET_LANGUAGE or Russian)",
    )
    parser.add_argument(
        "--docx-test",
        action="store_true",
        help="Generate a sample DOCX without calling external APIs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write per-stage debug artifacts as Markdown/JSON instead of DOCX/PDF",
    )
    return parser.parse_args()

def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"").strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


def load_project_env() -> None:
    load_dotenv(ENV_PATH)


def resolve_target_language(target_language: Optional[str]) -> str:
    value = (target_language or os.getenv("DEFAULT_TARGET_LANGUAGE") or DEFAULT_TARGET_LANGUAGE).strip()
    return value or DEFAULT_TARGET_LANGUAGE


def extract_video_id(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path
    query = parse_qs(parsed.query)

    vid = query.get("v", [None])[0]
    if vid:
        return vid

    if host.endswith("youtu.be"):
        vid = path.strip("/").split("/")[0]
        return vid or None

    if "youtube.com" in host:
        if path.startswith("/watch"):
            return query.get("v", [None])[0]
        if path.startswith("/shorts/"):
            parts = path.split("/")
            return parts[2] if len(parts) > 2 else None
        if path.startswith("/embed/"):
            parts = path.split("/")
            return parts[2] if len(parts) > 2 else None
        if path.startswith("/live/"):
            parts = path.split("/")
            return parts[2] if len(parts) > 2 else None

    return None


def canonicalize_youtube_url(url: str) -> Optional[str]:
    video_id = extract_video_id(url)
    if not video_id:
        return None
    return f"https://youtu.be/{video_id}"


def fetch_video_metadata(video_id: str, api_key: str) -> Dict[str, Any]:
    params = {
        "part": "snippet",
        "id": video_id,
        "key": api_key,
    }
    resp = requests.get(YOUTUBE_API_URL, params=params, timeout=30)
    if resp.status_code != 200:
        msg = ""
        try:
            data = resp.json()
            msg = data.get("error", {}).get("message", "")
        except Exception:
            msg = resp.text
        raise RuntimeError(
            f"YouTube API error ({resp.status_code}): {msg or 'Unknown error'}"
        )

    data = resp.json()
    items = data.get("items", [])
    if not items:
        raise RuntimeError("No video metadata found (invalid video ID?)")

    snippet = items[0].get("snippet", {})
    return {
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "defaultLanguage": snippet.get("defaultLanguage"),
        "defaultAudioLanguage": snippet.get("defaultAudioLanguage"),
    }


def pick_transcript(transcripts: List[Any], preferred_langs: List[str]) -> Any:
    def find_match(predicate):
        for t in transcripts:
            if predicate(t):
                return t
        return None

    for lang in preferred_langs:
        if not lang:
            continue
        match = find_match(
            lambda t, lang=lang: (not t.is_generated) and t.language_code == lang
        )
        if match:
            return match

    manual = find_match(lambda t: not t.is_generated)
    if manual:
        return manual

    for lang in preferred_langs:
        if not lang:
            continue
        match = find_match(lambda t, lang=lang: t.language_code == lang)
        if match:
            return match

    return transcripts[0] if transcripts else None


def list_transcripts(video_id: str):
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        return YouTubeTranscriptApi.list_transcripts(video_id)

    api = YouTubeTranscriptApi()
    if hasattr(api, "list"):
        return api.list(video_id)
    if hasattr(api, "list_transcripts"):
        return api.list_transcripts(video_id)

    raise RuntimeError("Unsupported youtube-transcript-api version")


def fetch_transcript(video_id: str, preferred_langs: List[str]) -> Dict[str, Any]:
    transcript_list = list_transcripts(video_id)
    transcripts = list(transcript_list)
    if not transcripts:
        raise NoTranscriptFound(video_id)

    transcript = pick_transcript(transcripts, preferred_langs)
    if not transcript:
        raise NoTranscriptFound(video_id)

    fetched = transcript.fetch()
    if hasattr(fetched, "to_raw_data"):
        segments = fetched.to_raw_data()
    else:
        segments = fetched

    return {
        "language_code": transcript.language_code,
        "language": transcript.language,
        "is_generated": transcript.is_generated,
        "segments": segments,
    }


def clean_segment_text(text: str) -> str:
    text = (text or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    last_text = None
    for seg in segments:
        if isinstance(seg, dict):
            text = clean_segment_text(seg.get("text", ""))
            start = seg.get("start")
            duration = seg.get("duration")
        else:
            text = clean_segment_text(getattr(seg, "text", ""))
            start = getattr(seg, "start", None)
            duration = getattr(seg, "duration", None)
        if not text:
            continue
        if text == last_text:
            continue
        normalized.append({
            "start": start,
            "duration": duration,
            "text": text,
        })
        last_text = text
    return normalized


def format_timecode(seconds: Optional[float]) -> str:
    if seconds is None:
        return "??:??:??"
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_segments(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, seg in enumerate(segments, 1):
        if seg.get("start") is not None:
            lines.append(f"[{idx} @ {format_timecode(seg.get('start'))}] {seg['text']}")
        else:
            lines.append(f"[{idx}] {seg['text']}")
    return "\n".join(lines)


def write_json_file(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_debug_output_dir(video_id: str, title: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_title = sanitize_filename(title)[:60]
    folder_name = f"ytranslate-debug-{video_id}-{timestamp}"
    if safe_title:
        folder_name += f"-{safe_title}"
    output_dir = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def render_markdown_transcript(
    title_translated: str,
    speakers: List[Dict[str, str]],
    turns: List[Dict[str, str]],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    lines: List[str] = [f"# {title_translated.strip() or 'Translated Transcript'}", ""]

    if metadata:
        lines.append("## Run Info")
        lines.append("")
        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    if speakers:
        lines.append("## Speakers")
        lines.append("")
        for speaker in speakers:
            label_full = speaker.get("label_full") or speaker.get("label_short") or speaker.get("id", "Speaker")
            label_short = speaker.get("label_short") or speaker.get("id", "Speaker")
            lines.append(f"- **{label_short}**: {label_full}")
        lines.append("")

    lines.append("## Transcript")
    lines.append("")
    speaker_labels = {
        speaker.get("id"): speaker.get("label_short") or speaker.get("id") or "Speaker"
        for speaker in speakers
    }
    for turn in turns:
        speaker_id = turn.get("speaker_id")
        label = speaker_labels.get(speaker_id) or speaker_id or "Speaker"
        text = (turn.get("text_translated") or "").strip()
        if not text:
            continue
        lines.append(f"**{label}:** {text}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_system_prompt() -> str:
    return (
        "You are a professional transcript translator and dialogue editor. "
        "Translate the entire transcript into the target language. "
        "Do not summarize or omit any spoken content. "
        "Do not add information that is not present in the source. "
        "Translate the video title into the target language. "
        "Use the video title and description to infer speaker identities/roles. "
        "If a speaker is uncertain, use consistent labels like 'Speaker 1', 'Speaker 2'. "
        "Do not split a single sentence, clause, or thought across multiple speakers. "
        "Do not assign a new speaker to a dangling fragment that clearly continues the previous sentence. "
        "Prefer fewer speaker switches over speculative speaker changes. "
        "Only switch speakers when the transcript clearly indicates a turn change. "
        "Merge adjacent text into coherent turns rather than producing many tiny fragments. "
        "Produce idiomatic, natural-sounding language. You may lightly paraphrase for fluency, "
        "but stay as close to the original as possible. "
        "Preserve the tone of the conversation (serious, friendly, banter, etc.). "
        "Translate business, financial, and technical jargon into natural target-language phrasing. "
        "Do not leave obvious English terms untranslated unless they are standard in the target language. "
        "Do not invent pseudo-translations, fake calques, or unnatural Russified forms. "
        "If a natural target-language rendering exists, use it directly instead of preserving the English. "
        "Translate inline explanations and side remarks fully, including quoted definitions by other speakers. "
        "Use concise speaker labels (e.g., first name or role like 'Host', 'Guest'). "
        "Provide full speaker names and titles for the speaker list, but keep per-turn labels short. "
        "Do not repeat full titles in every turn; keep labels short and consistent. "
        "Remove filler words like 'um', 'uh', and stutters while preserving meaning. "
        "If the transcript includes non-speech cues (e.g., [music]), omit those. "
        "Keep the original order of the conversation. "
        "Return only JSON that matches the provided schema."
    )


def build_target_terminology_guidance(target_language: str) -> str:
    target_norm = (target_language or "").strip().lower()
    if "russian" in target_norm or "рус" in target_norm:
        return (
            "Terminology requirements for Russian:\n"
            "- Use natural Russian equivalents where possible instead of transliterated English.\n"
            "- Never invent fake Russian-looking words such as bad calques or pseudo-technical slang.\n"
            "- Prefer fully translated phrases like 'рабочее пространство Slack' over mixed English-Russian forms.\n"
            "- Do not add bracketed explanations in this pass unless the source itself includes an explanation.\n"
        )
    return (
        "Terminology requirements:\n"
        "- Do not leave specialized English phrases untranslated in the target language.\n"
        "- If keeping acronyms, translate their expanded meaning on first mention.\n"
        "- If there is no exact equivalent, use a concise explanatory bracket.\n"
    )


def build_user_prompt(
    url: str,
    title: str,
    description: str,
    target_language: str,
    transcript_text: str,
    known_speakers: Optional[List[Dict[str, str]]],
    source_language_hint: Optional[str],
) -> str:
    speaker_block = ""
    if known_speakers:
        speaker_lines = []
        for speaker in known_speakers:
            label_short = speaker.get("label_short") or speaker.get("id")
            label_full = speaker.get("label_full")
            if label_full:
                speaker_lines.append(f"- {speaker.get('id')}: {label_short} ({label_full})")
            else:
                speaker_lines.append(f"- {speaker.get('id')}: {label_short}")
        speaker_block = "Known speakers (reuse these IDs if they match):\n" + "\n".join(speaker_lines)

    source_hint = f"Source language hint: {source_language_hint}\n" if source_language_hint else ""
    terminology_guidance = build_target_terminology_guidance(target_language)

    return (
        f"Video URL: {url}\n"
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Target language: {target_language}\n"
        f"{source_hint}"
        f"{speaker_block}\n"
        f"{terminology_guidance}\n"
        "Transcript (ordered lines):\n"
        f"{transcript_text}"
    )


def get_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title_translated": {"type": "string"},
            "speakers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "label_short": {"type": "string"},
                        "label_full": {"type": "string"},
                    },
                    "required": ["id", "label_short", "label_full"],
                },
            },
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "speaker_id": {"type": "string"},
                        "text_translated": {"type": "string"},
                    },
                    "required": ["speaker_id", "text_translated"],
                },
            },
        },
        "required": ["title_translated", "speakers", "turns"],
    }


def get_speaker_attribution_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "speakers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "label_short": {"type": "string"},
                        "label_full": {"type": "string"},
                    },
                    "required": ["id", "label_short", "label_full"],
                },
            },
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "speaker_id": {"type": "string"},
                        "text_source": {"type": "string"},
                    },
                    "required": ["speaker_id", "text_source"],
                },
            },
        },
        "required": ["speakers", "turns"],
    }


def get_turn_translation_schema(turn_count: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title_translated": {"type": "string"},
            "turns": {
                "type": "array",
                "minItems": turn_count,
                "maxItems": turn_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text_translated": {"type": "string"},
                    },
                    "required": ["text_translated"],
                },
            },
        },
        "required": ["title_translated", "turns"],
    }


def get_turn_cleanup_schema(turn_count: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "turns": {
                "type": "array",
                "minItems": turn_count,
                "maxItems": turn_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text_translated": {"type": "string"},
                    },
                    "required": ["text_translated"],
                },
            }
        },
        "required": ["turns"],
    }


def build_speaker_attribution_system_prompt() -> str:
    return (
        "You are a professional transcript dialogue editor. "
        "Your task is to reconstruct coherent dialogue turns and infer likely speaker identities from an ordered transcript. "
        "Do not translate. Keep the wording in the source language. "
        "Use the video title, description, and transcript content to infer who is speaking. "
        "If known speakers are provided, reuse those IDs and labels whenever they still fit. "
        "If a speaker is uncertain, use consistent labels like 'speaker_1', 'speaker_2'. "
        "Do not split a single sentence, clause, or thought across multiple speakers. "
        "Do not assign a new speaker to a dangling fragment that clearly continues the previous sentence. "
        "Prefer fewer speaker switches over speculative speaker changes. "
        "Only switch speakers when the transcript clearly indicates a turn change. "
        "Merge adjacent lines into coherent turns instead of many tiny fragments. "
        "Remove filler words like 'um', 'uh', and obvious stutters while preserving meaning. "
        "If the transcript includes non-speech cues like [music], omit them. "
        "Use concise per-turn labels such as a first name or role like Host or Guest. "
        "Provide fuller names or roles in the speakers list when they can be inferred reliably. "
        "Keep the original order of the conversation. "
        "Return only JSON that matches the provided schema."
    )


def build_speaker_attribution_user_prompt(
    url: str,
    title: str,
    description: str,
    transcript_text: str,
    known_speakers: Optional[List[Dict[str, str]]],
    source_language_hint: Optional[str],
) -> str:
    speaker_block = ""
    if known_speakers:
        speaker_lines = []
        for speaker in known_speakers:
            label_short = speaker.get("label_short") or speaker.get("id")
            label_full = speaker.get("label_full")
            if label_full:
                speaker_lines.append(f"- {speaker.get('id')}: {label_short} ({label_full})")
            else:
                speaker_lines.append(f"- {speaker.get('id')}: {label_short}")
        speaker_block = "Known speakers (reuse these IDs if they match):\n" + "\n".join(speaker_lines)

    source_hint = f"Source language hint: {source_language_hint}\n" if source_language_hint else ""
    return (
        f"Video URL: {url}\n"
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"{source_hint}"
        f"{speaker_block}\n"
        "Transcript (ordered lines with approximate timestamps):\n"
        f"{transcript_text}"
    )


def format_source_turns(turns: List[Dict[str, str]]) -> str:
    lines = []
    for idx, turn in enumerate(turns, 1):
        speaker_id = turn.get("speaker_id") or "speaker"
        text = (turn.get("text_source") or "").strip()
        lines.append(f"[{idx}] {speaker_id}: {text}")
    return "\n".join(lines)


def build_turn_translation_system_prompt(target_language: str) -> str:
    return (
        "You are a professional transcript translator. "
        "Translate already attributed dialogue turns into the target language. "
        "Preserve the meaning, tone, and order of the conversation. "
        "Do not change speaker assignment. Do not merge turns. Do not split turns. "
        "Do not add information that is not present in the source. "
        "Produce idiomatic, natural-sounding language. "
        "Translate business, financial, and technical jargon into natural target-language phrasing. "
        "Do not leave obvious English terms untranslated unless they are standard in the target language. "
        "Do not invent pseudo-translations, fake calques, or unnatural target-language forms. "
        "If a natural target-language rendering exists, use it directly instead of preserving the English. "
        "Translate the video title into the target language. "
        "Return only JSON that matches the provided schema."
    )


def build_turn_translation_user_prompt(
    url: str,
    title: str,
    description: str,
    target_language: str,
    speakers: List[Dict[str, str]],
    turns: List[Dict[str, str]],
    source_language_hint: Optional[str],
) -> str:
    speaker_lines = []
    for speaker in speakers:
        label_short = speaker.get("label_short") or speaker.get("id")
        label_full = speaker.get("label_full")
        if label_full:
            speaker_lines.append(f"- {speaker.get('id')}: {label_short} ({label_full})")
        else:
            speaker_lines.append(f"- {speaker.get('id')}: {label_short}")
    source_hint = f"Source language hint: {source_language_hint}\n" if source_language_hint else ""
    terminology_guidance = build_target_terminology_guidance(target_language)
    return (
        f"Video URL: {url}\n"
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Target language: {target_language}\n"
        f"{source_hint}"
        "Speakers (keep these IDs exactly):\n"
        + "\n".join(speaker_lines)
        + "\n"
        f"{terminology_guidance}\n"
        "Turns to translate (keep order; return one translated text per input turn):\n"
        f"{format_source_turns(turns)}"
    )


def extract_response_text(response: Any) -> Optional[str]:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    output = getattr(response, "output", None)
    if not output:
        return None

    texts = []
    for item in output:
        content = getattr(item, "content", None)
        if not content:
            continue
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text:
                texts.append(text)
    return "".join(texts) if texts else None


def is_context_length_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(
        phrase in msg
        for phrase in [
            "context length",
            "maximum context",
            "too many tokens",
            "token limit",
        ]
    )

def is_request_too_large_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "request too large" in msg
        or "tokens per min" in msg
        or "rate_limit_exceeded" in msg and "tokens" in msg
    )

def call_openai(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str = "translated_transcript",
    schema: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    schema = schema or get_schema()
    response_format_v1 = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }
    response_format_v2 = {
        "type": "json_schema",
        "name": schema_name,
        "strict": True,
        "schema": schema,
    }

    response_kwargs = {
        "model": model,
        "temperature": OPENAI_TEMPERATURE if temperature is None else temperature,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    params = inspect.signature(client.responses.create).parameters
    if "response_format" in params:
        response_kwargs["response_format"] = response_format_v1
    else:
        response_kwargs["text"] = {"format": response_format_v2}

    response = client.responses.create(**response_kwargs)

    text = extract_response_text(response)
    if not text:
        raise RuntimeError("OpenAI response contained no text")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI response was not valid JSON") from exc


def call_openai_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5,
    schema_name: str = "translated_transcript",
    schema: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return call_openai(
                client,
                model,
                system_prompt,
                user_prompt,
                schema_name=schema_name,
                schema=schema,
                temperature=temperature,
            )
        except Exception as exc:
            is_rate_limit = isinstance(exc, getattr(openai, "RateLimitError", ()))
            if is_rate_limit and is_request_too_large_error(exc):
                raise
            is_timeout = isinstance(exc, getattr(openai, "APITimeoutError", ()))
            is_connection = isinstance(exc, getattr(openai, "APIConnectionError", ()))
            if is_rate_limit or is_timeout or is_connection:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay + random.random())
                delay *= 2
                continue
            raise


def translate_single_pass(
    client: OpenAI,
    model: str,
    url: str,
    title: str,
    description: str,
    target_language: str,
    segments: List[Dict[str, Any]],
    source_language_hint: Optional[str],
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    transcript_text = format_segments(segments)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        url,
        title,
        description,
        target_language,
        transcript_text,
        None,
        source_language_hint,
    )
    result = call_openai_with_retry(client, model, system_prompt, user_prompt)
    if debug_sink is not None:
        debug_sink.append(
            {
                "mode": "single_pass",
                "segment_count": len(segments),
                "result": result,
            }
        )
    return result


def chunk_segments(segments: List[Dict[str, Any]], max_chars: int) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0
    for seg in segments:
        line_len = len(seg["text"]) + 6
        if current and current_chars + line_len > max_chars:
            chunks.append(current)
            current = [seg]
            current_chars = line_len
        else:
            current.append(seg)
            current_chars += line_len
    if current:
        chunks.append(current)
    return chunks


def merge_speakers(
    existing: List[Dict[str, str]],
    incoming: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    by_id = {s["id"]: s for s in existing}
    for speaker in incoming:
        existing_speaker = by_id.get(speaker["id"])
        if not existing_speaker:
            by_id[speaker["id"]] = speaker
            continue
        if not existing_speaker.get("label_full") and speaker.get("label_full"):
            existing_speaker["label_full"] = speaker["label_full"]
        if not existing_speaker.get("label_short") and speaker.get("label_short"):
            existing_speaker["label_short"] = speaker["label_short"]
    return list(by_id.values())


def attribute_speakers_single_pass(
    client: OpenAI,
    model: str,
    url: str,
    title: str,
    description: str,
    segments: List[Dict[str, Any]],
    source_language_hint: Optional[str],
    known_speakers: Optional[List[Dict[str, str]]] = None,
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    transcript_text = format_segments(segments)
    result = call_openai_with_retry(
        client,
        model,
        build_speaker_attribution_system_prompt(),
        build_speaker_attribution_user_prompt(
            url,
            title,
            description,
            transcript_text,
            known_speakers,
            source_language_hint,
        ),
        schema_name="speaker_attribution",
        schema=get_speaker_attribution_schema(),
        temperature=0.0,
    )
    if debug_sink is not None:
        debug_sink.append(
            {
                "mode": "single_pass",
                "segment_count": len(segments),
                "result": result,
            }
        )
    return result


def attribute_speakers_with_chunking(
    client: OpenAI,
    model: str,
    url: str,
    title: str,
    description: str,
    segments: List[Dict[str, Any]],
    source_language_hint: Optional[str],
    max_chars: int,
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    chunks = chunk_segments(segments, max_chars=max_chars)
    speakers: List[Dict[str, str]] = []
    turns: List[Dict[str, str]] = []

    for chunk_index, chunk in enumerate(chunks, 1):
        result = call_openai_with_retry(
            client,
            model,
            build_speaker_attribution_system_prompt(),
            build_speaker_attribution_user_prompt(
                url,
                title,
                description,
                format_segments(chunk),
                speakers or None,
                source_language_hint,
            ),
            schema_name="speaker_attribution",
            schema=get_speaker_attribution_schema(),
            temperature=0.0,
        )
        if debug_sink is not None:
            debug_sink.append(
                {
                    "mode": "chunked",
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "segment_count": len(chunk),
                    "max_chars": max_chars,
                    "result": result,
                }
            )
        speakers = merge_speakers(speakers, result.get("speakers", []))
        turns.extend(result.get("turns", []))

    return {
        "speakers": speakers,
        "turns": turns,
    }


def attribute_speakers(
    client: OpenAI,
    model: str,
    url: str,
    title: str,
    description: str,
    segments: List[Dict[str, Any]],
    source_language_hint: Optional[str],
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    try:
        return attribute_speakers_single_pass(
            client,
            model,
            url,
            title,
            description,
            segments,
            source_language_hint,
            debug_sink=debug_sink,
        )
    except Exception as exc:
        if not (is_context_length_error(exc) or is_request_too_large_error(exc)):
            raise

    for max_chars in (80_000, 50_000, 35_000, 25_000):
        try:
            return attribute_speakers_with_chunking(
                client,
                model,
                url,
                title,
                description,
                segments,
                source_language_hint,
                max_chars=max_chars,
                debug_sink=debug_sink,
            )
        except Exception as exc:
            if is_context_length_error(exc):
                continue
            raise

    raise RuntimeError("Unable to attribute speakers within context limits")


def chunk_source_turns(turns: List[Dict[str, str]], max_chars: int) -> List[List[Dict[str, str]]]:
    chunks: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    current_chars = 0
    for turn in turns:
        line_len = len(turn.get("text_source", "")) + len(turn.get("speaker_id", "")) + 8
        if current and current_chars + line_len > max_chars:
            chunks.append(current)
            current = [turn]
            current_chars = line_len
        else:
            current.append(turn)
            current_chars += line_len
    if current:
        chunks.append(current)
    return chunks


def translate_turn_chunk(
    client: OpenAI,
    model: str,
    url: str,
    title: str,
    description: str,
    target_language: str,
    speakers: List[Dict[str, str]],
    turns: List[Dict[str, str]],
    source_language_hint: Optional[str],
    debug_sink: Optional[List[Dict[str, Any]]] = None,
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> Dict[str, Any]:
    schema = get_turn_translation_schema(len(turns))
    result = call_openai_with_retry(
        client,
        model,
        build_turn_translation_system_prompt(target_language),
        build_turn_translation_user_prompt(
            url,
            title,
            description,
            target_language,
            speakers,
            turns,
            source_language_hint,
        ),
        schema_name="translated_turn_chunk",
        schema=schema,
        temperature=OPENAI_TEMPERATURE,
    )
    if debug_sink is not None:
        debug_sink.append(
            {
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "turn_count": len(turns),
                "result": result,
            }
        )
    return result


def translate_attributed_turns(
    client: OpenAI,
    model: str,
    url: str,
    title: str,
    description: str,
    target_language: str,
    speakers: List[Dict[str, str]],
    turns: List[Dict[str, str]],
    source_language_hint: Optional[str],
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    try:
        result = translate_turn_chunk(
            client,
            model,
            url,
            title,
            description,
            target_language,
            speakers,
            turns,
            source_language_hint,
            debug_sink=debug_sink,
        )
        translated_turns = result.get("turns", [])
        if len(translated_turns) != len(turns):
            raise RuntimeError("Translation pass returned the wrong number of turns")
        return {
            "title_translated": result.get("title_translated", ""),
            "speakers": speakers,
            "turns": [
                {
                    "speaker_id": turn.get("speaker_id"),
                    "text_translated": (translated.get("text_translated") or "").strip(),
                }
                for turn, translated in zip(turns, translated_turns)
            ],
        }
    except Exception as exc:
        if not (is_context_length_error(exc) or is_request_too_large_error(exc)):
            raise

    chunks = chunk_source_turns(turns, max_chars=60_000)
    translated_turns: List[Dict[str, str]] = []
    title_translated = ""
    for chunk_index, chunk in enumerate(chunks, 1):
        result = translate_turn_chunk(
            client,
            model,
            url,
            title,
            description,
            target_language,
            speakers,
            chunk,
            source_language_hint,
            debug_sink=debug_sink,
            chunk_index=chunk_index,
            chunk_count=len(chunks),
        )
        if not title_translated:
            title_translated = result.get("title_translated", "")
        translated_items = result.get("turns", [])
        if len(translated_items) != len(chunk):
            raise RuntimeError("Translation pass returned the wrong number of turns")
        translated_turns.extend(
            {
                "speaker_id": turn.get("speaker_id"),
                "text_translated": (translated.get("text_translated") or "").strip(),
            }
            for turn, translated in zip(chunk, translated_items)
        )

    return {
        "title_translated": title_translated,
        "speakers": speakers,
        "turns": translated_turns,
    }


def translate_with_chunking(
    client: OpenAI,
    model: str,
    url: str,
    title: str,
    description: str,
    target_language: str,
    segments: List[Dict[str, Any]],
    source_language_hint: Optional[str],
    max_chars: int,
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt()
    chunks = chunk_segments(segments, max_chars=max_chars)
    speakers: List[Dict[str, str]] = []
    turns: List[Dict[str, str]] = []
    title_translated: Optional[str] = None

    for chunk_index, chunk in enumerate(chunks, 1):
        transcript_text = format_segments(chunk)
        user_prompt = build_user_prompt(
            url,
            title,
            description,
            target_language,
            transcript_text,
            speakers or None,
            source_language_hint,
        )
        result = call_openai_with_retry(client, model, system_prompt, user_prompt)
        if debug_sink is not None:
            debug_sink.append(
                {
                    "mode": "chunked",
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "segment_count": len(chunk),
                    "max_chars": max_chars,
                    "result": result,
                }
            )
        if not title_translated:
            title_translated = result.get("title_translated")
        speakers = merge_speakers(speakers, result.get("speakers", []))
        turns.extend(result.get("turns", []))

    return {
        "title_translated": title_translated or "",
        "speakers": speakers,
        "turns": turns,
    }


def translate_transcript(
    client: OpenAI,
    model: str,
    url: str,
    title: str,
    description: str,
    target_language: str,
    segments: List[Dict[str, Any]],
    source_language_hint: Optional[str],
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    try:
        return translate_single_pass(
            client,
            model,
            url,
            title,
            description,
            target_language,
            segments,
            source_language_hint,
            debug_sink=debug_sink,
        )
    except Exception as exc:
        if not (is_context_length_error(exc) or is_request_too_large_error(exc)):
            raise

    for max_chars in (80_000, 50_000, 35_000, 25_000):
        try:
            return translate_with_chunking(
                client,
                model,
                url,
                title,
                description,
                target_language,
                segments,
                source_language_hint,
                max_chars=max_chars,
                debug_sink=debug_sink,
            )
        except Exception as exc:
            if is_context_length_error(exc):
                continue
            raise

    raise RuntimeError("Unable to translate transcript within context limits")


def is_russian_target_language(target_language: Optional[str]) -> bool:
    target_norm = (target_language or "").strip().lower()
    return "russian" in target_norm or "рус" in target_norm


def format_turns_for_cleanup(turns: List[Dict[str, str]]) -> str:
    lines = []
    for idx, turn in enumerate(turns, 1):
        text = (turn.get("text_translated") or "").strip()
        lines.append(f"[{idx}] {text}")
    return "\n".join(lines)


def build_russian_cleanup_system_prompt() -> str:
    return (
        "You are a Russian-language copy editor cleaning up an already translated podcast transcript. "
        "Preserve the meaning, tone, and turn boundaries exactly, but improve wording so it reads like natural, competent Russian. "
        "Do not change speaker order, do not merge or split turns, and do not change which speaker says which turn. "
        "Your job is to fix awkward phrasing, mixed-language artifacts, bad grammar, bad glossary glosses, and clumsy brackets. "
        "Assume the reader is intelligent but not an expert in the domain. "
        "Keep helpful bracketed explanations for non-expert readers, but remove or rewrite bad ones. "
        "Add a short bracketed explanation on first mention only when a specialist term, acronym, internal model name, or industry phrase would otherwise be unclear to a non-expert reader. "
        "Do not add brackets for obvious brands, ordinary product names, or phrases that already read naturally in Russian. "
        "Do not invent fake Russian words or calques such as unnatural pseudo-technical slang. "
        "Do not leave ordinary English words or phrases inside brackets if they can be translated naturally into Russian. "
        "Do not add redundant glosses for obvious brand names or product names such as Ferrari, Gmail, or Google Calendar when the Russian phrase already reads naturally. "
        "Prefer a fully natural Russian phrase over mixed forms like 'Slack workspace [рабочее пространство Slack]'. "
        "If a bracketed gloss is useful, make it short, idiomatic, and genuinely informative. "
        "If a term's exact meaning is unclear, do not hallucinate certainty; keep the term and use a brief generic gloss only if needed. "
        "Examples of bad output to fix: 'биллинг [счетинг]', 'contagion' left in brackets, 'Ferrari [автомобили Ferrari]', "
        "'Google Calendar [календарь Google]', 'credits [лимит использования]', 'one-shotted it' left in English, "
        "'корпоративный долг [долг компаний]', 'Max 7 [вероятно, Max 7]', or ungrammatical phrases like 'ценный зернышко'. "
        "Return only JSON matching the provided schema."
    )


def build_russian_annotation_system_prompt() -> str:
    return (
        "You are a Russian-language editor adding concise glossary-style clarifications to an already translated podcast transcript. "
        "Preserve wording, meaning, tone, turn boundaries, and speaker assignment exactly. "
        "Do not merge turns, split turns, or rewrite sentences beyond minimal edits needed to insert a bracketed gloss. "
        "Assume the reader is intelligent but not an expert in the domain. "
        "Add a short Russian bracketed gloss on first mention only when a specialist term, acronym, metric, industry phrase, or product-specific concept would likely be unclear to a non-expert reader. "
        "Good candidates include terms like SLA, KYC, InMail, RAG, HVAC, InfiniBand, cryptowinter, and quant trader if they appear. "
        "Do not add glosses for obvious brands or ordinary product names such as Ferrari, Gmail, Google Calendar, Slack, Mac Mini, LinkedIn, Reddit, Claude, OpenAI, or Ethereum. "
        "Do not add glosses for plain concepts that already read naturally in Russian, such as corporate debt. "
        "Do not add glosses for already understandable Russian technical terms such as 'языковая модель', 'токены', 'субагент', or other phrases that are already self-explanatory in Russian context. "
        "If the main text already contains a clear Russian rendering, leave it alone instead of adding a bracket. "
        "Bracket text must be Russian only; never put English inside brackets. "
        "If a leftover English phrase can be translated naturally into Russian, prefer translating the phrase itself rather than explaining the English in brackets. "
        "Keep each gloss short, idiomatic, and genuinely informative. "
        "If you are not confident enough to write a useful gloss, leave the text unchanged. "
        "Never invent fake Russian words, pseudo-calques, or uncertain notes like 'вероятно'. "
        "Return only JSON matching the provided schema."
    )


def build_russian_cleanup_user_prompt(
    title_translated: str,
    turns: List[Dict[str, str]],
) -> str:
    return (
        f"Transcript title in Russian: {title_translated}\n"
        "Rewrite each turn below into better Russian while preserving meaning exactly. "
        "Keep the same number of turns and the same order. "
        "Return only the revised turn texts; do not include speaker labels inside the text.\n\n"
        "Turns:\n"
        f"{format_turns_for_cleanup(turns)}"
    )


def build_russian_annotation_user_prompt(
    title_translated: str,
    turns: List[Dict[str, str]],
) -> str:
    return (
        f"Transcript title in Russian: {title_translated}\n"
        "Review each turn below. Keep the same number of turns and the same order. "
        "Only add concise bracketed glosses where they are genuinely helpful for a non-expert reader. "
        "Return only the revised turn texts; do not include speaker labels inside the text.\n\n"
        "Turns:\n"
        f"{format_turns_for_cleanup(turns)}"
    )


def chunk_turns_by_chars(turns: List[Dict[str, str]], max_chars: int) -> List[List[Dict[str, str]]]:
    chunks: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    current_chars = 0
    for turn in turns:
        line_len = len(turn.get("text_translated", "")) + len(turn.get("speaker_id", "")) + 8
        if current and current_chars + line_len > max_chars:
            chunks.append(current)
            current = [turn]
            current_chars = line_len
        else:
            current.append(turn)
            current_chars += line_len
    if current:
        chunks.append(current)
    return chunks


def cleanup_russian_turn_chunk(
    client: OpenAI,
    model: str,
    title_translated: str,
    turns: List[Dict[str, str]],
    chunk_index: int = 1,
    chunk_count: int = 1,
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    system_prompt = build_russian_cleanup_system_prompt()
    user_prompt = build_russian_cleanup_user_prompt(title_translated, turns)
    schema = get_turn_cleanup_schema(len(turns))
    result = call_openai_with_retry(
        client,
        model,
        system_prompt,
        user_prompt,
        schema_name="cleaned_transcript_turns",
        schema=schema,
        temperature=OPENAI_CLEANUP_TEMPERATURE,
    )
    if debug_sink is not None:
        debug_sink.append(
            {
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "turn_count": len(turns),
                "result": result,
            }
        )
    cleaned_turns = result.get("turns", [])
    if len(cleaned_turns) != len(turns):
        raise RuntimeError("Cleanup pass returned the wrong number of turns")
    return [(item.get("text_translated") or "").strip() for item in cleaned_turns]


def cleanup_russian_turns(
    client: OpenAI,
    model: str,
    title_translated: str,
    turns: List[Dict[str, str]],
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    if not turns:
        return turns

    try:
        cleaned_texts = cleanup_russian_turn_chunk(
            client,
            model,
            title_translated,
            turns,
            debug_sink=debug_sink,
        )
    except Exception as exc:
        if not (is_context_length_error(exc) or is_request_too_large_error(exc)):
            raise
        cleaned_texts = []
        chunks = chunk_turns_by_chars(turns, max_chars=60_000)
        for chunk_index, chunk in enumerate(chunks, 1):
            cleaned_texts.extend(
                cleanup_russian_turn_chunk(
                    client,
                    model,
                    title_translated,
                    chunk,
                    chunk_index=chunk_index,
                    chunk_count=len(chunks),
                    debug_sink=debug_sink,
                )
            )

    if len(cleaned_texts) != len(turns):
        raise RuntimeError("Cleanup pass returned the wrong number of turns")

    cleaned_turns: List[Dict[str, str]] = []
    for turn, cleaned_text in zip(turns, cleaned_texts):
        cleaned_turn = dict(turn)
        cleaned_turn["text_translated"] = cleaned_text or (turn.get("text_translated") or "")
        cleaned_turns.append(cleaned_turn)
    return cleaned_turns


def annotate_russian_turn_chunk(
    client: OpenAI,
    model: str,
    title_translated: str,
    turns: List[Dict[str, str]],
    chunk_index: int = 1,
    chunk_count: int = 1,
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    schema = get_turn_cleanup_schema(len(turns))
    result = call_openai_with_retry(
        client,
        model,
        build_russian_annotation_system_prompt(),
        build_russian_annotation_user_prompt(title_translated, turns),
        schema_name="annotated_transcript_turns",
        schema=schema,
        temperature=OPENAI_ANNOTATION_TEMPERATURE,
    )
    if debug_sink is not None:
        debug_sink.append(
            {
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "turn_count": len(turns),
                "result": result,
            }
        )
    annotated_turns = result.get("turns", [])
    if len(annotated_turns) != len(turns):
        raise RuntimeError("Annotation pass returned the wrong number of turns")
    return [(item.get("text_translated") or "").strip() for item in annotated_turns]


def annotate_russian_turns(
    client: OpenAI,
    model: str,
    title_translated: str,
    turns: List[Dict[str, str]],
    debug_sink: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    if not turns:
        return turns

    try:
        annotated_texts = annotate_russian_turn_chunk(
            client,
            model,
            title_translated,
            turns,
            debug_sink=debug_sink,
        )
    except Exception as exc:
        if not (is_context_length_error(exc) or is_request_too_large_error(exc)):
            raise
        annotated_texts = []
        chunks = chunk_turns_by_chars(turns, max_chars=60_000)
        for chunk_index, chunk in enumerate(chunks, 1):
            annotated_texts.extend(
                annotate_russian_turn_chunk(
                    client,
                    model,
                    title_translated,
                    chunk,
                    chunk_index=chunk_index,
                    chunk_count=len(chunks),
                    debug_sink=debug_sink,
                )
            )

    if len(annotated_texts) != len(turns):
        raise RuntimeError("Annotation pass returned the wrong number of turns")

    annotated_turns: List[Dict[str, str]] = []
    for turn, annotated_text in zip(turns, annotated_texts):
        annotated_turn = dict(turn)
        annotated_turn["text_translated"] = annotated_text or (turn.get("text_translated") or "")
        annotated_turns.append(annotated_turn)
    return annotated_turns


def sanitize_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9\-_. ]+", "", ascii_text)
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    ascii_text = ascii_text.replace(" ", "-")
    ascii_text = re.sub(r"-+", "-", ascii_text)
    return ascii_text or "video"


def render_docx(
    title_translated: str,
    speakers: List[Dict[str, str]],
    turns: List[Dict[str, str]],
    output_path: str,
) -> None:
    doc = Document()
    doc.styles["Normal"].font.name = DOCX_FONT_NAME
    doc.styles["Normal"].font.size = DOCX_FONT_SIZE
    doc.styles["List Bullet"].font.name = DOCX_FONT_NAME
    doc.styles["List Bullet"].font.size = DOCX_FONT_SIZE
    doc.styles["Heading 1"].font.name = DOCX_FONT_NAME
    doc.styles["Heading 1"].font.size = DOCX_HEADING_FONT_SIZE
    doc.add_heading(title_translated.strip() or "Translated Transcript", level=1)

    if speakers:
        for speaker in speakers:
            label_full = speaker.get("label_full") or speaker.get("label_short") or speaker.get("id", "Speaker")
            doc.add_paragraph(label_full, style="List Bullet")

    for turn in turns:
        speaker_id = turn.get("speaker_id")
        text = (turn.get("text_translated") or "").strip()
        if not text:
            continue
        label = None
        for sp in speakers:
            if sp.get("id") == speaker_id:
                label = sp.get("label_short") or sp.get("id")
                break
        label = label or speaker_id or "Speaker"
        para = doc.add_paragraph()
        run = para.add_run(f"{label}: ")
        run.bold = True
        para.add_run(text)

    doc.save(output_path)


def convert_docx_to_pdf(docx_path: str) -> str:
    output_dir = os.path.dirname(docx_path) or os.getcwd()
    base_name = os.path.splitext(os.path.basename(docx_path))[0]
    expected_pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

    candidate_libreoffice_paths = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice.bin",
    ]

    converters = [
        (shutil.which("soffice"), [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            docx_path,
        ]),
        (next((p for p in candidate_libreoffice_paths if os.access(p, os.X_OK)), None), [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            docx_path,
        ]),
        (shutil.which("libreoffice"), [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            docx_path,
        ]),
    ]

    for exe, cmd in converters:
        if not exe:
            continue
        result = subprocess.run(
            cmd,
            cwd=output_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and os.path.exists(expected_pdf_path):
            return expected_pdf_path

    try:
        from docx2pdf import convert
    except Exception:
        raise RuntimeError(
            "PDF conversion is unavailable. Install one of:\n"
            "- LibreOffice (soffice/libreoffice in PATH, or a standard macOS install at /Applications/LibreOffice.app), or\n"
            "- Python package docx2pdf (pip install docx2pdf)."
        )

    convert(docx_path, expected_pdf_path)
    if not os.path.exists(expected_pdf_path):
        raise RuntimeError("docx2pdf did not generate the expected PDF file.")
    return expected_pdf_path


def send_completion_notification(message: str, title: str = "ytranslate") -> None:
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    safe_message = message.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{safe_message}" with title "{safe_title}"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"Failed to send macOS notification: {exc}", file=sys.stderr)

def sample_docx_payload(target_language: str) -> Dict[str, Any]:
    return {
        "title_translated": f"Пример перевода ({target_language})",
        "speakers": [
            {
                "id": "Host",
                "label_short": "Host",
                "label_full": "Alex Kantrowitz — Host, Big Technology Podcast",
            },
            {
                "id": "Guest",
                "label_short": "Guest",
                "label_full": "Demis Hassabis — CEO, Google DeepMind",
            },
        ],
        "turns": [
            {
                "speaker_id": "Host",
                "text_translated": "Это тестовый фрагмент для проверки генерации DOCX.",
            },
            {
                "speaker_id": "Guest",
                "text_translated": "Отлично. Убедимся, что перенос строк и шрифты работают.",
            },
        ],
    }


def run_sample_generation(
    target_language: Optional[str],
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    resolved_target_language = resolve_target_language(target_language)
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    filename = "sample.docx"
    output_path = os.path.join(output_dir, filename)
    sample = sample_docx_payload(resolved_target_language)
    render_docx(
        sample.get("title_translated", "Sample"),
        sample.get("speakers", []),
        sample.get("turns", []),
        output_path,
    )
    log(f"Saved sample DOCX to {output_path}")
    try:
        pdf_path = convert_docx_to_pdf(output_path)
        log(f"Saved sample PDF to {pdf_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to generate sample PDF: {exc}") from exc
    output_files = [output_path, pdf_path]
    send_completion_notification(
        "Sample conversion finished: " + ", ".join(os.path.basename(p) for p in output_files)
    )
    return {
        "target_language": resolved_target_language,
        "docx_path": output_path,
        "pdf_path": pdf_path,
        "output_files": output_files,
    }


def run_translation_job(
    url: str,
    target_language: Optional[str] = None,
    debug: bool = False,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    load_project_env()
    canonical_url = canonicalize_youtube_url(url)
    if not canonical_url:
        raise RuntimeError("Could not extract video ID from URL")

    resolved_target_language = resolve_target_language(target_language)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_key:
        raise RuntimeError("YOUTUBE_API_KEY is not set")

    video_id = extract_video_id(canonical_url)
    if not video_id:
        raise RuntimeError("Could not extract video ID from URL")

    log(f"Received URL: {canonical_url}")
    log(f"Target language: {resolved_target_language}")

    log("Fetching metadata...")
    metadata = fetch_video_metadata(video_id, youtube_key)
    title = metadata.get("title") or "Untitled"
    description = metadata.get("description", "")
    source_language_hint = metadata.get("defaultAudioLanguage") or metadata.get("defaultLanguage")
    debug_dir = make_debug_output_dir(video_id, title) if debug else None
    speaker_pass_debug: List[Dict[str, Any]] = []
    translation_pass_debug: List[Dict[str, Any]] = []
    cleanup_pass_debug: List[Dict[str, Any]] = []
    annotation_pass_debug: List[Dict[str, Any]] = []

    log("Fetching transcript...")
    preferred_langs = [
        metadata.get("defaultAudioLanguage"),
        metadata.get("defaultLanguage"),
    ]
    try:
        transcript_info = fetch_transcript(video_id, preferred_langs)
    except TranscriptsDisabled as exc:
        raise RuntimeError("Transcripts are disabled for this video.") from exc
    except NoTranscriptFound as exc:
        raise RuntimeError("No transcript found for this video.") from exc

    segments = normalize_segments(transcript_info.get("segments", []))
    if not segments:
        raise RuntimeError("Transcript was empty after normalization.")

    if debug_dir:
        write_json_file(
            os.path.join(debug_dir, "metadata.json"),
            metadata,
        )
        write_json_file(
            os.path.join(debug_dir, "raw-transcript.json"),
            transcript_info,
        )
        write_text_file(
            os.path.join(debug_dir, "normalized-transcript.md"),
            "# Normalized Transcript\n\n" + format_segments(segments) + "\n",
        )
        log(f"Wrote transcript debug artifacts to {debug_dir}")

    log("Attributing speakers and structuring turns...")
    client = OpenAI(api_key=openai_key, timeout=OPENAI_TIMEOUT_SECONDS)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    attributed = attribute_speakers(
        client,
        model,
        canonical_url,
        title,
        description,
        segments,
        source_language_hint,
        debug_sink=speaker_pass_debug if debug else None,
    )

    if debug_dir:
        if speaker_pass_debug:
            for idx, item in enumerate(speaker_pass_debug, 1):
                write_json_file(
                    os.path.join(debug_dir, f"speaker-pass-{idx:02d}.json"),
                    item,
                )
        else:
            write_json_file(os.path.join(debug_dir, "speaker-pass.json"), attributed)

    log("Translating attributed turns...")
    result = translate_attributed_turns(
        client,
        model,
        canonical_url,
        title,
        description,
        resolved_target_language,
        attributed.get("speakers", []),
        attributed.get("turns", []),
        source_language_hint,
        debug_sink=translation_pass_debug if debug else None,
    )

    if debug_dir:
        if translation_pass_debug:
            for idx, item in enumerate(translation_pass_debug, 1):
                write_json_file(
                    os.path.join(debug_dir, f"translation-pass-{idx:02d}.json"),
                    item,
                )
        else:
            write_json_file(os.path.join(debug_dir, "translation-pass.json"), result)

    if is_russian_target_language(resolved_target_language):
        log("Polishing Russian wording and glossary explanations...")
        result["turns"] = cleanup_russian_turns(
            client,
            model,
            result.get("title_translated", "").strip() or title,
            result.get("turns", []),
            debug_sink=cleanup_pass_debug if debug else None,
        )

    if debug_dir and cleanup_pass_debug:
        for idx, item in enumerate(cleanup_pass_debug, 1):
            write_json_file(
                os.path.join(debug_dir, f"cleanup-pass-{idx:02d}.json"),
                item,
            )

    if is_russian_target_language(resolved_target_language):
        log("Adding targeted glossary annotations for non-expert readers...")
        result["turns"] = annotate_russian_turns(
            client,
            model,
            result.get("title_translated", "").strip() or title,
            result.get("turns", []),
            debug_sink=annotation_pass_debug if debug else None,
        )

    if debug_dir and annotation_pass_debug:
        for idx, item in enumerate(annotation_pass_debug, 1):
            write_json_file(
                os.path.join(debug_dir, f"annotation-pass-{idx:02d}.json"),
                item,
            )

    title_translated = result.get("title_translated", "").strip() or title
    if debug_dir:
        write_json_file(os.path.join(debug_dir, "final.json"), result)
        final_md_path = os.path.join(debug_dir, "final.md")
        final_md = render_markdown_transcript(
            title_translated,
            result.get("speakers", []),
            result.get("turns", []),
            metadata={
                "url": canonical_url,
                "video_id": video_id,
                "target_language": resolved_target_language,
                "model": model,
                "timeout_seconds": OPENAI_TIMEOUT_SECONDS,
                "temperature": OPENAI_TEMPERATURE,
                "cleanup_temperature": OPENAI_CLEANUP_TEMPERATURE,
                "annotation_temperature": OPENAI_ANNOTATION_TEMPERATURE,
                "source_language_hint": source_language_hint or "",
                "cleanup_ran": is_russian_target_language(resolved_target_language),
            },
        )
        write_text_file(final_md_path, final_md)
        log(f"Saved debug Markdown to {final_md_path}")
        log(f"Finished generating debug artifacts for {canonical_url}")
        return {
            "url": canonical_url,
            "video_id": video_id,
            "title": title,
            "title_translated": title_translated,
            "target_language": resolved_target_language,
            "debug_dir": debug_dir,
            "final_md_path": final_md_path,
            "output_files": [final_md_path],
        }

    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{sanitize_filename(title)}.docx"
    output_path = os.path.join(output_dir, filename)

    render_docx(
        title_translated,
        result.get("speakers", []),
        result.get("turns", []),
        output_path,
    )
    log(f"Saved translated transcript to {output_path}")
    try:
        pdf_path = convert_docx_to_pdf(output_path)
        log(f"Saved translated transcript PDF to {pdf_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to generate transcript PDF: {exc}") from exc

    output_files = [output_path, pdf_path]
    send_completion_notification(
        "Translation completed: " + ", ".join(os.path.basename(p) for p in output_files)
    )
    log(f"Finished generating files for {canonical_url}")
    return {
        "url": canonical_url,
        "video_id": video_id,
        "title": title,
        "title_translated": title_translated,
        "target_language": resolved_target_language,
        "docx_path": output_path,
        "pdf_path": pdf_path,
        "output_files": output_files,
    }


def main() -> int:
    args = parse_args()
    try:
        if args.docx_test:
            if args.debug:
                raise RuntimeError("--debug is not supported together with --docx-test")
            load_project_env()
            run_sample_generation(args.target_language)
        else:
            run_translation_job(args.url, args.target_language, debug=args.debug)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
