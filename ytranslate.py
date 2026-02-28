#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
import inspect
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
from openai import OpenAI
import openai
from docx import Document
from docx.shared import Pt
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled


YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"
DEFAULT_MODEL = "gpt-5.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate a YouTube video's transcript into a target language "
            "and structure it as a conversation."
        )
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("target_language", help="Target language (e.g., French)")
    parser.add_argument(
        "--docx-test",
        action="store_true",
        help="Generate a sample DOCX without calling external APIs",
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


def format_segments(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, seg in enumerate(segments, 1):
        lines.append(f"[{idx}] {seg['text']}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are a professional transcript translator and dialogue editor. "
        "Translate the entire transcript into the target language. "
        "Do not summarize or omit any spoken content. "
        "Do not add new information. "
        "Translate the video title into the target language. "
        "Use the video title and description to infer speaker identities/roles. "
        "If a speaker is uncertain, use consistent labels like 'Speaker 1', 'Speaker 2'. "
        "Produce idiomatic, natural-sounding language. You may lightly paraphrase for fluency, "
        "but stay as close to the original as possible. "
        "Preserve the tone of the conversation (serious, friendly, banter, etc.). "
        "Translate English loanwords and jargon into natural target-language phrasing. "
        "Do not leave untranslated/transliterated English terms (for example, avoid outputs like 'дисрапт'). "
        "Before finalizing, self-check that no English business/technical phrase is left unexplained in the target language. "
        "Translate inline explanations and side remarks fully, including quoted definitions by other speakers. "
        "If no exact equivalent exists, use a concise explanatory phrase in brackets on first mention, "
        "then use a consistent idiomatic term afterward. "
        "For acronyms, keep the acronym where helpful but translate the expanded meaning into the target language. "
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
            "- Do not leave English phrases like 'net dollar retention', 'value prop', 'foundation models' untranslated.\n"
            "- Translate term definitions fully (example: 'remaining performance obligations' must be in Russian).\n"
            "- If keeping an acronym (ARR/NDR/RPO/AGI), translate the expanded meaning into Russian on first mention.\n"
            "- If there is no exact term, use a short explanatory bracket after an idiomatic Russian phrasing.\n"
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
) -> Dict[str, Any]:
    response_format_v1 = {
        "type": "json_schema",
        "json_schema": {
            "name": "translated_transcript",
            "strict": True,
            "schema": get_schema(),
        },
    }
    response_format_v2 = {
        "type": "json_schema",
        "name": "translated_transcript",
        "strict": True,
        "schema": get_schema(),
    }

    response_kwargs = {
        "model": model,
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
) -> Dict[str, Any]:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return call_openai(client, model, system_prompt, user_prompt)
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
    return call_openai_with_retry(client, model, system_prompt, user_prompt)


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
) -> Dict[str, Any]:
    system_prompt = build_system_prompt()
    chunks = chunk_segments(segments, max_chars=max_chars)
    speakers: List[Dict[str, str]] = []
    turns: List[Dict[str, str]] = []
    title_translated: Optional[str] = None

    for chunk in chunks:
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
            )
        except Exception as exc:
            if is_context_length_error(exc):
                continue
            raise

    raise RuntimeError("Unable to translate transcript within context limits")


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


def main() -> int:
    args = parse_args()
    url = args.url
    target_language = args.target_language

    load_dotenv(os.path.join(os.getcwd(), ".env"))

    if args.docx_test:
        output_dir = os.path.join(os.getcwd(), "translations")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"sample--{sanitize_filename(target_language)}.docx"
        output_path = os.path.join(output_dir, filename)
        sample = sample_docx_payload(target_language)
        render_docx(
            sample.get("title_translated", "Sample"),
            sample.get("speakers", []),
            sample.get("turns", []),
            output_path,
        )
        print(f"Saved sample DOCX to {output_path}")
        return 0

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_key:
        print("YOUTUBE_API_KEY is not set", file=sys.stderr)
        return 1

    video_id = extract_video_id(url)
    if not video_id:
        print("Could not extract video ID from URL", file=sys.stderr)
        return 1

    print("Fetching metadata...")
    try:
        metadata = fetch_video_metadata(video_id, youtube_key)
    except Exception as exc:
        print(f"Failed to fetch video metadata: {exc}", file=sys.stderr)
        return 1
    title = metadata.get("title") or "Untitled"
    description = metadata.get("description", "")
    source_language_hint = metadata.get("defaultAudioLanguage") or metadata.get("defaultLanguage")

    print("Fetching transcript...")
    preferred_langs = [
        metadata.get("defaultAudioLanguage"),
        metadata.get("defaultLanguage"),
    ]

    try:
        transcript_info = fetch_transcript(video_id, preferred_langs)
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.", file=sys.stderr)
        return 1
    except NoTranscriptFound:
        print("No transcript found for this video.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed to fetch transcript: {exc}", file=sys.stderr)
        return 1

    segments = normalize_segments(transcript_info.get("segments", []))
    if not segments:
        print("Transcript was empty after normalization.", file=sys.stderr)
        return 1

    print("Translating and structuring transcript (this may take a while)...")
    client = OpenAI(api_key=openai_key)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    result = translate_transcript(
        client,
        model,
        url,
        title,
        description,
        target_language,
        segments,
        source_language_hint,
    )

    title_translated = result.get("title_translated", "").strip() or title

    output_dir = os.path.join(os.getcwd(), "translations")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{sanitize_filename(title)}--{sanitize_filename(target_language)}.docx"
    output_path = os.path.join(output_dir, filename)

    render_docx(
        title_translated,
        result.get("speakers", []),
        result.get("turns", []),
        output_path,
    )

    print(f"Saved translated transcript to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
