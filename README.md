# ytranslate

Translate a YouTube video into a target language and structure it as a conversation.

## Requirements
- Python 3.10+
- Google Chrome
- OpenAI API key (`OPENAI_API_KEY`)
- YouTube Data API key (`YOUTUBE_API_KEY`)
- Audio transcription dependencies from `requirements.txt` (`yt-dlp` and bundled `imageio-ffmpeg`)

## Install
### With uv (recommended)
```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

### With pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
### Recommended: launchd-managed local server + Chrome extension
Install and start the per-user macOS LaunchAgent:
```bash
scripts/install-launchagent.sh install
```

Check it:
```bash
curl http://127.0.0.1:8765/health
scripts/install-launchagent.sh status
```

Open the status page:
```text
http://127.0.0.1:8765/
```

JSON status endpoints:
```bash
curl http://127.0.0.1:8765/api/status
curl http://127.0.0.1:8765/jobs/latest
curl http://127.0.0.1:8765/jobs
```

The LaunchAgent:
- starts automatically at login
- keeps the server running
- runs from this checkout and reads the local `.env`
- writes logs to `~/Library/Logs/ytranslate/server.log` and `~/Library/Logs/ytranslate/server.err.log`
- writes recent job status to `~/Library/Application Support/ytranslate/jobs.json`

Stop and remove it:
```bash
scripts/install-launchagent.sh uninstall
```

For a foreground/manual server run:
```bash
source .venv/bin/activate
python ytranslate_server.py
```

Load the unpacked extension from:
```text
extension/
```

Then open a YouTube watch page. A compact `PDF` button is injected to the left of the like/dislike control. Clicking it queues the current video for processing on the local server.

The server:
- prints logs to the terminal when run manually
- saves `.docx` and `.pdf` to `~/Downloads/`
- shows a macOS desktop notification on completion
- uses the status page as the progress and failure surface

The status UI is a small React/Vite app in `frontend/` and is served by the Python server from `frontend/dist/`. After changing frontend source:
```bash
npm --prefix frontend install
npm --prefix frontend run build
```

### Manual CLI
```bash
python ytranslate.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Override the default target language:
```bash
python ytranslate.py "https://www.youtube.com/watch?v=VIDEO_ID" "French"
```

Debug a run without generating DOCX/PDF:
```bash
python ytranslate.py "https://www.youtube.com/watch?v=VIDEO_ID" --debug
```

Or with uv:
```bash
uv run ytranslate "https://www.youtube.com/watch?v=VIDEO_ID"
```

PDF output uses one of:
- LibreOffice CLI (`soffice` or `libreoffice`)
- or `docx2pdf` (`pip install docx2pdf`)

DOCX test without calling external APIs:
```bash
python ytranslate.py "https://www.youtube.com/watch?v=VIDEO_ID" "French" --docx-test
```

DOCX test (also generates PDF):
```bash
python ytranslate.py "https://www.youtube.com/watch?v=VIDEO_ID" "French" --docx-test
```

You can also put your keys in a local `.env` file in the project root:
```bash
OPENAI_API_KEY=...
YOUTUBE_API_KEY=...
OPENAI_MODEL=gpt-5.4-mini
OPENAI_ASR_MODEL=gpt-4o-transcribe-diarize
DEFAULT_TARGET_LANGUAGE=Russian
```

The output is saved to `~/Downloads/` as:
```
<video-title>.docx
```

PDF is also generated alongside it:
```
<video-title>.pdf
```

Debug mode writes a per-run folder under `~/Downloads/`, for example:
```
ytranslate-debug-<video-id>-<timestamp>-<video-title>/
```

Artifacts include:
- `metadata.json`
- `youtube-transcript.json` and `youtube-normalized-transcript.md` when a YouTube transcript is available
- `openai-asr.json` when OpenAI diarized ASR is used
- `speaker-mapping*.json` when ASR chunk-local speakers are reconciled globally
- `source-attributed-turns.json`
- `translation-pass-*.json`
- `cleanup-pass-*.json` (when Russian cleanup runs)
- `annotation-pass-*.json` (when Russian annotation runs)
- `final.json`
- `final.md`

## YouTube API key
1) Go to Google Cloud Console and select (or create) a project.
2) Enable "YouTube Data API v3" for the project.
3) Go to APIs & Services -> Credentials -> Create credentials -> API key.
4) (Recommended) Restrict the key to YouTube Data API v3.

Then set:
```bash
export YOUTUBE_API_KEY=\"...\"
```

## Notes
- Transcript source selection is either/or. A manual YouTube transcript is used only when it clearly includes speaker labels. Otherwise the tool downloads audio and uses OpenAI diarized ASR (`gpt-4o-transcribe-diarize`). It does not fall back to speakerless YouTube auto-captions.
- OpenAI ASR audio is cached under `~/Library/Caches/ytranslate/` and split into compressed chunks under the upload limit. The default ASR chunk length is 10 minutes, chosen for long-run reliability and clearer progress; override with `OPENAI_ASR_CHUNK_SECONDS` if needed.
- OpenAI ASR uploads use `curl` by default on macOS (`OPENAI_ASR_TRANSPORT=curl`) so large multipart uploads go through the system curl TLS stack. Set `OPENAI_ASR_TRANSPORT=requests` to use Python requests instead.
- Each ASR chunk has request-level retries and the job can make later passes over only the chunks that failed. Tune with `OPENAI_ASR_MAX_RETRIES`, `OPENAI_ASR_MAX_PASSES`, and `OPENAI_ASR_RETRY_PASS_DELAY_SECONDS`.
- ASR requests use a separate timeout from text-generation calls. The default is 600 seconds per chunk attempt; override with `OPENAI_ASR_TIMEOUT_SECONDS`.
- Metadata (title/description) is fetched via the official YouTube Data API to help infer speakers.
- The Chrome extension talks to `http://127.0.0.1:8765`.
- The status page at `http://127.0.0.1:8765/` shows the latest job, recent jobs, pipeline steps, ASR chunk progress, output paths, and recent events.
- The extension displays YouTube-style bottom-left toast notifications for request feedback.
- On macOS, successful completion sends a desktop notification (no configuration required).
- OpenAI ASR defaults to one upload at a time for reliability on long episodes. Set `OPENAI_ASR_JOBS=2` or higher only when you prefer speed over lower TLS/upload risk.
