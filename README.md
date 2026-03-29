# ytranslate

Translate a YouTube video's transcript into a target language and structure it as a conversation.

## Requirements
- Python 3.9+
- OpenAI API key (`OPENAI_API_KEY`)
- YouTube Data API key (`YOUTUBE_API_KEY`)

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
```bash
python ytranslate.py "https://www.youtube.com/watch?v=VIDEO_ID" "French"
```

Or with uv:
```bash
uv run ytranslate "https://www.youtube.com/watch?v=VIDEO_ID" "French"
```

PDF is always generated automatically:
```bash
python ytranslate.py "https://www.youtube.com/watch?v=VIDEO_ID" "French"
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
OPENAI_MODEL=gpt-5.4
```

The output is saved to `~/Downloads/` as:
```
<video-title>--<target-language>.docx
```

PDF is also generated alongside it:
```
<video-title>--<target-language>.pdf
```

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
- Transcript retrieval uses an unofficial endpoint via `youtube-transcript-api`. Some videos do not expose transcripts or may block these requests.
- Metadata (title/description) is fetched via the official YouTube Data API to help infer speakers.
- Output is a DOCX file for easy editing in Google Docs.
- On macOS, successful completion sends a desktop notification (no configuration required).
