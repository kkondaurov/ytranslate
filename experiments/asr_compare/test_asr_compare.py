import json
from pathlib import Path

from asr_compare import (
    Episode,
    build_chunk_offsets,
    chunk_output_path,
    format_timestamp,
    list_transcripts_for_video,
    merge_diarized_segments,
    render_report_html,
    safe_slug,
)


def test_format_timestamp_uses_hours_when_needed():
    assert format_timestamp(83.2) == "01:23"
    assert format_timestamp(3723.8) == "01:02:04"


def test_safe_slug_keeps_identifying_text():
    assert safe_slug("All-In E219: AI, markets & chaos!") == "all-in-e219-ai-markets-chaos"


def test_chunk_output_path_is_stable_and_sortable(tmp_path):
    path = chunk_output_path(tmp_path, "abc123", 7)
    assert path == tmp_path / "abc123" / "chunks" / "chunk-007.mp3"


def test_build_chunk_offsets_uses_cumulative_durations():
    assert build_chunk_offsets([5.5, 10.25, 2.0]) == [0.0, 5.5, 15.75]


def test_list_transcripts_for_video_supports_new_instance_api():
    class FakeApi:
        def list(self, video_id):
            return [f"new:{video_id}"]

    assert list_transcripts_for_video("abc123", api_cls=FakeApi) == ["new:abc123"]


def test_list_transcripts_for_video_supports_old_static_api():
    class FakeApi:
        @staticmethod
        def list_transcripts(video_id):
            return [f"old:{video_id}"]

    assert list_transcripts_for_video("abc123", api_cls=FakeApi) == ["old:abc123"]


def test_merge_diarized_segments_offsets_chunks_and_preserves_speakers():
    chunks = [
        {
            "offset_seconds": 0,
            "segments": [
                {"start": 1.0, "end": 2.5, "speaker": "speaker_0", "text": "hello"},
            ],
        },
        {
            "offset_seconds": 300,
            "segments": [
                {"start": 0.5, "end": 4.0, "speaker": "speaker_1", "text": "there"},
            ],
        },
    ]

    merged = merge_diarized_segments(chunks)

    assert merged == [
        {"start": 1.0, "end": 2.5, "speaker": "speaker_0", "text": "hello"},
        {"start": 300.5, "end": 304.0, "speaker": "speaker_1", "text": "there"},
    ]


def test_render_report_html_contains_parallel_episode_columns(tmp_path):
    episode = Episode(
        label="all-in 1",
        url="https://youtu.be/example",
        video_id="example",
        title="Example Episode",
    )
    episode_dir = tmp_path / "example"
    episode_dir.mkdir()
    (episode_dir / "youtube_transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "duration": 2.0, "text": "YouTube text"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (episode_dir / "openai_diarized.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "speaker": "speaker_0",
                        "text": "OpenAI text",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    html = render_report_html([episode], tmp_path)

    assert "Example Episode" in html
    assert "YouTube text" in html
    assert "OpenAI text" in html
    assert 'class="comparison-grid"' in html
