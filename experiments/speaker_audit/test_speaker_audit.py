from pathlib import Path
import tempfile
import unittest

from speaker_audit import build_audit_rows, render_audit_html


class SpeakerAuditTests(unittest.TestCase):
    def test_build_audit_rows_keeps_raw_and_mapped_speakers(self):
        asr_segments = [
            {
                "chunk_index": 2,
                "local_speaker": "A",
                "speaker": "A",
                "start": 1200.02,
                "end": 1203.5,
                "text": "That was my question.",
            },
            {
                "chunk_index": 2,
                "local_speaker": "B",
                "speaker": "B",
                "start": 1204,
                "end": 1207,
                "text": "Here is the answer.",
            },
        ]
        speaker_mapping = {
            "speakers": [
                {"id": "speaker_alice", "label_short": "Alice", "label_full": "Alice Example"},
                {"id": "speaker_bob", "label_short": "Bob", "label_full": "Bob Example"},
            ],
            "local_speakers": [
                {"chunk_index": 2, "local_speaker": "A", "speaker_id": "speaker_alice"},
                {"chunk_index": 2, "local_speaker": "B", "speaker_id": "speaker_bob"},
            ],
        }

        rows = build_audit_rows(asr_segments, speaker_mapping)

        self.assertEqual(rows[0]["local_speaker"], "A")
        self.assertEqual(rows[0]["mapped_speaker_id"], "speaker_alice")
        self.assertEqual(rows[0]["mapped_speaker_label"], "Alice")
        self.assertEqual(rows[0]["start_timecode"], "00:20:00")
        self.assertEqual(rows[1]["mapped_speaker_label"], "Bob")

    def test_build_audit_rows_falls_back_to_local_speaker_without_mapping(self):
        rows = build_audit_rows(
            [
                {
                    "chunk_index": 1,
                    "local_speaker": "C",
                    "speaker": "C",
                    "start": 12.3,
                    "end": 14.6,
                    "text": "Unmapped text.",
                }
            ],
            speaker_mapping=None,
        )

        self.assertEqual(rows[0]["mapped_speaker_id"], "speaker_c")
        self.assertEqual(rows[0]["mapped_speaker_label"], "Speaker C")

    def test_render_audit_html_contains_audio_and_seek_controls(self):
        rows = [
            {
                "index": 1,
                "chunk_index": 1,
                "local_speaker": "A",
                "mapped_speaker_id": "speaker_alice",
                "mapped_speaker_label": "Alice",
                "start": 1.25,
                "end": 3.5,
                "start_timecode": "00:00:01",
                "end_timecode": "00:00:03",
                "text": "Hello <world>.",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "source.m4a"
            html = render_audit_html(
                title="Audit <Episode>",
                video_id="abc123",
                source_url="https://youtu.be/abc123",
                audio_path=audio_path,
                rows=rows,
                speaker_mapping={"speakers": []},
                asr_summary={"model": "gpt-4o-transcribe-diarize", "chunk_seconds": 1200},
            )

        self.assertIn("Audit &lt;Episode&gt;", html)
        self.assertIn("file://", html)
        self.assertIn("seekTo(1.250)", html)
        self.assertIn("Hello &lt;world&gt;.", html)


if __name__ == "__main__":
    unittest.main()
