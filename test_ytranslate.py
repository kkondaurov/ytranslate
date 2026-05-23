import unittest
from unittest.mock import Mock, patch

import ytranslate


class TranscriptSourceTests(unittest.TestCase):
    def test_generated_speakerless_youtube_transcript_is_not_high_quality(self):
        transcript = {
            "is_generated": True,
            "segments": [
                {"start": 0, "duration": 2, "text": "welcome back everyone"},
                {"start": 2, "duration": 2, "text": "this is a fast exchange"},
            ],
        }

        self.assertFalse(ytranslate.is_high_quality_youtube_transcript(transcript))

    def test_manual_speaker_labeled_youtube_transcript_is_high_quality(self):
        transcript = {
            "is_generated": False,
            "segments": [
                {"start": 0, "duration": 2, "text": "Host: Welcome back."},
                {"start": 2, "duration": 2, "text": "Guest: Thanks for having me."},
                {"start": 4, "duration": 2, "text": "Host: Let's start with AI."},
                {"start": 6, "duration": 2, "text": "Guest: Absolutely."},
                {"start": 8, "duration": 2, "text": "Host: What changed?"},
                {"start": 10, "duration": 2, "text": "Guest: The systems got better."},
                {"start": 12, "duration": 2, "text": "Host: That's useful."},
                {"start": 14, "duration": 2, "text": "Guest: It is."},
            ],
        }

        self.assertTrue(ytranslate.is_high_quality_youtube_transcript(transcript))

    def test_labeled_youtube_segments_become_attributed_turns(self):
        segments = ytranslate.normalize_segments(
            [
                {"start": 0, "duration": 2, "text": "Host: Welcome back."},
                {"start": 2, "duration": 2, "text": "Guest: Thanks."},
                {"start": 4, "duration": 2, "text": "I am glad to be here."},
            ]
        )

        attributed = ytranslate.attributed_turns_from_labeled_segments(segments)

        self.assertEqual(
            attributed["speakers"],
            [
                {"id": "speaker_host", "label_short": "Host", "label_full": "Host"},
                {"id": "speaker_guest", "label_short": "Guest", "label_full": "Guest"},
            ],
        )
        self.assertEqual(
            attributed["turns"],
            [
                {"speaker_id": "speaker_host", "text_source": "Welcome back."},
                {"speaker_id": "speaker_guest", "text_source": "Thanks. I am glad to be here."},
            ],
        )


class DiarizedAsrTests(unittest.TestCase):
    def test_default_asr_chunk_length_stays_under_model_limit(self):
        self.assertLessEqual(ytranslate.ASR_CHUNK_SECONDS, ytranslate.ASR_MAX_CHUNK_SECONDS)

    def test_asr_chunk_length_rejects_values_above_model_limit(self):
        with patch.dict("os.environ", {"OPENAI_ASR_CHUNK_SECONDS": "2700"}):
            with self.assertRaisesRegex(RuntimeError, "exceeds the OpenAI ASR model limit"):
                ytranslate.get_asr_chunk_seconds()

    def test_extract_diarized_segments_normalizes_openai_response(self):
        response = {
            "segments": [
                {
                    "speaker": "A",
                    "start": 1.2,
                    "end": 3.4,
                    "text": " Hello there ",
                }
            ]
        }

        self.assertEqual(
            ytranslate.extract_diarized_segments(response),
            [{"speaker": "A", "start": 1.2, "end": 3.4, "text": "Hello there"}],
        )

    def test_merge_diarized_chunks_applies_offsets(self):
        chunks = [
            {
                "chunk_index": 1,
                "chunk": "chunk-001.mp3",
                "offset_seconds": 0,
                "segments": [
                    {"speaker": "A", "start": 1, "end": 2, "text": "first"},
                ],
            },
            {
                "chunk_index": 2,
                "chunk": "chunk-002.mp3",
                "offset_seconds": 300,
                "segments": [
                    {"speaker": "B", "start": 2.5, "end": 4, "text": "second"},
                ],
            },
        ]

        self.assertEqual(
            ytranslate.merge_diarized_chunks(chunks),
            [
                {
                    "speaker": "A",
                    "local_speaker": "A",
                    "chunk_index": 1,
                    "chunk": "chunk-001.mp3",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "first",
                },
                {
                    "speaker": "B",
                    "local_speaker": "B",
                    "chunk_index": 2,
                    "chunk": "chunk-002.mp3",
                    "start": 302.5,
                    "end": 304.0,
                    "text": "second",
                },
            ],
        )

    def test_diarized_segments_become_attributed_turns(self):
        segments = [
            {"speaker": "A", "start": 0, "end": 2, "text": "Hello."},
            {"speaker": "B", "start": 2, "end": 4, "text": "Hi."},
        ]

        attributed = ytranslate.attributed_turns_from_diarized_segments(segments)

        self.assertEqual(
            attributed["speakers"],
            [
                {"id": "speaker_a", "label_short": "Speaker A", "label_full": "Speaker A"},
                {"id": "speaker_b", "label_short": "Speaker B", "label_full": "Speaker B"},
            ],
        )
        self.assertEqual(
            attributed["turns"],
            [
                {"speaker_id": "speaker_a", "text_source": "Hello."},
                {"speaker_id": "speaker_b", "text_source": "Hi."},
            ],
        )

    def test_speaker_profiles_group_by_chunk_and_local_speaker(self):
        segments = [
            {"chunk_index": 1, "local_speaker": "A", "start": 0, "end": 1, "text": "Alice one."},
            {"chunk_index": 1, "local_speaker": "A", "start": 2, "end": 3, "text": "Alice two."},
            {"chunk_index": 2, "local_speaker": "A", "start": 10, "end": 11, "text": "Bob one."},
        ]

        profiles = ytranslate.build_local_speaker_profiles(segments, max_chars_per_profile=50)

        self.assertEqual(
            profiles,
            [
                {
                    "chunk_index": 1,
                    "local_speaker": "A",
                    "segment_count": 2,
                    "start": 0.0,
                    "end": 3.0,
                    "samples": ["Alice one.", "Alice two."],
                },
                {
                    "chunk_index": 2,
                    "local_speaker": "A",
                    "segment_count": 1,
                    "start": 10.0,
                    "end": 11.0,
                    "samples": ["Bob one."],
                },
            ],
        )

    def test_local_speaker_mapping_is_applied_before_merging_turns(self):
        segments = [
            {"chunk_index": 1, "local_speaker": "A", "speaker": "A", "text": "Alice first."},
            {"chunk_index": 1, "local_speaker": "B", "speaker": "B", "text": "Bob first."},
            {"chunk_index": 2, "local_speaker": "A", "speaker": "A", "text": "Bob second."},
            {"chunk_index": 2, "local_speaker": "B", "speaker": "B", "text": "Alice second."},
        ]
        speaker_mapping = {
            "speakers": [
                {"id": "speaker_alice", "label_short": "Alice", "label_full": "Alice"},
                {"id": "speaker_bob", "label_short": "Bob", "label_full": "Bob"},
            ],
            "local_speakers": [
                {"chunk_index": 1, "local_speaker": "A", "speaker_id": "speaker_alice"},
                {"chunk_index": 1, "local_speaker": "B", "speaker_id": "speaker_bob"},
                {"chunk_index": 2, "local_speaker": "A", "speaker_id": "speaker_bob"},
                {"chunk_index": 2, "local_speaker": "B", "speaker_id": "speaker_alice"},
            ],
        }

        attributed = ytranslate.attributed_turns_from_diarized_segments(segments, speaker_mapping)

        self.assertEqual(attributed["speakers"], speaker_mapping["speakers"])
        self.assertEqual(
            attributed["turns"],
            [
                {"speaker_id": "speaker_alice", "text_source": "Alice first."},
                {"speaker_id": "speaker_bob", "text_source": "Bob first. Bob second."},
                {"speaker_id": "speaker_alice", "text_source": "Alice second."},
            ],
        )


class RunTranslationJobSourceChoiceTests(unittest.TestCase):
    def run_with_common_mocks(self, transcript_info):
        asr_result = {
            "model": "gpt-4o-transcribe-diarize",
            "chunk_seconds": 1200,
            "segments": [
                {
                    "chunk_index": 1,
                    "local_speaker": "A",
                    "speaker": "A",
                    "start": 0,
                    "end": 1,
                    "text": "ASR text.",
                }
            ],
        }
        speaker_mapping = {
            "speakers": [
                {"id": "speaker_asr", "label_short": "ASR", "label_full": "ASR"},
            ],
            "local_speakers": [
                {"chunk_index": 1, "local_speaker": "A", "speaker_id": "speaker_asr"},
            ],
        }

        def fake_translate(_client, _model, _url, title, _description, _target_language, speakers, turns, _hint, debug_sink=None):
            return {
                "title_translated": title,
                "speakers": speakers,
                "turns": [
                    {
                        "speaker_id": turn["speaker_id"],
                        "text_translated": turn["text_source"],
                    }
                    for turn in turns
                ],
            }

        with (
            patch.dict(
                "os.environ",
                {"OPENAI_API_KEY": "test-openai", "YOUTUBE_API_KEY": "test-youtube"},
            ),
            patch.object(ytranslate, "fetch_video_metadata", return_value={"title": "Video", "description": ""}),
            patch.object(ytranslate, "fetch_transcript", return_value=transcript_info),
            patch.object(ytranslate, "transcribe_youtube_audio_with_openai", return_value=asr_result) as asr_mock,
            patch.object(ytranslate, "assign_global_speakers_for_diarized_segments", return_value=speaker_mapping) as mapping_mock,
            patch.object(ytranslate, "translate_attributed_turns", side_effect=fake_translate),
            patch.object(ytranslate, "render_docx") as render_mock,
            patch.object(ytranslate, "convert_docx_to_pdf", return_value="/tmp/video.pdf"),
            patch.object(ytranslate, "send_completion_notification"),
            patch.object(ytranslate, "OpenAI"),
        ):
            return ytranslate.run_translation_job(
                "https://youtu.be/HGbA6ze0_3M",
                target_language="French",
                log=lambda _message: None,
            ), asr_mock, mapping_mock, render_mock

    def test_low_quality_youtube_transcript_uses_openai_asr(self):
        transcript_info = {
            "is_generated": True,
            "segments": [
                {"start": 0, "duration": 1, "text": "speakerless caption"},
            ],
        }

        _result, asr_mock, mapping_mock, render_mock = self.run_with_common_mocks(transcript_info)

        asr_mock.assert_called_once()
        mapping_mock.assert_called_once()
        rendered_turns = render_mock.call_args.args[2]
        self.assertEqual(rendered_turns[0]["speaker_id"], "speaker_asr")

    def test_speaker_labeled_youtube_transcript_skips_openai_asr(self):
        transcript_info = {
            "is_generated": False,
            "segments": [
                {"start": 0, "duration": 1, "text": "Host: Welcome."},
                {"start": 1, "duration": 1, "text": "Guest: Thanks."},
                {"start": 2, "duration": 1, "text": "Host: Question."},
                {"start": 3, "duration": 1, "text": "Guest: Answer."},
                {"start": 4, "duration": 1, "text": "Host: Follow-up."},
                {"start": 5, "duration": 1, "text": "Guest: Detail."},
                {"start": 6, "duration": 1, "text": "Host: Great."},
                {"start": 7, "duration": 1, "text": "Guest: Yes."},
            ],
        }

        _result, asr_mock, mapping_mock, render_mock = self.run_with_common_mocks(transcript_info)

        asr_mock.assert_not_called()
        mapping_mock.assert_not_called()
        rendered_speakers = render_mock.call_args.args[1]
        self.assertEqual(rendered_speakers[0]["label_short"], "Host")


if __name__ == "__main__":
    unittest.main()
