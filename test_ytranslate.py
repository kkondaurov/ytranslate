import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import ytranslate


class OpenAICallTests(unittest.TestCase):
    def test_call_openai_requires_explicit_json_schema(self):
        with self.assertRaisesRegex(RuntimeError, "JSON schema must be provided"):
            ytranslate.call_openai(Mock(), "gpt-test", "system", "user")


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

    def test_default_asr_chunk_length_is_ten_minutes_for_long_upload_reliability(self):
        self.assertEqual(ytranslate.ASR_CHUNK_SECONDS, 600)

    def test_asr_chunk_length_rejects_values_above_model_limit(self):
        with patch.dict("os.environ", {"OPENAI_ASR_CHUNK_SECONDS": "2700"}):
            with self.assertRaisesRegex(RuntimeError, "exceeds the OpenAI ASR model limit"):
                ytranslate.get_asr_chunk_seconds()

    def test_default_asr_request_timeout_is_shorter_than_text_generation_timeout(self):
        self.assertEqual(ytranslate.get_asr_timeout_seconds(), 600)
        self.assertLess(ytranslate.get_asr_timeout_seconds(), ytranslate.OPENAI_TIMEOUT_SECONDS)

    def test_openai_asr_request_logs_attempts_and_uses_retry_setting(self):
        class FakeResponse:
            status_code = 200
            text = "{}"

            def json(self):
                return {"segments": []}

        with tempfile.TemporaryDirectory() as tmp:
            chunk = Path(tmp) / "chunk-000.mp3"
            chunk.write_bytes(b"fake audio")
            messages = []

            with (
                patch.dict("os.environ", {"OPENAI_ASR_MAX_RETRIES": "2"}),
                patch.dict("os.environ", {"OPENAI_ASR_TRANSPORT": "requests"}),
                patch.object(
                    ytranslate.requests,
                    "post",
                    side_effect=[ytranslate.requests.Timeout("slow upload"), FakeResponse()],
                ) as post,
                patch.object(ytranslate.time, "sleep"),
            ):
                result = ytranslate.transcribe_audio_chunk(
                    str(chunk),
                    "test-key",
                    "gpt-4o-transcribe-diarize",
                    123,
                    log=messages.append,
                )

        self.assertEqual(result, {"segments": []})
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["timeout"], 123)
        self.assertIn(
            "OpenAI ASR request attempt 1/2 for chunk-000.mp3 via requests (timeout=123s)",
            messages,
        )
        self.assertTrue(
            any("retryable error on attempt 1/2 for chunk-000.mp3" in message for message in messages)
        )

    def test_openai_asr_curl_transport_keeps_api_key_out_of_process_args(self):
        completed = Mock(returncode=0, stdout='{"segments": []}', stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            chunk = Path(tmp) / "chunk-000.mp3"
            chunk.write_bytes(b"fake audio")

            with (
                patch.dict(
                    "os.environ",
                    {"OPENAI_ASR_TRANSPORT": "curl", "OPENAI_ASR_MAX_RETRIES": "1"},
                ),
                patch.object(ytranslate.shutil, "which", return_value="/usr/bin/curl"),
                patch.object(ytranslate.subprocess, "run", return_value=completed) as run,
            ):
                result = ytranslate.transcribe_audio_chunk(
                    str(chunk),
                    "test-key",
                    "gpt-4o-transcribe-diarize",
                    123,
                )

        self.assertEqual(result, {"segments": []})
        self.assertEqual(run.call_args.args[0], ["/usr/bin/curl", "--config", "-"])
        self.assertNotIn("test-key", " ".join(run.call_args.args[0]))
        self.assertIn('header = "Authorization: Bearer test-key"', run.call_args.kwargs["input"])
        self.assertIn("http1.1", run.call_args.kwargs["input"])
        self.assertIn("retry-all-errors", run.call_args.kwargs["input"])

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

    def test_openai_asr_reports_failed_chunks_after_caching_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            chunks = [
                cache_dir / "chunks-600s" / "chunk-000.mp3",
                cache_dir / "chunks-600s" / "chunk-001.mp3",
            ]
            chunks[0].parent.mkdir()
            for chunk in chunks:
                chunk.write_bytes(b"fake audio")

            calls = []
            messages = []

            def fake_transcribe(chunk_path, _openai_key, _asr_model, _timeout_seconds, log=None):
                calls.append((Path(chunk_path).name, _timeout_seconds))
                if Path(chunk_path).name == "chunk-000.mp3":
                    raise RuntimeError("upload failed")
                return {"segments": []}

            with (
                patch.dict(
                    "os.environ",
                    {"OPENAI_ASR_JOBS": "1", "OPENAI_ASR_MAX_PASSES": "1"},
                ),
                patch.object(ytranslate, "get_video_cache_dir", return_value=str(cache_dir)),
                patch.object(ytranslate, "download_youtube_audio", return_value=str(cache_dir / "source.m4a")),
                patch.object(ytranslate, "transcode_and_chunk_audio", return_value=[str(chunk) for chunk in chunks]),
                patch.object(ytranslate, "build_chunk_offsets", return_value=[0.0, 600.0]),
                patch.object(ytranslate, "transcribe_audio_chunk", side_effect=fake_transcribe),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "OpenAI ASR completed 1/2 chunks; failed: chunk-000.mp3",
                ):
                    ytranslate.transcribe_youtube_audio_with_openai(
                        "https://youtu.be/example",
                        "example",
                        "test-key",
                        log=messages.append,
                    )

            self.assertEqual(calls, [("chunk-000.mp3", 600), ("chunk-001.mp3", 600)])
            self.assertTrue(
                any("OpenAI ASR failed chunk-000.mp3 (1/2)" in message for message in messages)
            )
            self.assertTrue(
                (cache_dir / "openai-asr-chunks-gpt-4o-transcribe-diarize-600s" / "chunk-001.json").exists()
            )

    def test_openai_asr_retries_failed_chunks_in_later_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            chunks = [
                cache_dir / "chunks-600s" / "chunk-000.mp3",
                cache_dir / "chunks-600s" / "chunk-001.mp3",
            ]
            chunks[0].parent.mkdir()
            for chunk in chunks:
                chunk.write_bytes(b"fake audio")

            calls = []
            messages = []
            failed_once = False

            def fake_transcribe(chunk_path, _openai_key, _asr_model, _timeout_seconds, log=None):
                nonlocal failed_once
                chunk_name = Path(chunk_path).name
                calls.append(chunk_name)
                if chunk_name == "chunk-000.mp3" and not failed_once:
                    failed_once = True
                    raise RuntimeError("temporary ssl failure")
                return {"segments": []}

            with (
                patch.dict(
                    "os.environ",
                    {
                        "OPENAI_ASR_JOBS": "1",
                        "OPENAI_ASR_MAX_PASSES": "2",
                        "OPENAI_ASR_RETRY_PASS_DELAY_SECONDS": "0",
                    },
                ),
                patch.object(ytranslate, "get_video_cache_dir", return_value=str(cache_dir)),
                patch.object(ytranslate, "download_youtube_audio", return_value=str(cache_dir / "source.m4a")),
                patch.object(ytranslate, "transcode_and_chunk_audio", return_value=[str(chunk) for chunk in chunks]),
                patch.object(ytranslate, "build_chunk_offsets", return_value=[0.0, 600.0]),
                patch.object(ytranslate, "transcribe_audio_chunk", side_effect=fake_transcribe),
            ):
                result = ytranslate.transcribe_youtube_audio_with_openai(
                    "https://youtu.be/example",
                    "example",
                    "test-key",
                    log=messages.append,
                )

            self.assertEqual(calls, ["chunk-000.mp3", "chunk-001.mp3", "chunk-000.mp3"])
            self.assertEqual(len(result["chunks"]), 2)
            self.assertTrue(
                any(
                    "Retrying failed OpenAI ASR chunks (pass 2/2, 1 remaining)." in message
                    for message in messages
                )
            )
            self.assertTrue(
                (cache_dir / "openai-asr-chunks-gpt-4o-transcribe-diarize-600s" / "chunk-000.json").exists()
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

    def test_speaker_mapping_override_remaps_chunk_local_speaker_by_id(self):
        speaker_mapping = {
            "speakers": [
                {"id": "speaker_gavin", "label_short": "Gavin", "label_full": "Gavin Baker"},
                {"id": "speaker_chamath", "label_short": "Chamath", "label_full": "Chamath Palihapitiya"},
            ],
            "local_speakers": [
                {"chunk_index": 3, "local_speaker": "B", "speaker_id": "speaker_gavin"},
                {"chunk_index": 3, "local_speaker": "C", "speaker_id": "speaker_chamath"},
            ],
        }
        overrides = {
            "local_speakers": [
                {"chunk_index": 3, "local_speaker": "B", "speaker_id": "speaker_chamath"},
            ]
        }

        effective = ytranslate.apply_speaker_mapping_overrides(speaker_mapping, overrides)

        self.assertEqual(
            effective["local_speakers"],
            [
                {"chunk_index": 3, "local_speaker": "B", "speaker_id": "speaker_chamath"},
                {"chunk_index": 3, "local_speaker": "C", "speaker_id": "speaker_chamath"},
            ],
        )

    def test_speaker_mapping_override_can_resolve_existing_speaker_by_label(self):
        speaker_mapping = {
            "speakers": [
                {"id": "speaker_gavin", "label_short": "Gavin", "label_full": "Gavin Baker"},
                {"id": "speaker_chamath", "label_short": "Chamath", "label_full": "Chamath Palihapitiya"},
            ],
            "local_speakers": [
                {"chunk_index": 3, "local_speaker": "B", "speaker_id": "speaker_gavin"},
            ],
        }
        overrides = {
            "local_speakers": [
                {"chunk_index": 3, "local_speaker": "B", "speaker_label": "Chamath"},
            ]
        }

        effective = ytranslate.apply_speaker_mapping_overrides(speaker_mapping, overrides)

        self.assertEqual(effective["local_speakers"][0]["speaker_id"], "speaker_chamath")

    def test_voice_reconciliation_can_override_mixed_local_speaker_segments(self):
        segments = [
            {"chunk_index": 1, "local_speaker": "A", "speaker": "A", "start": 0, "end": 8, "text": "Alice anchor."},
            {"chunk_index": 1, "local_speaker": "B", "speaker": "B", "start": 10, "end": 18, "text": "Bob anchor."},
            {"chunk_index": 2, "local_speaker": "A", "speaker": "A", "start": 20, "end": 27, "text": "This is actually Bob."},
            {"chunk_index": 2, "local_speaker": "A", "speaker": "A", "start": 27.1, "end": 28, "text": "short continuation"},
            {"chunk_index": 2, "local_speaker": "B", "speaker": "B", "start": 30, "end": 37, "text": "This is actually Alice."},
        ]
        speaker_mapping = {
            "speakers": [
                {"id": "speaker_alice", "label_short": "Alice", "label_full": "Alice"},
                {"id": "speaker_bob", "label_short": "Bob", "label_full": "Bob"},
            ],
            "local_speakers": [
                {"chunk_index": 1, "local_speaker": "A", "speaker_id": "speaker_alice"},
                {"chunk_index": 1, "local_speaker": "B", "speaker_id": "speaker_bob"},
                {"chunk_index": 2, "local_speaker": "A", "speaker_id": "speaker_alice"},
                {"chunk_index": 2, "local_speaker": "B", "speaker_id": "speaker_bob"},
            ],
        }
        embeddings = {
            0: [1.0, 0.0],
            1: [0.0, 1.0],
            2: [0.02, 1.0],
            4: [1.0, 0.01],
        }

        resolved, debug = ytranslate.reconcile_segment_speakers_with_voice_embeddings(
            segments,
            speaker_mapping,
            embeddings,
            min_similarity=0.8,
            min_margin=0.2,
            neighbor_gap_seconds=1.0,
        )

        self.assertEqual(resolved[2]["speaker_id"], "speaker_bob")
        self.assertEqual(resolved[2]["speaker_id_source"], "voice")
        self.assertEqual(resolved[3]["speaker_id"], "speaker_bob")
        self.assertEqual(resolved[3]["speaker_id_source"], "voice_neighbor")
        self.assertEqual(resolved[4]["speaker_id"], "speaker_alice")
        self.assertEqual(debug["voice_changed_count"], 2)
        self.assertEqual(debug["neighbor_assigned_count"], 1)

        attributed = ytranslate.attributed_turns_from_diarized_segments(resolved, speaker_mapping)
        self.assertEqual(
            attributed["turns"],
            [
                {"speaker_id": "speaker_alice", "text_source": "Alice anchor."},
                {"speaker_id": "speaker_bob", "text_source": "Bob anchor. This is actually Bob. short continuation"},
                {"speaker_id": "speaker_alice", "text_source": "This is actually Alice."},
            ],
        )

    def test_voice_reconciliation_keeps_baseline_on_low_margin_match(self):
        segments = [
            {"chunk_index": 1, "local_speaker": "A", "speaker": "A", "start": 0, "end": 8, "text": "Alice anchor."},
            {"chunk_index": 1, "local_speaker": "B", "speaker": "B", "start": 10, "end": 18, "text": "Bob anchor."},
            {"chunk_index": 2, "local_speaker": "A", "speaker": "A", "start": 20, "end": 27, "text": "Ambiguous voice."},
        ]
        speaker_mapping = {
            "speakers": [
                {"id": "speaker_alice", "label_short": "Alice", "label_full": "Alice"},
                {"id": "speaker_bob", "label_short": "Bob", "label_full": "Bob"},
            ],
            "local_speakers": [
                {"chunk_index": 1, "local_speaker": "A", "speaker_id": "speaker_alice"},
                {"chunk_index": 1, "local_speaker": "B", "speaker_id": "speaker_bob"},
                {"chunk_index": 2, "local_speaker": "A", "speaker_id": "speaker_alice"},
            ],
        }
        embeddings = {
            0: [1.0, 0.0],
            1: [0.0, 1.0],
            2: [0.62, 0.78],
        }

        resolved, debug = ytranslate.reconcile_segment_speakers_with_voice_embeddings(
            segments,
            speaker_mapping,
            embeddings,
            min_similarity=0.5,
            min_margin=0.4,
        )

        self.assertEqual(resolved[2]["speaker_id"], "speaker_alice")
        self.assertEqual(resolved[2]["speaker_id_source"], "local_mapping")
        self.assertEqual(debug["voice_changed_count"], 0)


class RunTranslationJobSourceChoiceTests(unittest.TestCase):
    def run_with_common_mocks(self, transcript_info, speaker_overrides=None):
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

        def fake_translate(
            _client,
            _model,
            _url,
            title,
            _description,
            _target_language,
            speakers,
            turns,
            _hint,
            debug_sink=None,
            log=None,
        ):
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
            patch.object(ytranslate, "load_speaker_mapping_overrides", return_value=speaker_overrides),
            patch.object(
                ytranslate,
                "reconcile_diarized_segments_with_voice",
                side_effect=lambda _url, _video_id, segments, _speaker_mapping, _log: (segments, {"status": "test"}),
            ),
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

    def test_low_quality_asr_applies_speaker_mapping_overrides(self):
        transcript_info = {
            "is_generated": True,
            "segments": [
                {"start": 0, "duration": 1, "text": "speakerless caption"},
            ],
        }
        speaker_overrides = {
            "speakers": [
                {"id": "speaker_override", "label_short": "Override", "label_full": "Override"},
            ],
            "local_speakers": [
                {"chunk_index": 1, "local_speaker": "A", "speaker_id": "speaker_override"},
            ],
        }

        _result, _asr_mock, _mapping_mock, render_mock = self.run_with_common_mocks(
            transcript_info,
            speaker_overrides=speaker_overrides,
        )

        rendered_speakers = render_mock.call_args.args[1]
        rendered_turns = render_mock.call_args.args[2]
        self.assertIn(
            {"id": "speaker_override", "label_short": "Override", "label_full": "Override"},
            rendered_speakers,
        )
        self.assertEqual(rendered_turns[0]["speaker_id"], "speaker_override")

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


class TurnTextAlignmentTests(unittest.TestCase):
    def test_align_turn_texts_by_returned_turn_index(self):
        returned_turns = [
            {"turn_index": 2, "text_translated": "second"},
            {"turn_index": 1, "text_translated": "first"},
        ]

        self.assertEqual(
            ytranslate.align_turn_texts_by_index(returned_turns, 2, "translation"),
            ["first", "second"],
        )

    def test_align_turn_texts_rejects_missing_turn_index(self):
        returned_turns = [
            {"turn_index": 1, "text_translated": "first"},
            {"text_translated": "second"},
        ]

        with self.assertRaisesRegex(RuntimeError, "missing turn_index"):
            ytranslate.align_turn_texts_by_index(returned_turns, 2, "translation")

    def test_align_turn_texts_by_exact_turn_keys(self):
        returned_turns = {
            "turn_0002": "second",
            "turn_0001": "first",
        }

        self.assertEqual(
            ytranslate.align_turn_texts_by_key(returned_turns, 2, "translation"),
            ["first", "second"],
        )

    def test_align_turn_texts_by_key_rejects_missing_turn_key(self):
        returned_turns = {
            "turn_0001": "first",
            "turn_0003": "third",
        }

        with self.assertRaisesRegex(RuntimeError, "missing turn key turn_0002"):
            ytranslate.align_turn_texts_by_key(returned_turns, 3, "translation")

    def test_translate_attributed_turns_preserves_turn_order_from_keyed_results(self):
        turns = [
            {"speaker_id": "speaker_a", "text_source": "Alpha."},
            {"speaker_id": "speaker_a", "text_source": "Beta."},
        ]
        speakers = [
            {"id": "speaker_a", "label_short": "A", "label_full": "A"},
        ]
        result_from_model = {
            "title_translated": "Title",
            "translations": {
                "turn_0002": "Бета.",
                "turn_0001": "Альфа.",
            },
        }

        with patch.object(ytranslate, "translate_turn_chunk", return_value=result_from_model):
            result = ytranslate.translate_attributed_turns(
                client=Mock(),
                model="model",
                url="https://youtu.be/test",
                title="Title",
                description="",
                target_language="Russian",
                speakers=speakers,
                turns=turns,
                source_language_hint=None,
            )

        self.assertEqual(
            result["turns"],
            [
                {"speaker_id": "speaker_a", "text_translated": "Альфа."},
                {"speaker_id": "speaker_a", "text_translated": "Бета."},
            ],
        )

    def test_translate_attributed_turns_chunks_before_calling_model(self):
        turns = [
            {"speaker_id": "speaker_a", "text_source": "Alpha " * 10},
            {"speaker_id": "speaker_b", "text_source": "Beta " * 10},
            {"speaker_id": "speaker_c", "text_source": "Gamma " * 10},
        ]
        speakers = [
            {"id": "speaker_a", "label_short": "A", "label_full": "A"},
            {"id": "speaker_b", "label_short": "B", "label_full": "B"},
            {"id": "speaker_c", "label_short": "C", "label_full": "C"},
        ]

        def fake_translate_chunk(
            _client,
            _model,
            _url,
            _title,
            _description,
            _target_language,
            _speakers,
            chunk,
            _source_language_hint,
            debug_sink=None,
            chunk_index=1,
            chunk_count=1,
        ):
            return {
                "title_translated": "Title",
                "translations": {
                    ytranslate.turn_key(index): turn["text_source"].strip()
                    for index, turn in enumerate(chunk, 1)
                },
            }

        with (
            patch.object(ytranslate, "TURN_TEXT_PASS_MAX_CHARS", 70),
            patch.object(ytranslate, "translate_turn_chunk", side_effect=fake_translate_chunk) as chunk_mock,
        ):
            result = ytranslate.translate_attributed_turns(
                client=Mock(),
                model="model",
                url="https://youtu.be/test",
                title="Title",
                description="",
                target_language="Russian",
                speakers=speakers,
                turns=turns,
                source_language_hint=None,
            )

        self.assertGreater(chunk_mock.call_count, 1)
        self.assertEqual(
            [turn["speaker_id"] for turn in result["turns"]],
            ["speaker_a", "speaker_b", "speaker_c"],
        )

    def test_source_turn_chunks_group_by_speaker(self):
        turns = [
            {"speaker_id": "speaker_friedberg", "text_source": "Friedberg first."},
            {"speaker_id": "speaker_chamath", "text_source": "Chamath interjection."},
            {"speaker_id": "speaker_friedberg", "text_source": "Friedberg continues."},
        ]

        chunks = ytranslate.chunk_source_turns(turns, max_chars=10_000)

        self.assertEqual(chunks, [[turns[0], turns[2]], [turns[1]]])

    def test_translate_attributed_turns_reassembles_speaker_grouped_chunks_in_original_order(self):
        turns = [
            {"speaker_id": "speaker_friedberg", "text_source": "Friedberg first."},
            {"speaker_id": "speaker_chamath", "text_source": "Chamath interjection."},
            {"speaker_id": "speaker_friedberg", "text_source": "Friedberg continues."},
        ]
        speakers = [
            {"id": "speaker_friedberg", "label_short": "Friedberg", "label_full": "David Friedberg"},
            {"id": "speaker_chamath", "label_short": "Chamath", "label_full": "Chamath Palihapitiya"},
        ]
        seen_chunks = []

        def fake_translate_chunk(
            _client,
            _model,
            _url,
            _title,
            _description,
            _target_language,
            _speakers,
            chunk,
            _source_language_hint,
            debug_sink=None,
            chunk_index=1,
            chunk_count=1,
        ):
            seen_chunks.append([turn["speaker_id"] for turn in chunk])
            return {
                "title_translated": "Title",
                "translations": {
                    ytranslate.turn_key(index): f"translated:{turn['text_source']}"
                    for index, turn in enumerate(chunk, 1)
                },
            }

        with patch.object(ytranslate, "translate_turn_chunk", side_effect=fake_translate_chunk):
            result = ytranslate.translate_attributed_turns(
                client=Mock(),
                model="model",
                url="https://youtu.be/test",
                title="Title",
                description="",
                target_language="Russian",
                speakers=speakers,
                turns=turns,
                source_language_hint=None,
            )

        self.assertEqual(
            seen_chunks,
            [
                ["speaker_friedberg", "speaker_friedberg"],
                ["speaker_chamath"],
            ],
        )
        self.assertEqual(
            result["turns"],
            [
                {"speaker_id": "speaker_friedberg", "text_translated": "translated:Friedberg first."},
                {"speaker_id": "speaker_chamath", "text_translated": "translated:Chamath interjection."},
                {"speaker_id": "speaker_friedberg", "text_translated": "translated:Friedberg continues."},
            ],
        )

    def test_cleanup_russian_turns_chunks_before_calling_model(self):
        turns = [
            {"speaker_id": "speaker_a", "text_translated": "первый " * 10},
            {"speaker_id": "speaker_b", "text_translated": "второй " * 10},
            {"speaker_id": "speaker_c", "text_translated": "третий " * 10},
        ]

        def fake_cleanup_chunk(
            _client,
            _model,
            _title_translated,
            chunk,
            chunk_index=1,
            chunk_count=1,
            debug_sink=None,
        ):
            return [turn["text_translated"].strip() for turn in chunk]

        with (
            patch.object(ytranslate, "TURN_TEXT_PASS_MAX_CHARS", 80),
            patch.object(ytranslate, "cleanup_russian_turn_chunk", side_effect=fake_cleanup_chunk) as chunk_mock,
        ):
            result = ytranslate.cleanup_russian_turns(
                client=Mock(),
                model="model",
                title_translated="Title",
                turns=turns,
            )

        self.assertGreater(chunk_mock.call_count, 1)
        self.assertEqual(
            [turn["speaker_id"] for turn in result],
            ["speaker_a", "speaker_b", "speaker_c"],
        )

    def test_translated_turn_chunks_group_by_speaker(self):
        turns = [
            {"speaker_id": "speaker_friedberg", "text_translated": "Фридберг начал."},
            {"speaker_id": "speaker_chamath", "text_translated": "Чамат перебил."},
            {"speaker_id": "speaker_friedberg", "text_translated": "Фридберг продолжил."},
        ]

        chunks = ytranslate.chunk_turns_by_chars(turns, max_chars=10_000)

        self.assertEqual(chunks, [[turns[0], turns[2]], [turns[1]]])

    def test_cleanup_russian_turns_reassembles_speaker_grouped_chunks_in_original_order(self):
        turns = [
            {"speaker_id": "speaker_friedberg", "text_translated": "Фридберг начал."},
            {"speaker_id": "speaker_chamath", "text_translated": "Чамат перебил."},
            {"speaker_id": "speaker_friedberg", "text_translated": "Фридберг продолжил."},
        ]
        seen_chunks = []

        def fake_cleanup_chunk(
            _client,
            _model,
            _title_translated,
            chunk,
            chunk_index=1,
            chunk_count=1,
            debug_sink=None,
        ):
            seen_chunks.append([turn["speaker_id"] for turn in chunk])
            return [f"cleaned:{turn['text_translated']}" for turn in chunk]

        with patch.object(ytranslate, "cleanup_russian_turn_chunk", side_effect=fake_cleanup_chunk):
            result = ytranslate.cleanup_russian_turns(
                client=Mock(),
                model="model",
                title_translated="Title",
                turns=turns,
            )

        self.assertEqual(
            seen_chunks,
            [
                ["speaker_friedberg", "speaker_friedberg"],
                ["speaker_chamath"],
            ],
        )
        self.assertEqual(
            result,
            [
                {"speaker_id": "speaker_friedberg", "text_translated": "cleaned:Фридберг начал."},
                {"speaker_id": "speaker_chamath", "text_translated": "cleaned:Чамат перебил."},
                {"speaker_id": "speaker_friedberg", "text_translated": "cleaned:Фридберг продолжил."},
            ],
        )

    def test_annotate_russian_turns_chunks_before_calling_model(self):
        turns = [
            {"speaker_id": "speaker_a", "text_translated": "первый " * 10},
            {"speaker_id": "speaker_b", "text_translated": "второй " * 10},
            {"speaker_id": "speaker_c", "text_translated": "третий " * 10},
        ]

        def fake_annotate_chunk(
            _client,
            _model,
            _title_translated,
            chunk,
            chunk_index=1,
            chunk_count=1,
            debug_sink=None,
        ):
            return [turn["text_translated"].strip() for turn in chunk]

        with (
            patch.object(ytranslate, "TURN_TEXT_PASS_MAX_CHARS", 80),
            patch.object(ytranslate, "annotate_russian_turn_chunk", side_effect=fake_annotate_chunk) as chunk_mock,
        ):
            result = ytranslate.annotate_russian_turns(
                client=Mock(),
                model="model",
                title_translated="Title",
                turns=turns,
            )

        self.assertGreater(chunk_mock.call_count, 1)
        self.assertEqual(
            [turn["speaker_id"] for turn in result],
            ["speaker_a", "speaker_b", "speaker_c"],
        )

    def test_cleanup_russian_turn_chunk_aligns_returned_texts_by_turn_key(self):
        turns = [
            {"speaker_id": "speaker_a", "text_translated": "первый"},
            {"speaker_id": "speaker_b", "text_translated": "второй"},
        ]
        result_from_model = {
            "turns": {
                "turn_0002": "второй чистый",
                "turn_0001": "первый чистый",
            },
        }

        with patch.object(ytranslate, "call_openai_with_retry", return_value=result_from_model):
            self.assertEqual(
                ytranslate.cleanup_russian_turn_chunk(
                    client=Mock(),
                    model="model",
                    title_translated="Заголовок",
                    turns=turns,
                ),
                ["первый чистый", "второй чистый"],
            )

    def test_annotate_russian_turn_chunk_aligns_returned_texts_by_turn_key(self):
        turns = [
            {"speaker_id": "speaker_a", "text_translated": "первый"},
            {"speaker_id": "speaker_b", "text_translated": "второй"},
        ]
        result_from_model = {
            "turns": {
                "turn_0002": "второй с пояснением",
                "turn_0001": "первый с пояснением",
            },
        }

        with patch.object(ytranslate, "call_openai_with_retry", return_value=result_from_model):
            self.assertEqual(
                ytranslate.annotate_russian_turn_chunk(
                    client=Mock(),
                    model="model",
                    title_translated="Заголовок",
                    turns=turns,
                ),
                ["первый с пояснением", "второй с пояснением"],
            )


if __name__ == "__main__":
    unittest.main()
