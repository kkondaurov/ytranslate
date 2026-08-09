import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.turn_constrained_diarization import turn_constrained_eval as experiment


class TurnConstrainedEvalTests(unittest.TestCase):
    def test_all_in_reference_banks_have_three_reviewed_clips_per_identity(self):
        for identity, references in experiment.TRUSTED_HOST_REFERENCES.items():
            with self.subTest(identity=identity):
                self.assertEqual(len(references), 3)
                self.assertTrue(all(reference.source_video_id for reference in references))

        friedberg_spans = {
            (reference.source_video_id, reference.start, reference.end)
            for reference in experiment.TRUSTED_HOST_REFERENCES["friedberg"]
        }
        self.assertNotIn(("wcV0SRPFK9s", 372.18, 380.73), friedberg_spans)

    def test_core_holdout_ground_truth_stops_at_reviewed_handoff(self):
        episode = next(
            item for item in experiment.EPISODES if item.key == "core-four-holdout"
        )
        self.assertLessEqual(max(window.end for window in episode.windows), 2724.218)

    def test_archive_panel_uses_independent_four_speaker_references(self):
        episode = next(
            item
            for item in experiment.EPISODES
            if item.key == "uncapped-founders-fund-panel"
        )
        self.assertEqual(
            {reference.name for reference in episode.references},
            {"Jack Altman", "Ev Randle", "Trae Stephens", "Delian Asparouhov"},
        )
        self.assertFalse(episode.auto_reference_seed)

    def test_external_windows_are_excluded_from_comparable_host_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.json"
            audit_path.write_text(
                json.dumps(
                    [
                        {"start": 0.0, "end": 5.0, "mapped_speaker_label": "Host"},
                        {"start": 5.0, "end": 10.0, "mapped_speaker_label": "Host"},
                    ]
                ),
                encoding="utf-8",
            )
            episode = experiment.Episode(
                key="test",
                video_id="test",
                title="Test",
                context="Test",
                asr_filename="asr.json",
                references=(),
                audit_path=audit_path,
                windows=(
                    experiment.AttributionWindow(
                        "External clip",
                        5.0,
                        10.0,
                        experiment.SIL_UNKNOWN_LABEL,
                    ),
                ),
            )
            metrics = experiment.evaluate_result(
                episode,
                [
                    {"start": 0.0, "end": 5.0, "speaker_label": "Host"},
                    {
                        "start": 5.0,
                        "end": 10.0,
                        "speaker_label": experiment.SIL_UNKNOWN_LABEL,
                    },
                ],
            )

        self.assertEqual(metrics["audit"]["duration_accuracy"], 0.5)
        self.assertEqual(metrics["known_host_audit"]["duration_accuracy"], 1.0)

    def test_auto_reference_selection_uses_confident_separated_clips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            cache_dir = cache_root / "episode"
            cache_dir.mkdir()
            (cache_dir / "speaker-mapping-effective.json").write_text(
                json.dumps(
                    {
                        "speakers": [
                            {"id": "speaker_1", "label_short": "Host"},
                            {"id": "speaker_2", "label_short": "Guest"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (cache_dir / "openai-asr-resolved-segments.json").write_text(
                json.dumps(
                    [
                        {
                            "speaker_id": "speaker_1",
                            "start": 10.0,
                            "end": 14.0,
                            "voice_similarity": 0.95,
                            "voice_similarity_margin": 0.2,
                        },
                        {
                            "speaker_id": "speaker_1",
                            "start": 20.0,
                            "end": 24.0,
                            "voice_similarity": 0.96,
                            "voice_similarity_margin": 0.19,
                        },
                        {
                            "speaker_id": "speaker_1",
                            "start": 40.0,
                            "end": 44.0,
                            "voice_similarity": 0.94,
                            "voice_similarity_margin": 0.18,
                        },
                        {
                            "speaker_id": "speaker_2",
                            "start": 50.0,
                            "end": 55.0,
                            "voice_similarity": 0.91,
                            "voice_similarity_margin": 0.1,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            episode = experiment.Episode(
                key="test",
                video_id="episode",
                title="Test",
                context="Test",
                asr_filename="asr.json",
                references=(),
                auto_reference_seed=True,
            )
            original_cache_root = experiment.CACHE_ROOT
            experiment.CACHE_ROOT = cache_root
            try:
                references = experiment.select_cached_speaker_references(episode)
            finally:
                experiment.CACHE_ROOT = original_cache_root

        self.assertEqual(
            [(reference.name, reference.start) for reference in references],
            [("Host", 10.0), ("Host", 40.0), ("Guest", 50.0)],
        )

    def test_explicit_cross_episode_references_do_not_auto_enroll(self):
        reference = experiment.SpeakerReference("Host", 1.0, 4.0, "source-video")
        episode = experiment.Episode(
            key="test",
            video_id="holdout",
            title="Test",
            context="Test",
            asr_filename="asr.json",
            references=(reference,),
            auto_reference_seed=True,
        )
        self.assertEqual(experiment.resolve_speaker_references(episode), (reference,))

    def test_supplemental_auto_enrollment_adds_only_active_missing_speakers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            cache_dir = cache_root / "episode"
            cache_dir.mkdir()
            (cache_dir / "speaker-mapping-effective.json").write_text(
                json.dumps(
                    {
                        "speakers": [
                            {"id": "speaker_1", "label_short": "Host"},
                            {"id": "speaker_2", "label_short": "Guest"},
                            {"id": "speaker_3", "label_short": "Absent"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            baseline = [
                {
                    "speaker_id": "speaker_1",
                    "speaker_label": "Host",
                    "start": 0.0,
                    "end": 4.0,
                },
                {
                    "speaker_id": "speaker_2",
                    "speaker_label": "Guest",
                    "start": 20.0,
                    "end": 25.0,
                },
            ]
            explicit = experiment.SpeakerReference("Host", 1.0, 4.0, "trusted")
            episode = experiment.Episode(
                key="test",
                video_id="episode",
                title="Test",
                context="Test",
                asr_filename="asr.json",
                references=(explicit,),
                auto_reference_seed=True,
                supplement_auto_references=True,
            )
            original_cache_root = experiment.CACHE_ROOT
            experiment.CACHE_ROOT = cache_root
            try:
                references = experiment.resolve_speaker_references(episode, baseline)
            finally:
                experiment.CACHE_ROOT = original_cache_root

        self.assertEqual(
            [(reference.name, reference.start) for reference in references],
            [("Host", 1.0), ("Guest", 20.0)],
        )

    def test_display_labels_follow_the_baseline_identity_spelling(self):
        baseline = [
            {
                "speaker_id": "speaker_chamath_palihapitiya",
                "speaker_label": "Chamath",
            }
        ]
        repaired = [
            {
                "speaker_id": "speaker_chamath_palihapitiya",
                "speaker_label": "Chamath Palihapitiya",
            }
        ]
        normalized = experiment.normalize_speaker_display_labels(repaired, baseline)
        self.assertEqual(normalized[0]["speaker_label"], "Chamath")

    def test_boundary_schema_has_exact_count(self):
        schema = experiment.boundary_schema(17)
        boundaries = schema["properties"]["boundaries"]
        self.assertEqual(boundaries["minItems"], 17)
        self.assertEqual(boundaries["maxItems"], 17)

    def test_boundary_validation_rejects_missing_ids(self):
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            experiment.validate_boundary_assignments(
                [{"segment_id": 0, "boundary_before": "change"}],
                range(2),
            )

    def test_split_span_keeps_units_short_and_tail_embeddable(self):
        spans = experiment.split_span(0.0, 13.0, 6.0)
        self.assertEqual(spans[0], (0.0, 6.0))
        self.assertTrue(all(end - start <= 6.0 for start, end in spans))
        self.assertGreaterEqual(spans[-1][1] - spans[-1][0], 1.5)
        self.assertEqual(spans[-1][1], 13.0)

    def test_embedding_units_encode_must_and_cannot_links(self):
        segments = [
            {"start": 0.0, "end": 4.0, "text": "A"},
            {"start": 4.0, "end": 8.0, "text": "B"},
            {"start": 8.0, "end": 10.0, "text": "C"},
        ]
        units = experiment.build_embedding_units(
            segments, ["change", "same", "change"]
        )
        self.assertEqual([unit["anonymous_turn_id"] for unit in units], [0, 0, 1])
        matrix = experiment.turn_constraint_matrix(units)
        self.assertEqual(matrix[0, 1], 1)
        self.assertEqual(matrix[1, 2], -1)

    def test_embedding_units_can_respect_raw_local_handoffs(self):
        segments = [
            {"chunk_index": 1, "local_speaker": "A", "start": 0.0, "end": 3.0},
            {"chunk_index": 1, "local_speaker": "B", "start": 3.1, "end": 6.1},
        ]
        units = experiment.build_embedding_units(
            segments, ["change", "same"], respect_local_changes=True
        )
        self.assertEqual([unit["anonymous_turn_id"] for unit in units], [0, 1])

    def test_sil_units_split_on_local_identity_even_when_luna_says_same(self):
        segments = [
            {"chunk_index": 1, "local_speaker": "A", "start": 0.0, "end": 3.0},
            {"chunk_index": 1, "local_speaker": "B", "start": 3.1, "end": 6.1},
        ]
        units = experiment.build_sil_units(segments, ["change", "same"])
        self.assertEqual(len(units), 2)
        self.assertTrue(units[1]["local_change_before"])

    def test_sil_units_do_not_lock_a_long_predicted_turn_to_one_identity(self):
        segments = [
            {"chunk_index": 1, "local_speaker": "A", "start": 0.0, "end": 14.0},
        ]
        units = experiment.build_sil_units(segments, ["change"])
        self.assertEqual(len(units), 3)
        self.assertTrue(all(unit["segment_ids"] == [0] for unit in units))

    def test_consolidation_uses_duration_weighted_majority(self):
        units = [
            {"anonymous_turn_id": 0, "duration": 5.0},
            {"anonymous_turn_id": 0, "duration": 1.0},
            {"anonymous_turn_id": 1, "duration": 2.0},
        ]
        resolved = experiment.consolidate_anonymous_turn_clusters([2, 1, 1], units)
        self.assertEqual(resolved.tolist(), [2, 2, 1])

    def test_hungarian_mapping_is_one_to_one(self):
        centroids = {
            0: np.asarray([1.0, 0.0, 0.0]),
            1: np.asarray([0.0, 1.0, 0.0]),
        }
        references = {
            "Jason Calacanis": np.asarray([0.99, 0.01, 0.0]),
            "David Sacks": np.asarray([0.01, 0.99, 0.0]),
        }
        mapping, debug = experiment.map_clusters_to_references(
            centroids, references, minimum_similarity=0.5, minimum_margin=0.1
        )
        self.assertEqual(mapping[0], "Jason Calacanis")
        self.assertEqual(mapping[1], "David Sacks")
        self.assertTrue(all(item["accepted"] for item in debug["assignments"]))

    def test_reference_classifier_keeps_each_anonymous_turn_together(self):
        embeddings = np.asarray(
            [
                [0.99, 0.01],
                [0.90, 0.10],
                [0.05, 0.95],
            ],
            dtype=np.float32,
        )
        units = [
            {"anonymous_turn_id": 0, "duration": 5.0, "start": 0.0, "end": 5.0},
            {"anonymous_turn_id": 0, "duration": 1.0, "start": 5.0, "end": 6.0},
            {"anonymous_turn_id": 1, "duration": 3.0, "start": 6.0, "end": 9.0},
        ]
        references = {
            "Jason": np.asarray([1.0, 0.0], dtype=np.float32),
            "Sacks": np.asarray([0.0, 1.0], dtype=np.float32),
        }
        labels, names, debug = experiment.classify_anonymous_turns_by_reference(
            embeddings, units, references
        )
        self.assertEqual(labels.tolist(), [0, 0, 1])
        self.assertEqual(names, {0: "Jason", 1: "Sacks"})
        self.assertEqual(debug["turn_count"], 2)

    def test_sil_decoder_follows_sustained_acoustics_despite_same_boundary(self):
        units = [
            {
                "unit_id": 0,
                "start": 0.0,
                "end": 3.0,
                "duration": 3.0,
                "chunk_index": 1,
                "local_speaker": "A",
                "boundary_before": "change",
                "local_change_before": False,
                "chunk_change_before": False,
                "segment_ids": [0],
            },
            {
                "unit_id": 1,
                "start": 3.0,
                "end": 6.0,
                "duration": 3.0,
                "chunk_index": 1,
                "local_speaker": "A",
                "boundary_before": "same",
                "local_change_before": False,
                "chunk_change_before": False,
                "segment_ids": [1],
            },
            {
                "unit_id": 2,
                "start": 6.0,
                "end": 9.0,
                "duration": 3.0,
                "chunk_index": 1,
                "local_speaker": "A",
                "boundary_before": "same",
                "local_change_before": False,
                "chunk_change_before": False,
                "segment_ids": [2],
            },
        ]
        scores = np.asarray(
            [[0.92, 0.70], [0.72, 0.91], [0.70, 0.93]], dtype=np.float32
        )
        labels, debug = experiment.decode_sil_identities(
            scores, units, ["Jason", "Brad"]
        )
        self.assertEqual(labels.tolist(), [0, 1, 1])
        self.assertEqual(debug["assignments"][1]["speaker"], "Brad")

    def test_sil_decoder_reports_negative_margin_when_continuity_beats_acoustics(self):
        units = [
            {
                "unit_id": 0,
                "start": 0.0,
                "end": 3.0,
                "duration": 3.0,
                "chunk_index": 1,
                "local_speaker": "A",
                "boundary_before": "change",
                "local_change_before": False,
                "chunk_change_before": False,
                "segment_ids": [0],
            },
            {
                "unit_id": 1,
                "start": 3.0,
                "end": 6.0,
                "duration": 3.0,
                "chunk_index": 1,
                "local_speaker": "A",
                "boundary_before": "same",
                "local_change_before": False,
                "chunk_change_before": False,
                "segment_ids": [1],
            },
        ]
        scores = np.asarray(
            [[0.70, 0.69, 0.76], [0.77, 0.70, 0.76]], dtype=np.float32
        )
        labels, debug = experiment.decode_sil_identities(
            scores,
            units,
            ["Jason", "Brad", experiment.SIL_UNKNOWN_LABEL],
            known_speaker_count=2,
        )
        self.assertEqual(labels.tolist(), [2, 2])
        self.assertEqual(debug["assignments"][1]["acoustic_winner"], "Jason")
        self.assertAlmostEqual(debug["assignments"][1]["margin"], -0.01, places=3)

    def test_sil_pooling_joins_same_track_until_a_real_boundary_hint(self):
        scores = np.asarray(
            [[0.9, 0.7], [0.88, 0.72], [0.7, 0.91]], dtype=np.float32
        )
        units = [
            {
                "unit_id": 0,
                "start": 0.0,
                "end": 3.0,
                "duration": 3.0,
                "chunk_index": 1,
                "local_speaker": "A",
                "boundary_before": "change",
                "local_change_before": False,
                "chunk_change_before": False,
                "segment_ids": [0],
            },
            {
                "unit_id": 1,
                "start": 3.1,
                "end": 6.1,
                "duration": 3.0,
                "chunk_index": 1,
                "local_speaker": "A",
                "boundary_before": "same",
                "local_change_before": False,
                "chunk_change_before": False,
                "segment_ids": [1],
            },
            {
                "unit_id": 2,
                "start": 6.2,
                "end": 9.2,
                "duration": 3.0,
                "chunk_index": 1,
                "local_speaker": "B",
                "boundary_before": "same",
                "local_change_before": True,
                "chunk_change_before": False,
                "segment_ids": [2],
            },
        ]
        pooled, groups, indexes = experiment.pool_sil_units(scores, units)
        self.assertEqual(pooled.shape, (2, 2))
        self.assertEqual(indexes, [[0, 1], [2]])
        self.assertEqual(groups[0]["segment_ids"], [0, 1])

    def test_sil_segment_assignment_uses_unit_duration_votes(self):
        segments = [{"start": 0.0, "end": 6.0, "text": "hello"}]
        units = [
            {"start": 0.0, "end": 2.0, "segment_ids": [0]},
            {"start": 2.0, "end": 6.0, "segment_ids": [0]},
        ]
        resolved = experiment.assign_segments_from_sil_units(
            segments, units, [0, 1], ["Jason", "Brad"], ["change"]
        )
        self.assertEqual(resolved[0]["speaker_label"], "Brad")
        self.assertEqual(resolved[0]["speaker_id_source"], "sil-global-acoustic")

    def test_episode_scale_unit_repair_does_not_require_target_self_anchor(self):
        baseline = [
            {
                "start": index * 10.0,
                "end": (index + 1) * 10.0,
                "speaker_id": "friedberg",
                "speaker_label": "Friedberg",
            }
            for index in range(3)
        ]
        assignments = [
            {
                "unit_id": index,
                "start": index * 10.0,
                "end": (index + 1) * 10.0,
                "duration": 10.0,
                "speaker": "David Sacks",
                "similarity": 0.91,
                "margin": 0.08,
                "chunk_index": 1,
                "local_speaker": "A",
                "segment_ids": [index],
            }
            for index in range(3)
        ]
        resolved, debug = experiment.apply_episode_scale_unit_repairs(
            baseline, assignments, ["change", "same", "same"]
        )
        self.assertEqual([item["speaker_label"] for item in resolved], ["David Sacks"] * 3)
        self.assertIn("friedberg->sacks", debug["activated_pairs"])
        self.assertEqual(len(debug["activated_scopes"]), 1)

    def test_episode_scale_unit_repair_does_not_leak_across_chunks(self):
        baseline = [
            {
                "start": start,
                "end": start + 10.0,
                "speaker_id": "friedberg",
                "speaker_label": "Friedberg",
            }
            for start in [0.0, 10.0, 20.0, 120.0]
        ]
        assignments = [
            {
                "unit_id": index,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "duration": 10.0,
                "speaker": "David Sacks",
                "similarity": 0.91,
                "margin": 0.03,
                "chunk_index": 1 if index < 3 else 2,
                "local_speaker": "A",
                "segment_ids": [index],
            }
            for index, segment in enumerate(baseline)
        ]

        resolved, debug = experiment.apply_episode_scale_unit_repairs(
            baseline, assignments, ["change", "same", "same", "change"]
        )

        self.assertEqual(
            [item["speaker_label"] for item in resolved],
            ["David Sacks", "David Sacks", "David Sacks", "Friedberg"],
        )
        self.assertEqual(
            [item["chunk_indexes"] for item in debug["activated_scopes"]],
            [[1]],
        )

    def test_episode_scale_unit_repair_does_not_accumulate_disjoint_passages(self):
        baseline = [
            {
                "start": start,
                "end": start + 6.0,
                "speaker_id": "jack",
                "speaker_label": "Jack Altman",
            }
            for start in [0.0, 80.0, 160.0, 240.0]
        ]
        assignments = [
            {
                "unit_id": index,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "duration": 6.0,
                "speaker": "Ev Randle",
                "similarity": 0.91,
                "margin": 0.03,
                "chunk_index": 2,
                "local_speaker": "C",
                "segment_ids": [index],
            }
            for index, segment in enumerate(baseline)
        ]

        resolved, debug = experiment.apply_episode_scale_unit_repairs(
            baseline, assignments, ["change", "change", "change", "change"]
        )

        self.assertEqual(
            [item["speaker_label"] for item in resolved],
            ["Jack Altman"] * 4,
        )
        self.assertIn("jack altman->ev randle", debug["activated_pairs"])
        self.assertEqual(debug["activated_scopes"], [])

    def test_short_unit_inside_rejected_run_cannot_borrow_global_activation(self):
        starts_and_durations = [
            (0.0, 10.0),
            (10.0, 10.0),
            (20.0, 10.0),
            (100.0, 4.2),
            (104.2, 2.8),
            (107.0, 4.0),
        ]
        baseline = [
            {
                "start": start,
                "end": start + duration,
                "speaker_id": "jack",
                "speaker_label": "Jack Altman",
            }
            for start, duration in starts_and_durations
        ]
        assignments = [
            {
                "unit_id": index,
                "start": start,
                "end": start + duration,
                "duration": duration,
                "speaker": "Ev Randle",
                "similarity": 0.90,
                "margin": 0.08 if index < 3 else [0.03, 0.06, 0.01][index - 3],
                "chunk_index": 1,
                "local_speaker": "A" if index < 3 else "C",
                "segment_ids": [index],
            }
            for index, (start, duration) in enumerate(starts_and_durations)
        ]

        resolved, debug = experiment.apply_episode_scale_unit_repairs(
            baseline,
            assignments,
            ["change", "same", "same", "change", "same", "same"],
        )

        self.assertEqual(
            [item["speaker_label"] for item in resolved],
            ["Ev Randle"] * 3 + ["Jack Altman"] * 3,
        )
        self.assertIn("jack altman->ev randle", debug["activated_pairs"])

    def test_direct_repair_preserves_a_mixed_unit_minority_segment(self):
        baseline = [
            {
                "start": start,
                "end": end,
                "speaker_id": speaker.lower(),
                "speaker_label": speaker,
            }
            for start, end, speaker in [
                (0.0, 10.0, "Friedberg"),
                (10.0, 20.0, "Friedberg"),
                (20.0, 29.0, "Friedberg"),
                (29.0, 30.0, "Chamath"),
            ]
        ]
        assignments = [
            {
                "unit_id": index,
                "start": start,
                "end": end,
                "duration": end - start,
                "speaker": "David Sacks",
                "similarity": 0.92,
                "margin": 0.10,
                "chunk_index": 1,
                "local_speaker": "A",
                "segment_ids": segment_ids,
            }
            for index, (start, end, segment_ids) in enumerate(
                [(0.0, 10.0, [0]), (10.0, 20.0, [1]), (20.0, 30.0, [2, 3])]
            )
        ]

        resolved, _ = experiment.apply_episode_scale_unit_repairs(
            baseline, assignments, ["change", "same", "same", "change"]
        )

        self.assertEqual(
            [item["speaker_label"] for item in resolved],
            ["David Sacks", "David Sacks", "David Sacks", "Chamath"],
        )

    def test_window_evaluation_tolerates_bounded_crosstalk(self):
        segments = [
            {
                "start": 0.0,
                "end": 9.1,
                "speaker_id": "friedberg",
                "speaker_label": "Friedberg",
            },
            {
                "start": 9.1,
                "end": 10.0,
                "speaker_id": "chamath",
                "speaker_label": "Chamath",
            },
        ]
        result = experiment.evaluate_windows(
            segments,
            [experiment.AttributionWindow("mostly Friedberg", 0.0, 10.0, "Friedberg")],
        )

        self.assertEqual(result["windows"][0]["accuracy"], 0.91)
        self.assertTrue(result["windows"][0]["passed"])

    def test_episode_scale_unit_repair_emits_only_sustained_unknown(self):
        baseline = [
            {
                "start": 0.0,
                "end": 7.0,
                "speaker_id": "sacks",
                "speaker_label": "Sacks",
            },
            {
                "start": 7.0,
                "end": 14.0,
                "speaker_id": "sacks",
                "speaker_label": "Sacks",
            },
        ]
        assignments = [
            {
                "unit_id": index,
                "start": index * 7.0,
                "end": (index + 1) * 7.0,
                "duration": 7.0,
                "speaker": experiment.SIL_UNKNOWN_LABEL,
                "similarity": experiment.SIL_UNKNOWN_STATE_SCORE,
                "margin": 0.04,
                "best_known_similarity": 0.72,
                "segment_ids": [index],
            }
            for index in range(2)
        ]
        resolved, debug = experiment.apply_episode_scale_unit_repairs(
            baseline, assignments, ["change", "same"]
        )
        self.assertEqual(
            [item["speaker_label"] for item in resolved],
            [experiment.SIL_UNKNOWN_LABEL] * 2,
        )
        self.assertTrue(debug["unknown_runs"][0]["accepted"])

    def test_episode_scale_unit_repair_preserves_short_unknown(self):
        baseline = [
            {
                "start": 0.0,
                "end": 3.0,
                "speaker_id": "jason",
                "speaker_label": "Jason",
            }
        ]
        assignments = [
            {
                "unit_id": 0,
                "start": 0.0,
                "end": 3.0,
                "duration": 3.0,
                "speaker": experiment.SIL_UNKNOWN_LABEL,
                "similarity": experiment.SIL_UNKNOWN_STATE_SCORE,
                "margin": 0.04,
                "best_known_similarity": 0.72,
                "segment_ids": [0],
            }
        ]
        resolved, debug = experiment.apply_episode_scale_unit_repairs(
            baseline, assignments, ["change"]
        )
        self.assertEqual(resolved[0]["speaker_label"], "Jason")
        self.assertFalse(debug["unknown_runs"][0]["accepted"])

    def test_sustained_unknown_uses_run_evidence_not_one_noisy_peak(self):
        baseline = [
            {
                "start": index * 5.0,
                "end": (index + 1) * 5.0,
                "speaker_id": "host",
                "speaker_label": "Host",
            }
            for index in range(3)
        ]
        assignments = [
            {
                "unit_id": index,
                "start": index * 5.0,
                "end": (index + 1) * 5.0,
                "duration": 5.0,
                "speaker": experiment.SIL_UNKNOWN_LABEL,
                "similarity": experiment.SIL_UNKNOWN_STATE_SCORE,
                "margin": 0.04,
                "best_known_similarity": 0.77 if index == 1 else 0.70,
                "segment_ids": [index],
            }
            for index in range(3)
        ]

        resolved, debug = experiment.apply_episode_scale_unit_repairs(
            baseline, assignments, ["change", "same", "same"]
        )

        self.assertEqual(
            [item["speaker_label"] for item in resolved],
            [experiment.SIL_UNKNOWN_LABEL] * 3,
        )
        self.assertTrue(debug["unknown_runs"][0]["accepted"])
        self.assertEqual(
            debug["unknown_runs"][0]["best_known_average_similarity"], 0.7233
        )

    def test_sustained_unknown_rejects_known_voice_with_high_run_average(self):
        baseline = [
            {
                "start": index * 5.0,
                "end": (index + 1) * 5.0,
                "speaker_id": "host",
                "speaker_label": "Host",
            }
            for index in range(3)
        ]
        assignments = [
            {
                "unit_id": index,
                "start": index * 5.0,
                "end": (index + 1) * 5.0,
                "duration": 5.0,
                "speaker": experiment.SIL_UNKNOWN_LABEL,
                "similarity": experiment.SIL_UNKNOWN_STATE_SCORE,
                "margin": 0.01,
                "best_known_similarity": 0.75,
                "segment_ids": [index],
            }
            for index in range(3)
        ]

        resolved, debug = experiment.apply_episode_scale_unit_repairs(
            baseline, assignments, ["change", "same", "same"]
        )

        self.assertEqual([item["speaker_label"] for item in resolved], ["Host"] * 3)
        self.assertFalse(debug["unknown_runs"][0]["accepted"])

    def test_rejected_unit_evidence_preserves_each_overlapped_baseline_label(self):
        baseline = [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker_id": "chamath",
                "speaker_label": "Chamath",
            },
            {
                "start": 2.0,
                "end": 4.0,
                "speaker_id": "friedberg",
                "speaker_label": "Friedberg",
            },
        ]
        assignments = [
            {
                "unit_id": 0,
                "start": 0.0,
                "end": 4.0,
                "duration": 4.0,
                "speaker": "Chamath",
                "similarity": 0.91,
                "margin": 0.08,
                "segment_ids": [0, 1],
            }
        ]
        resolved, debug = experiment.apply_episode_scale_unit_repairs(
            baseline, assignments, ["change", "same"]
        )
        self.assertEqual(
            [item["speaker_label"] for item in resolved],
            ["Chamath", "Friedberg"],
        )
        self.assertEqual(debug["activated_pairs"], {})

    def test_conservative_repair_opens_and_propagates_inside_baseline_run(self):
        baseline = [
            {"start": 0.0, "end": 3.0, "speaker_id": "jason", "speaker_label": "Jason"},
            {"start": 3.0, "end": 15.0, "speaker_id": "jason", "speaker_label": "Jason"},
            {"start": 15.0, "end": 27.0, "speaker_id": "jason", "speaker_label": "Jason"},
        ]
        reference = [
            {**baseline[0], "boundary_before": "change", "cluster_id": 0},
            {
                **baseline[1],
                "speaker_id": "brad",
                "speaker_label": "Brad",
                "boundary_before": "change",
                "cluster_id": 1,
            },
            {
                **baseline[2],
                "speaker_id": "brad",
                "speaker_label": "Brad",
                "boundary_before": "change",
                "cluster_id": 1,
            },
        ]
        assignments = [
            {"speaker": "Jason", "similarity": 0.95, "margin": 0.1, "start": 0.0, "end": 3.0},
            {"speaker": "Brad", "similarity": 0.95, "margin": 0.1, "start": 3.0, "end": 15.0},
            {"speaker": "Brad", "similarity": 0.95, "margin": 0.1, "start": 15.0, "end": 27.0},
        ]
        resolved, debug = experiment.apply_conservative_turn_repairs(
            baseline, reference, assignments
        )
        self.assertEqual([segment["speaker_label"] for segment in resolved], ["Jason", "Brad", "Brad"])
        self.assertEqual(debug["repaired_turn_count"], 2)
        self.assertEqual(debug["repaired_segment_count"], 2)

    def test_missing_identity_repair_requires_repeated_episode_evidence(self):
        baseline = [
            {"start": 0.0, "end": 12.0, "speaker_id": "jason", "speaker_label": "Jason"},
            {"start": 12.0, "end": 24.0, "speaker_id": "jason", "speaker_label": "Jason"},
        ]
        reference = [
            {**baseline[0], "speaker_id": "brad", "speaker_label": "Brad"},
            {**baseline[1], "speaker_id": "brad", "speaker_label": "Brad"},
        ]
        assignments = [
            {"speaker": "Brad", "similarity": 0.94, "margin": 0.08, "start": 0.0, "end": 12.0},
            {"speaker": "Brad", "similarity": 0.95, "margin": 0.09, "start": 12.0, "end": 24.0},
        ]
        resolved, debug = experiment.apply_missing_identity_repairs(
            baseline, baseline, reference, assignments
        )
        self.assertEqual([segment["speaker_label"] for segment in resolved], ["Brad", "Brad"])
        self.assertIn("brad", debug["activated_identities"])

    def test_missing_identity_repair_leaves_complete_roster_unchanged(self):
        baseline = [
            {"start": 0.0, "end": 12.0, "speaker_id": "brad", "speaker_label": "Brad"},
            {"start": 12.0, "end": 24.0, "speaker_id": "jason", "speaker_label": "Jason"},
        ]
        reference = [dict(segment) for segment in baseline]
        assignments = [
            {"speaker": "Brad", "similarity": 0.94, "margin": 0.08, "start": 0.0, "end": 12.0},
            {"speaker": "Brad", "similarity": 0.95, "margin": 0.09, "start": 12.0, "end": 24.0},
        ]
        resolved, debug = experiment.apply_missing_identity_repairs(
            baseline, baseline, reference, assignments
        )
        self.assertEqual(resolved, baseline)
        self.assertEqual(debug["activated_identities"], {})

    def test_systematic_confusion_repair_requires_episode_scale_evidence(self):
        baseline = [
            {"start": 0.0, "end": 31.0, "speaker_id": "friedberg", "speaker_label": "Friedberg"},
            {"start": 31.0, "end": 62.0, "speaker_id": "friedberg", "speaker_label": "Friedberg"},
            {"start": 62.0, "end": 93.0, "speaker_id": "sacks", "speaker_label": "Sacks"},
            {"start": 93.0, "end": 124.0, "speaker_id": "sacks", "speaker_label": "Sacks"},
        ]
        reference = [
            {**baseline[0], "speaker_id": "sacks", "speaker_label": "Sacks"},
            {**baseline[1], "speaker_id": "sacks", "speaker_label": "Sacks"},
            dict(baseline[2]),
            dict(baseline[3]),
        ]
        assignments = [
            {"speaker": "Sacks", "similarity": 0.94, "margin": 0.08, "start": 0.0, "end": 31.0},
            {"speaker": "Sacks", "similarity": 0.95, "margin": 0.09, "start": 31.0, "end": 62.0},
            {"speaker": "Sacks", "similarity": 0.95, "margin": 0.09, "start": 62.0, "end": 93.0},
            {"speaker": "Sacks", "similarity": 0.95, "margin": 0.09, "start": 93.0, "end": 124.0},
        ]
        resolved, debug = experiment.apply_systematic_confusion_repairs(
            baseline, baseline, reference, assignments
        )
        self.assertEqual([segment["speaker_label"] for segment in resolved[:2]], ["Sacks", "Sacks"])
        self.assertIn("friedberg->sacks", debug["activated_pairs"])

    def test_systematic_confusion_rejects_unanchored_known_target(self):
        baseline = [
            {"start": 0.0, "end": 31.0, "speaker_id": "sacks", "speaker_label": "Sacks"},
            {"start": 31.0, "end": 62.0, "speaker_id": "sacks", "speaker_label": "Sacks"},
            {"start": 62.0, "end": 72.0, "speaker_id": "friedberg", "speaker_label": "Friedberg"},
        ]
        reference = [
            {**baseline[0], "speaker_id": "friedberg", "speaker_label": "David Friedberg"},
            {**baseline[1], "speaker_id": "friedberg", "speaker_label": "David Friedberg"},
            dict(baseline[2]),
        ]
        assignments = [
            {
                "speaker": "David Friedberg",
                "similarity": 0.94,
                "margin": 0.08,
                "start": 0.0,
                "end": 31.0,
            },
            {
                "speaker": "David Friedberg",
                "similarity": 0.95,
                "margin": 0.09,
                "start": 31.0,
                "end": 62.0,
            },
        ]
        resolved, debug = experiment.apply_systematic_confusion_repairs(
            baseline, baseline, reference, assignments
        )
        self.assertEqual(resolved, baseline)
        self.assertEqual(debug["activated_pairs"], {})
        self.assertIn("sacks->friedberg", debug["rejected_unanchored_pairs"])

    def test_turns_preserve_time_and_split_long_display_turns(self):
        segments = [
            {
                "start": 0.0,
                "end": 15.0,
                "text": "one",
                "speaker_id": "jason",
                "speaker_label": "Jason",
            },
            {
                "start": 15.1,
                "end": 30.0,
                "text": "two",
                "speaker_id": "jason",
                "speaker_label": "Jason",
            },
        ]
        turns = experiment.turns_from_segments(segments, max_turn_seconds=25.0)
        self.assertEqual(len(turns), 2)
        self.assertEqual((turns[0]["start"], turns[1]["end"]), (0.0, 30.0))

    def test_audit_metrics_are_duration_weighted(self):
        segments = [
            {"start": 0.0, "end": 9.0, "speaker_label": "Jason"},
            {"start": 9.0, "end": 10.0, "speaker_label": "Jason"},
        ]
        rows = [
            {"start": 0.0, "end": 9.0, "mapped_speaker_label": "Jason"},
            {"start": 9.0, "end": 10.0, "mapped_speaker_label": "Brad"},
        ]
        metrics = experiment.evaluate_audit(segments, rows)
        self.assertEqual(metrics["segment_accuracy"], 0.5)
        self.assertEqual(metrics["duration_accuracy"], 0.9)

    def test_boundary_metrics_report_coverage(self):
        rows = [
            {"mapped_speaker_label": "Jason"},
            {"mapped_speaker_label": "Jason"},
            {"mapped_speaker_label": "Brad"},
        ]
        metrics = experiment.evaluate_boundaries(
            ["change", "uncertain", "change"], rows
        )
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["confident_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
