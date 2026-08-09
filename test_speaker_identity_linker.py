import unittest

import numpy as np

import speaker_identity_linker as linker


class FakeEncoder:
    def embed_utterance(self, _audio):
        return np.asarray([0.0, 1.0], dtype=np.float32)


class SpeakerIdentityReferenceTests(unittest.TestCase):
    def test_activates_only_present_frozen_voices_and_enrolls_guest(self):
        baseline = [
            {
                "start": 0.0,
                "end": 4.0,
                "speaker_id": "speaker_jason_calacanis",
                "speaker_label": "Jason Calacanis",
            },
            {
                "start": 20.0,
                "end": 24.0,
                "speaker_id": "speaker_gavin_baker",
                "speaker_label": "Gavin Baker",
            },
            {
                "start": 40.0,
                "end": 44.0,
                "speaker_id": "speaker_gavin_baker",
                "speaker_label": "Gavin Baker",
            },
        ]
        references, debug = linker.build_episode_reference_centroids(
            baseline,
            np.zeros(50 * 10, dtype=np.float32),
            10,
            FakeEncoder(),
            {
                "Jason Calacanis": [1.0, 0.0],
                "David Friedberg": [0.5, 0.5],
            },
            active_speaker_labels=["Jason Calacanis", "Gavin Baker"],
        )

        self.assertEqual(set(references), {"Jason Calacanis", "Gavin Baker"})
        self.assertEqual(debug["frozen"], [{"speaker": "Jason Calacanis"}])
        self.assertEqual(debug["automatic"][0]["speaker"], "Gavin Baker")

    def test_active_mapping_can_enable_frozen_guest_without_baseline_segments(self):
        references, debug = linker.build_episode_reference_centroids(
            [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "speaker_id": "speaker_jason_calacanis",
                    "speaker_label": "Jason Calacanis",
                }
            ],
            np.zeros(50, dtype=np.float32),
            10,
            FakeEncoder(),
            {
                "Jason Calacanis": [1.0, 0.0],
                "Brad Gerstner": [0.0, 1.0],
            },
            active_speaker_labels=["Jason Calacanis", "Brad Gerstner"],
        )

        self.assertEqual(set(references), {"Jason Calacanis", "Brad Gerstner"})
        self.assertEqual(
            debug["frozen"],
            [{"speaker": "Jason Calacanis"}, {"speaker": "Brad Gerstner"}],
        )

    def test_does_not_enroll_unknown_external_baseline_label(self):
        references, debug = linker.build_episode_reference_centroids(
            [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "speaker_id": "speaker_unknown_external",
                    "speaker_label": "Unknown/External",
                }
            ],
            np.zeros(50, dtype=np.float32),
            10,
            FakeEncoder(),
            {"Jason Calacanis": [1.0, 0.0]},
        )

        self.assertEqual(references, {})
        self.assertEqual(debug["automatic"], [])


class EpisodeScaleRepairTests(unittest.TestCase):
    @staticmethod
    def assignment(unit_id, start, end, speaker, similarity, margin):
        return {
            "unit_id": unit_id,
            "start": start,
            "end": end,
            "duration": end - start,
            "segment_ids": [unit_id],
            "chunk_index": 0,
            "local_speaker": "A",
            "speaker": speaker,
            "similarity": similarity,
            "margin": margin,
            "best_known_similarity": similarity,
        }

    def test_sustained_low_known_similarity_activates_unknown_external(self):
        baseline = [
            {
                "start": 0.0,
                "end": 6.0,
                "speaker_id": "speaker_david_sacks",
                "speaker_label": "David Sacks",
            },
            {
                "start": 6.0,
                "end": 12.0,
                "speaker_id": "speaker_david_sacks",
                "speaker_label": "David Sacks",
            },
        ]
        assignments = [
            self.assignment(0, 0.0, 6.0, "Unknown/External", 0.62, 0.0),
            self.assignment(1, 6.0, 12.0, "Unknown/External", 0.64, 0.0),
        ]

        resolved, debug = linker.apply_episode_scale_unit_repairs(
            baseline,
            assignments,
            ["change", "same"],
        )

        self.assertEqual(
            [segment["speaker_label"] for segment in resolved],
            ["Unknown/External", "Unknown/External"],
        )
        self.assertTrue(debug["unknown_runs"][0]["accepted"])

    def test_short_identity_disagreement_does_not_rewrite_baseline(self):
        baseline = [
            {
                "start": 0.0,
                "end": 4.0,
                "speaker_id": "speaker_david_sacks",
                "speaker_label": "David Sacks",
            }
        ]
        assignments = [
            self.assignment(0, 0.0, 4.0, "David Friedberg", 0.94, 0.12)
        ]

        resolved, debug = linker.apply_episode_scale_unit_repairs(
            baseline,
            assignments,
            ["change"],
        )

        self.assertEqual(resolved[0]["speaker_label"], "David Sacks")
        self.assertEqual(debug["repaired_segment_count"], 0)


if __name__ == "__main__":
    unittest.main()
