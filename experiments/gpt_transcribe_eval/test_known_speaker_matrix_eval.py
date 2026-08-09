import unittest

from known_speaker_matrix_eval import (
    AUDITED_SAMPLE,
    CURRENT_SAMPLE,
    TruthSpan,
    conditions_for,
    score_open_set,
)


class KnownSpeakerMatrixEvalTests(unittest.TestCase):
    def test_current_matrix_includes_absent_and_actual_rosters(self):
        conditions = {condition.key: condition for condition in conditions_for(CURRENT_SAMPLE)}

        self.assertEqual(len(conditions), 6)
        self.assertEqual(
            [clip.speaker for clip in conditions["cross-present3"].references],
            ["Jason", "Sacks", "Friedberg"],
        )
        self.assertEqual(
            [clip.speaker for clip in conditions["cross-present3-plus-absent-chamath"].references],
            ["Jason", "Sacks", "Friedberg", "Chamath"],
        )
        self.assertEqual(
            [clip.speaker for clip in conditions["cross-actual4"].references],
            ["Jason", "Sacks", "Friedberg", "Brad"],
        )
        self.assertEqual(
            {clip.video_id for clip in conditions["alternate-cross-actual4"].references},
            {"ViqYWhLimGg", "PHL1j2ti420"},
        )

    def test_audited_matrix_uses_absent_sacks_and_actual_gavin(self):
        conditions = {condition.key: condition for condition in conditions_for(AUDITED_SAMPLE)}

        self.assertEqual(
            [clip.speaker for clip in conditions["cross-present3-plus-absent-sacks"].references],
            ["Jason", "Chamath", "Friedberg", "Sacks"],
        )
        self.assertEqual(
            [clip.speaker for clip in conditions["cross-actual4"].references],
            ["Jason", "Chamath", "Friedberg", "Gavin"],
        )

    def test_open_set_score_rewards_anonymous_unknown_speaker(self):
        truth = (
            TruthSpan(0.0, 5.0, "Jason", "group"),
            TruthSpan(5.0, 10.0, "Brad", "group"),
        )
        candidate = [
            {"start": 0.0, "end": 5.0, "speaker": "Jason", "text": "known"},
            {"start": 5.0, "end": 10.0, "speaker": "A", "text": "unknown"},
        ]

        score = score_open_set(truth, candidate, ["Jason"])

        self.assertEqual(score["known_named_accuracy"], 1.0)
        self.assertEqual(score["unreferenced_false_known_rate"], 0.0)
        self.assertEqual(score["oracle_identity_accuracy"], 1.0)
        self.assertEqual(score["anonymous_label_map"], {"A": "Brad"})

    def test_open_set_score_detects_absent_reference_false_match(self):
        truth = (TruthSpan(0.0, 8.0, "Brad", "group"),)
        candidate = [
            {"start": 0.0, "end": 8.0, "speaker": "Chamath", "text": "wrong"},
        ]

        score = score_open_set(truth, candidate, ["Chamath"])

        self.assertEqual(score["unreferenced_false_known_rate"], 1.0)
        self.assertEqual(score["absent_reference_false_positive_seconds"], 8.0)
        self.assertEqual(score["oracle_identity_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
