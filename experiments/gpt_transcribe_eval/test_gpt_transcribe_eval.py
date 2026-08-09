import unittest

from gpt_transcribe_eval import (
    Sample,
    boundary_score,
    gpt_transcribe_fields,
    keyword_hits,
    normalize_words,
    speaker_overlap_score,
    text_metrics,
)


class GptTranscribeEvalTests(unittest.TestCase):
    def test_contextual_fields_use_new_plural_context_parameters(self):
        sample = Sample(
            key="sample",
            video_id="video",
            title="Title",
            episode_url="https://example.com",
            context="A technical interview.",
            keywords=("OpenAI", "Sam Altman"),
            reference_kind="test",
            reference_path=None,
            reference_names={},
        )

        fields = gpt_transcribe_fields(sample, contextual=True)

        self.assertIn(("model", "gpt-transcribe"), fields)
        self.assertIn(("languages[]", "en"), fields)
        self.assertIn(("keywords[]", "OpenAI"), fields)
        self.assertIn(("keywords[]", "Sam Altman"), fields)
        self.assertIn(("prompt", "A technical interview."), fields)

    def test_normalization_and_text_metrics_are_word_based(self):
        self.assertEqual(
            normalize_words("OpenAI's GPT-Transcribe, v2!"),
            ["openai's", "gpt", "transcribe", "v2"],
        )
        metrics = text_metrics("one two three", "one too three")
        self.assertEqual(metrics["token_edit_distance"], 1)
        self.assertEqual(metrics["disagreement_rate"], 0.3333)

    def test_keyword_hits_are_case_and_punctuation_insensitive(self):
        self.assertEqual(
            keyword_hits(
                "Sam Altman discussed OpenAI.",
                ["sam altman", "OpenAI", "AGI"],
            ),
            ["sam altman", "OpenAI"],
        )

    def test_speaker_overlap_score_uses_temporal_overlap_and_name_aliases(self):
        reference = [
            {"start": 0.0, "end": 4.0, "speaker": "Jason Calacanis", "text": "a"},
            {"start": 4.0, "end": 8.0, "speaker": "David Friedberg", "text": "b"},
        ]
        candidate = [
            {"start": 0.0, "end": 4.0, "speaker": "Jason", "text": "a"},
            {"start": 4.0, "end": 8.0, "speaker": "Jason", "text": "b"},
        ]

        score = speaker_overlap_score(
            reference,
            candidate,
            ["Jason Calacanis", "David Friedberg"],
        )

        self.assertEqual(score["agreement"], 0.5)
        self.assertEqual(len(score["mismatches"]), 1)

    def test_speaker_alias_does_not_confuse_two_davids(self):
        reference = [
            {"start": 0.0, "end": 4.0, "speaker": "David Friedberg", "text": "a"},
        ]
        candidate = [
            {"start": 0.0, "end": 4.0, "speaker": "David Friedberg", "text": "a"},
        ]
        score = speaker_overlap_score(
            reference,
            candidate,
            ["David Sacks", "David Friedberg"],
        )
        self.assertEqual(score["agreement"], 1.0)

    def test_boundary_score_matches_changes_with_tolerance(self):
        reference = [
            {"start": 0.0, "end": 5.0, "speaker": "A"},
            {"start": 5.0, "end": 9.0, "speaker": "B"},
            {"start": 9.0, "end": 12.0, "speaker": "A"},
        ]
        candidate = [
            {"start": 0.0, "end": 5.8, "speaker": "A"},
            {"start": 5.8, "end": 8.0, "speaker": "B"},
            {"start": 9.5, "end": 12.0, "speaker": "A"},
        ]

        score = boundary_score(reference, candidate, tolerance_seconds=1.0)

        self.assertEqual(score["matched_changes"], 2)
        self.assertEqual(score["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
