import unittest
from types import SimpleNamespace

from experiments.luna_text_pipeline_eval.luna_text_pipeline_eval import (
    KNOWN_ATTRIBUTION_WINDOWS,
    RequestRecorder,
    VALID_EFFORTS,
    apply_segment_assignments,
    estimated_cost_usd,
    evaluate_known_attribution_windows,
    segment_reconciliation_batches,
    segment_reconciliation_schema,
    usage_record,
)


class LunaTextPipelineEvalTests(unittest.TestCase):
    def test_effort_order_matches_artifact_numbering(self):
        self.assertEqual(VALID_EFFORTS.index("high") + 1, 4)

    def test_usage_record_supports_sdk_objects(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=500,
                total_tokens=1500,
                input_tokens_details=SimpleNamespace(cached_tokens=200),
                output_tokens_details=SimpleNamespace(reasoning_tokens=75),
            )
        )

        self.assertEqual(
            usage_record(response),
            {
                "input_tokens": 1000,
                "cached_input_tokens": 200,
                "output_tokens": 500,
                "reasoning_tokens": 75,
                "total_tokens": 1500,
            },
        )

    def test_estimated_cost_applies_cached_discount(self):
        records = [
            {
                "status": "ok",
                "input_tokens": 1_000_000,
                "cached_input_tokens": 250_000,
                "output_tokens": 100_000,
            },
            {
                "status": "error",
                "input_tokens": 9_000_000,
                "cached_input_tokens": 0,
                "output_tokens": 9_000_000,
            },
        ]

        self.assertEqual(estimated_cost_usd(records), 0.275)

    def test_request_recorder_summarizes_by_stage(self):
        recorder = RequestRecorder("low")
        recorder.records = [
            {
                "stage": "translation",
                "status": "ok",
                "seconds": 2.5,
                "input_tokens": 100,
                "cached_input_tokens": 10,
                "output_tokens": 50,
                "reasoning_tokens": 5,
                "total_tokens": 150,
            },
            {
                "stage": "translation",
                "status": "error",
                "seconds": 1.0,
                "error": "temporary",
            },
        ]

        summary = recorder.summarize()

        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["successful_request_count"], 1)
        self.assertEqual(summary["input_tokens"], 100)
        self.assertEqual(summary["reasoning_tokens"], 5)
        self.assertEqual(summary["by_stage"]["translation"]["requests"], 2)
        self.assertEqual(summary["by_stage"]["translation"]["api_seconds"], 3.5)

    def test_segment_schema_constrains_count_and_speaker_ids(self):
        schema = segment_reconciliation_schema(3, ["speaker_a", "speaker_b"])
        assignments = schema["properties"]["assignments"]

        self.assertEqual(assignments["minItems"], 3)
        self.assertEqual(assignments["maxItems"], 3)
        self.assertEqual(
            assignments["items"]["properties"]["speaker_id"]["enum"],
            ["speaker_a", "speaker_b"],
        )

    def test_segment_batches_cover_every_segment_once_as_target(self):
        segments = [
            {
                "chunk_index": chunk_index,
                "local_speaker": "A",
                "start": index,
                "end": index + 0.5,
                "text": str(index),
            }
            for index, chunk_index in enumerate([1, 1, 2, 2, 3, 3, 4, 4])
        ]

        batches = segment_reconciliation_batches(
            segments,
            chunks_per_batch=2,
            context_segments=1,
        )

        self.assertEqual(len(batches), 2)
        target_ids = [segment_id for _, batch_ids in batches for segment_id in batch_ids]
        self.assertEqual(target_ids, list(range(len(segments))))
        self.assertEqual([segment_id for segment_id, _ in batches[0][0]], [0, 1, 2, 3, 4])

    def test_segment_assignments_can_split_one_local_label(self):
        segments = [
            {
                "chunk_index": 1,
                "local_speaker": "A",
                "speaker_id": "speaker_jason",
                "speaker_id_source": "voice",
                "text": "Brad, what do you think?",
            },
            {
                "chunk_index": 1,
                "local_speaker": "A",
                "speaker_id": "speaker_jason",
                "speaker_id_source": "voice_neighbor",
                "text": "I think David nailed it.",
            },
        ]

        reconciled, changed = apply_segment_assignments(
            segments,
            [
                {"segment_id": 0, "speaker_id": "speaker_jason"},
                {"segment_id": 1, "speaker_id": "speaker_brad"},
            ],
            [0, 1],
        )

        self.assertEqual(changed, 1)
        self.assertEqual(reconciled[0]["speaker_id"], "speaker_jason")
        self.assertEqual(reconciled[1]["speaker_id"], "speaker_brad")
        self.assertEqual(
            reconciled[1]["speaker_id_before_text_reconciliation"],
            "speaker_jason",
        )

    def test_known_attribution_windows_pass_for_expected_speakers(self):
        speakers = [
            {"id": "jason", "label_short": "Jason", "label_full": "Jason Calacanis"},
            {"id": "friedberg", "label_short": "Friedberg", "label_full": "David Friedberg"},
            {"id": "brad", "label_short": "Brad", "label_full": "Brad Gerstner"},
        ]
        ids = {"Jason": "jason", "Friedberg": "friedberg", "Brad": "brad"}
        segments = [
            {
                "start": window["start"],
                "end": window["end"],
                "speaker_id": ids[window["expected"]],
            }
            for window in KNOWN_ATTRIBUTION_WINDOWS
        ]

        evaluation = evaluate_known_attribution_windows(segments, speakers)

        self.assertTrue(evaluation["all_groups_passed"])
        self.assertTrue(all(window["passed"] for window in evaluation["windows"]))


if __name__ == "__main__":
    unittest.main()
