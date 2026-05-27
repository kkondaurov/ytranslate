import tempfile
import unittest
from pathlib import Path

import ytranslate_server


def make_job(job_id="job123", status="running"):
    return ytranslate_server.JobRecord(
        job_id=job_id,
        url="https://youtu.be/example",
        canonical_url="https://youtu.be/example",
        target_language="Russian",
        status=status,
        created_at="2026-05-25T10:00:00+00:00",
        started_at="2026-05-25T10:01:00+00:00",
    )


class JobProgressTests(unittest.TestCase):
    def test_job_record_tracks_asr_chunk_progress_from_events(self):
        job = make_job()

        job.record_event("info", "Running OpenAI ASR on chunk-000.mp3 (1/3)")
        job.record_event("info", "OpenAI ASR completed chunk-000.mp3 (1/3)")
        job.record_event("info", "Using cached OpenAI ASR chunk chunk-001.mp3 (2/3)")

        self.assertEqual(job.phase, "asr")
        self.assertEqual(job.phase_detail, "Using cached OpenAI ASR chunk chunk-001.mp3 (2/3)")
        self.assertEqual(job.progress["asr_chunks_total"], 3)
        self.assertEqual(job.progress["asr_chunks_done"], 2)
        self.assertEqual(job.progress["current_chunk"], "chunk-001.mp3")
        self.assertEqual(job.events[-1]["message"], "Using cached OpenAI ASR chunk chunk-001.mp3 (2/3)")

    def test_job_record_tracks_failed_asr_chunks_from_events(self):
        job = make_job()

        job.record_event("error", "OpenAI ASR failed chunk-000.mp3 (1/3): upload failed")

        self.assertEqual(job.phase, "asr")
        self.assertEqual(job.progress["asr_chunks_total"], 3)
        self.assertEqual(job.progress["asr_chunks_done"], 0)
        self.assertEqual(job.progress["asr_failed_chunks"], ["chunk-000.mp3"])
        self.assertEqual(ytranslate_server.status_steps(job)[5]["detail"], "0 / 3, 1 failed")

    def test_status_page_renders_latest_job_steps(self):
        job = make_job()
        job.record_event("info", "Fetching metadata...")
        job.record_event("info", "Running OpenAI ASR on chunk-000.mp3 (1/2)")
        job.record_event("info", "OpenAI ASR completed chunk-000.mp3 (1/2)")

        html = ytranslate_server.render_status_page([job])

        self.assertIn("ytranslate status", html)
        self.assertIn("job123", html)
        self.assertIn("ASR chunks", html)
        self.assertIn("1 / 2", html)
        self.assertIn("Fetching metadata", html)

    def test_job_history_round_trips_recent_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "jobs.json"
            job = make_job(status="failed")
            job.error = "network failed"
            job.record_event("error", "network failed")

            ytranslate_server.save_job_history(history_path, [job])
            loaded = ytranslate_server.load_job_history(history_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].job_id, "job123")
        self.assertEqual(loaded[0].status, "failed")
        self.assertEqual(loaded[0].error, "network failed")
        self.assertEqual(loaded[0].events[0]["level"], "error")


if __name__ == "__main__":
    unittest.main()
