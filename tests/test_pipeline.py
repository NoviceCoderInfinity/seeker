import tempfile
import unittest
from pathlib import Path

from vidcurator.evaluate import evaluate_config


class PipelineTest(unittest.TestCase):
    def test_fixture_evaluation_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = evaluate_config(
                {
                    "frames_path": "data/fixtures/frames.jsonl",
                    "qa_path": "data/fixtures/qa.jsonl",
                    "output_dir": tmpdir,
                    "answerer": {"kind": "lexical"},
                    "budgets": [2, 4],
                    "policies": ["uniform", "mmr_retrieve", "scripted_curator", "best_single_frame"],
                    "policy": {"max_hops": 2, "retrieve_per_hop": 2, "seed_uniform": 1},
                    "retrieval_recall_at_k": [1, 2],
                }
            )
            self.assertTrue((Path(tmpdir) / "predictions.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "summary.json").exists())
            self.assertIn("scripted_curator@2", summary["groups"])
            self.assertIn("uniform@2", summary["groups"])


if __name__ == "__main__":
    unittest.main()
