import unittest

from vidcurator.metrics import context_efficiency_auc, frame_evidence_pr, is_correct
from vidcurator.types import EvidenceSpan, PolicyResult, QAExample


class MetricsTest(unittest.TestCase):
    def test_frame_evidence_precision_recall(self):
        example = QAExample(
            qid="q",
            video_id="v",
            question="q",
            evidence_spans=[EvidenceSpan(start=10, end=20, label="key")],
        )
        precision, recall = frame_evidence_pr([5, 12, 30], example)
        self.assertAlmostEqual(precision, 1 / 3)
        self.assertEqual(recall, 1.0)

    def test_accuracy_by_answer_index(self):
        example = QAExample(
            qid="q",
            video_id="v",
            question="q",
            options=["a", "b"],
            answer_index=1,
        )
        result = PolicyResult(
            qid="q",
            video_id="v",
            policy="p",
            budget=2,
            answer="b",
            answer_index=1,
            confidence=0.9,
            selected_frame_ids=[],
            selected_timestamps=[],
            probes=[],
            num_retrievals=0,
            num_frames=0,
            trace=[],
        )
        self.assertTrue(is_correct(result, example))

    def test_context_efficiency_auc(self):
        self.assertAlmostEqual(context_efficiency_auc([(4, 0.5), (8, 0.75), (16, 1.0)]), 0.7916666666666666)


if __name__ == "__main__":
    unittest.main()
