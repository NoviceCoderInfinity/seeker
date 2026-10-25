import unittest

from vidcurator.answerers import LexicalAnswerer
from vidcurator.io import load_examples, load_frames
from vidcurator.policies import build_policy
from vidcurator.retrieval import FrameIndex


class PolicyRegistryTest(unittest.TestCase):
    def test_reviewer_ablation_policies_run_on_fixture(self):
        index = FrameIndex(load_frames("data/fixtures/frames.jsonl"))
        example = load_examples("data/fixtures/qa.jsonl")[0]
        answerer = LexicalAnswerer()
        for name in [
            "oracle_span",
            "curator_probe_off",
            "curator_curation_off",
            "curator_summary_off",
            "scripted_curator",
        ]:
            policy = build_policy(
                name,
                index,
                answerer,
                budget=2,
                config={"max_hops": 2, "retrieve_per_hop": 2, "seed_uniform": 1},
            )
            result = policy.run(example)
            self.assertEqual(result.policy, name)
            self.assertLessEqual(result.num_frames, 2)


if __name__ == "__main__":
    unittest.main()
