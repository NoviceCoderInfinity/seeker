import unittest

from vidcurator.answerers import LexicalAnswerer
from vidcurator.types import FrameRecord, QAExample


class AnswererTest(unittest.TestCase):
    def test_options_in_prompt_do_not_leak_into_scores(self):
        example = QAExample(
            qid="q",
            video_id="v",
            question="What color jacket appears?",
            options=["red jacket", "white jacket"],
            answer_index=1,
        )
        frames = [FrameRecord(video_id="v", frame_id="f", timestamp=0.0, caption="A white jacket is visible.")]
        rendered_context = "Options:\n0. red jacket\n1. white jacket"
        pred = LexicalAnswerer().answer(example, frames, rendered_context)
        self.assertEqual(pred.answer_index, 1)


if __name__ == "__main__":
    unittest.main()
