import unittest

from vidcurator.buffer import RunningContextBuffer
from vidcurator.types import BufferBudget, FrameRecord


class BufferTest(unittest.TestCase):
    def test_budget_compresses_low_usefulness_frame(self):
        buffer = RunningContextBuffer("question", [], BufferBudget(max_frames=2, max_summary_chars=500))
        frames = [
            FrameRecord(video_id="v", frame_id="a", timestamp=0, caption="low"),
            FrameRecord(video_id="v", frame_id="b", timestamp=1, caption="high"),
            FrameRecord(video_id="v", frame_id="c", timestamp=2, caption="mid"),
        ]
        buffer.add_frame(frames[0], usefulness=0.1, hop=1)
        buffer.add_frame(frames[1], usefulness=0.9, hop=1)
        buffer.add_frame(frames[2], usefulness=0.5, hop=1)
        self.assertEqual(buffer.selected_ids(), ["b", "c"])
        self.assertIn("low", buffer.state.running_summary)


if __name__ == "__main__":
    unittest.main()
