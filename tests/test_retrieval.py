import unittest

from vidcurator.retrieval import FrameIndex, normalize_video_id
from vidcurator.types import FrameRecord


class FrameIndexVideoIdTest(unittest.TestCase):
    def test_normalize_strips_extension(self):
        self.assertEqual(normalize_video_id("test_BWG-4.mp4"), "test_BWG-4")
        self.assertEqual(normalize_video_id("test_BWG-4"), "test_BWG-4")
        self.assertEqual(normalize_video_id("kMryvefpcF8"), "kMryvefpcF8")

    def test_frames_lookup_accepts_qa_and_index_ids(self):
        frame = FrameRecord(
            video_id="test_BWG-4",
            frame_id="f000001",
            timestamp=1.0,
            path="data/frames/mlvu/test_BWG-4/f000001.jpg",
            caption="scene",
            metadata={},
            embedding=None,
        )
        index = FrameIndex([frame])
        self.assertEqual(len(index.frames("test_BWG-4.mp4")), 1)
        self.assertEqual(len(index.frames("test_BWG-4")), 1)
        self.assertIs(index.get("test_BWG-4.mp4", "f000001"), frame)


if __name__ == "__main__":
    unittest.main()
