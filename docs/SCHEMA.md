# Schema

All major artifacts are JSONL. This keeps data adapters, frame indexing, inference, and RL training loosely coupled.

## Frame Record

```json
{
  "video_id": "road_trip",
  "frame_id": "road_210",
  "timestamp": 210.0,
  "path": "frames/road_trip/road_210.jpg",
  "caption": "After coming down from the hill, a white jacket is hanging on her black backpack.",
  "embedding": [0.01, -0.02],
  "metadata": {"scene": "key evidence"}
}
```

`embedding` is optional. The local retriever uses captions; real runs should store SigLIP/CLIP vectors or keep them in a sidecar index.

## QA Example

```json
{
  "qid": "road_q1",
  "video_id": "road_trip",
  "question": "When she comes down from the hill, what changes occur to her backpack?",
  "options": ["Dark red jacket", "Nothing changed", "White jacket", "Dark blue jacket"],
  "answer": "White jacket",
  "answer_index": 2,
  "evidence_spans": [{"start": 200.0, "end": 225.0, "label": "white jacket hangs on backpack"}],
  "domain": "movie_like",
  "split": "val",
  "metadata": {}
}
```

For datasets without temporal evidence, set `evidence_spans` to `[]`. Metrics that require evidence spans will return `null`.

## Prediction

`runs/*/predictions.jsonl` stores one row per `(policy, budget, question)`. It includes:

- `answer`, `answer_index`, `confidence`
- `selected_frame_ids`, `selected_timestamps`
- `probes`, including `missing_moment` and `sufficiency`
- `trace`, including retrieval queries and tentative answers
- `metadata.running_summary` for curator policies

## Summary

`runs/*/summary.json` stores group metrics keyed as `policy@budget`, context-efficiency AUC by policy, single-frame shortcut deltas, and retrieval recall at K.
