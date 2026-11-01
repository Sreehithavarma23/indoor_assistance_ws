# Phase 5 dialogue development set

Hand-authored multi-turn scenarios. **Not** a real-world benchmark.

Each JSONL row:

```json
{
  "id": "retrieve_clarify_then_ready",
  "turns": ["Bring me the bottle.", "From the kitchen."],
  "expected_status": "READY",
  "expected_intent": "RETRIEVE",
  "expected_entities": {
    "object": "bottle",
    "source_location": "kitchen",
    "destination": "user"
  }
}
```

Run:

```bash
PYTHONPATH= python -m src.dialogue.evaluation
```
