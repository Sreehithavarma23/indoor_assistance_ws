# Phase 3 NLU development dataset

This is a **hand-authored text dataset** for developing and evaluating the
baseline NLU system. It is **not** a real-world speech benchmark and must
not be reported as deployment performance.

## Files

| File | Role |
|---|---|
| `train.jsonl` | Training examples for the intent classifier |
| `test.jsonl` | Held-out unseen phrasings |
| `_author_dataset.py` | Source used to write the JSONL files |

Runtime code loads only the JSONL files.

## Format

One JSON object per line:

```json
{"text": "Take me to the kitchen.", "intent": "NAVIGATE", "entities": {"destination": "kitchen", "action": "navigate"}}
```

Entity values are normalized catalog ids (`water_bottle`, `living_room`, `user`).
Missing slots are omitted. The system must not invent them.

## Split policy

- Train and test share **no** identical or hyphen-normalized sentences.
- Test uses different wording from train (polite, conversational, longer).
- Both splits cover: NAVIGATE, RETRIEVE, DELIVER, FOLLOW, STOP, RETURN, UNSUPPORTED.

Regenerate after editing `_author_dataset.py`:

```bash
PYTHONPATH=. python data/nlu/_author_dataset.py
```
