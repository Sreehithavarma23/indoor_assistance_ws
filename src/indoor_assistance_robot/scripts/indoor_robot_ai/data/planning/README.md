# Phase 6 planning development set

Hand-authored cases for deterministic plan checking. **Not** a robot-execution benchmark.

Each JSONL row provides `intent`, `entities`, `expected_status`, and optionally `expected_actions`.

```bash
PYTHONPATH= python -m src.planning.evaluation
```
