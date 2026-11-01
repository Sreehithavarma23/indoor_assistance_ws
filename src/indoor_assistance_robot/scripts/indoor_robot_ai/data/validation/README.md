# Phase 4 validation development set

Hand-authored status labels for `VALID`, `INCOMPLETE`, `AMBIGUOUS`,
and `UNSUPPORTED`. This is **not** a real-world benchmark.

Each JSONL row:

```json
{"text": "Bring me the bottle.", "status": "INCOMPLETE"}
```

The evaluator runs Phase 3 NLU, then Phase 4 `validate()`, and compares
the predicted status to `status`.
