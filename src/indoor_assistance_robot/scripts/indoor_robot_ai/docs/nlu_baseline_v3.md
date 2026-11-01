# Reproducible NLU baseline v3

Normal inference loads the TF-IDF artifact configured by
`config/nlu.yaml`: `models/nlu/baseline_v3/intent_tfidf_lr.joblib`. Inference
does not train when the artifact is missing. The old `load_or_train` method is
kept as a compatibility alias that only loads and now raises a clear model
error when the configured artifact is unavailable.

## Dataset and provenance

- Training data: `data/nlu_baseline_v3/train.jsonl`, 374 rows.
- Training SHA-256: `7c0cfecd90c572ade6a090a12b77d116add96223c3ea29a78356dc4d608b22dc`.
- Dataset manifest: `data/nlu_baseline_v3/manifest.json`.
- Exclusion audit: `data/nlu_baseline_v3/legacy_test_exclusion.json`.
- Collection requirements: `data/nlu_baseline_v3/COLLECTION_SPEC.md`.

All 70 rows from the legacy test file are kept out of the v3 training dataset.
The audit identifies the 42 that appeared in the v2 training split. The
remaining legacy rows also stay out of new test sets. The data does not record
independent source/session IDs or reliable template IDs, so the v3 data cannot
support source- or template-independent generalization claims yet.

## Commands

Build the new dataset once; the command refuses to overwrite an existing v3
dataset:

```bash
PYTHONPATH= .venv/bin/python -m src.nlu.baseline_dataset
```

Train the configured artifact with seed 42:

```bash
PYTHONPATH= .venv/bin/python -m src.nlu.train_baseline_v3 --seed 42
```

`PYTHONPATH= .venv/bin/python -m src.nlu.train --seed 42` is a compatibility
alias for the same versioned trainer; it no longer overwrites the v1 artifact.

Training writes a provenance sidecar and refuses to overwrite an existing
artifact. For a repeatability check, use a new output path:

```bash
PYTHONPATH= .venv/bin/python -m src.nlu.train_baseline_v3 --seed 42 --output /tmp/intent_tfidf_lr_repro.joblib
```

Score the exact configured artifact on the old hand-authored data as a
**diagnostic only**:

```bash
PYTHONPATH= .venv/bin/python -m src.nlu.evaluate_baseline_v3 --role legacy_diagnostic
```

The report is written to `results/nlu_baseline_v3/legacy_diagnostic.json` and
contains both artifact and dataset hashes. It is not an independent test
report. Use a unique `--output` path for additional report runs.

## Recorded artifact

- Artifact SHA-256: `fc38d80221a2089750d8ff1a82fb2ce3b431033308353ba943362444c5485a78`.
- Seed: 42.
- Labels: NAVIGATE, RETRIEVE, DELIVER, FOLLOW, STOP, RETURN, UNSUPPORTED.
- Features: word TF-IDF unigrams and bigrams; lowercase; Unicode accent
  stripping; `min_df=1`; outer whitespace stripped at both fit and inference.
- Classifier: balanced Logistic Regression, `lbfgs`, `max_iter=1000`,
  `random_state=42`.
- Package versions and dataset/manifest hashes are in the artifact provenance
  sidecar.

## Current evaluation status

The legacy diagnostic covers 70 examples and is hash-bound to the saved
artifact. It reports 0.857 accuracy, 0.855 macro-F1, OOS precision 0.75, OOS
recall 1.0, and zero false accepts of unsupported examples on this dataset.
Its source/template independence is unverified; these numbers do not establish
generalization. No independent dev, in-domain, unseen-template, paraphrase, or
OOS evaluation sets are available. Collect and annotate those splits using the
requirements in `data/nlu_baseline_v3/COLLECTION_SPEC.md` before making
generalization claims.
