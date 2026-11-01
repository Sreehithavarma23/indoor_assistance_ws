# NLU baseline v3 data collection specification

This directory currently contains a **training-only** dataset. No new dev or
test examples have been fabricated. Do not treat the legacy diagnostic report
as evidence from an independent source or template split.

## Required fields for every newly collected row

Use JSONL with the following fields:

```json
{
  "row_id": "stable unique ID assigned at intake",
  "text": "verbatim utterance",
  "intent": "one configured intent label",
  "entities": {},
  "source_id": "collection/session/author batch identifier",
  "template_id": "known template ID, or null for naturally authored text",
  "paraphrase_family_id": "shared ID for paraphrases/noise variants, or null",
  "source_type": "human_authored|synthetic_template|synthetic_paraphrase|...",
  "collection_method": "how and when the utterance was obtained",
  "annotator_id": "pseudonymous annotator identifier",
  "split": "dev|test_indomain|test_unseen_template|test_paraphrase|test_oos"
}
```

Keep original wording and annotation history. Corrections must be recorded as
reviewed label changes with an author and reason; do not silently relabel.

## Split assignment rules

- Assign by connected groups, not individual rows. A source/session, template,
  or paraphrase family may appear in only one split.
- Keep all ASR/noise variants with their source utterance and paraphrase family.
- `test_indomain` should use newly collected supported commands with source
  groups and template groups absent from training.
- `test_unseen_template` should use supported commands with newly authored
  template IDs and independent source groups.
- `test_paraphrase` should hold complete paraphrase families out of training;
  document the seed utterance and how each variant was authored.
- `test_oos` should include human-authored, capability-relevant out-of-scope
  requests from independent source groups. Include difficult near-domain
  negatives and record the intended rejection rationale.
- Keep dev distinct from all test groups and use it for model selection only.
- Freeze final tests before model selection. Record row counts, per-label counts,
  source/template/family disjointness, and SHA-256 hashes in the split manifest.

## Minimum provenance evidence

The current data does not identify author/session/source cohorts or true
template IDs. Broad source categories and sentence-derived legacy family IDs
are insufficient. Collectors must provide those identifiers at intake. If an
identifier is unknown, use null and keep the affected rows out of claims about
that kind of independence. Do not infer template IDs by replacing catalog
entities in the utterance text.

The old 70-row `data/nlu/test.jsonl` corpus remains excluded from v3 training
and from the new independent splits. The 42 rows found in the v2 training split
are listed in `legacy_test_exclusion.json`; all other legacy-test rows are also
excluded from the v3 dataset. Any future reuse requires explicit human review
and must not be presented as untouched-test evidence.
