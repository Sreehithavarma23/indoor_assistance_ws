# NLU v2 Dataset Report

**Total utterances:** 700
**Families:** 471

## Split sizes

| Split | Count |
|---|---:|
| `train` | 416 |
| `dev` | 79 |
| `test_indomain` | 90 |
| `test_paraphrase` | 55 |
| `test_oos` | 60 |

## Intent distribution (global)

| Intent | Count |
|---|---:|
| DELIVER | 107 |
| FOLLOW | 50 |
| NAVIGATE | 233 |
| RETRIEVE | 137 |
| RETURN | 52 |
| STOP | 41 |
| UNSUPPORTED | 80 |

## Provenance (source)

| Source | Count |
|---|---:|
| `adversarial_edge` | 17 |
| `asr_noise_augmented` | 85 |
| `human_authored` | 12 |
| `legacy_hand_authored` | 265 |
| `public_aux_oos` | 9 |
| `synthetic_paraphrase` | 15 |
| `synthetic_template` | 297 |

## Entity slot mentions

| Slot | Count |
|---|---:|
| `action` | 568 |
| `destination` | 342 |
| `object` | 243 |
| `person` | 157 |
| `source_location` | 116 |

**Multi-step (`task_order`) rows:** 9
**Near-duplicate prefixes spanning splits:** 71

## Leakage checks

- No exact-normalized or train↔frozen-test family overlaps detected.

## Known limitations

- Most utterances are synthetic_template or synthetic_paraphrase, not natural speech.
- ASR noise is approximate character/word corruption, not real Whisper errors.
- public_aux_oos examples are hand-authored methodology references, not redistributed CLINC/SNIPS/MASSIVE rows.
- Frozen test sets are project-internal; they are not an external independent benchmark.
- No consented multi-speaker audio corpus is bundled; ASR WER cannot be claimed from this release.
- Multi-step commands use a single primary intent label for classifier training; task_order is auxiliary.
