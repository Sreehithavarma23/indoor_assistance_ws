# Public dataset investigation (auxiliary use only)

This note records what was inspected for the indoor mobile-manipulator NLU
upgrade. Public assistant datasets are **not** treated as robot-specific
training labels.

## SNIPS NLU

| Field | Finding |
|---|---|
| Relevance | Task-oriented intent/slot data (weather, music, restaurants, etc.). Useful as a **methodology reference** for intent+slot schemas, not as robot NAVIGATE/RETRIEVE labels. |
| Language | Primarily English (also other locales in some releases). |
| License | Check the specific GitHub release / dataset card before redistribution. Historically associated with Snips / Sonos research releases. |
| Class coverage | No native mapping to NAVIGATE/RETRIEVE/DELIVER/FOLLOW/STOP/RETURN. |
| Use in this project | **Not merged.** Do not map SNIPS intents onto robot intents. |

## CLINC150

| Field | Finding |
|---|---|
| Relevance | 150 in-scope assistant intents plus a dedicated **out-of-scope (OOS)** evaluation set. Valuable for **OOS detection methodology**. |
| Language | English. |
| License | Research dataset; verify the authors' terms before bundling raw files. |
| Class coverage | Assistant domains (banking, travel, …). Labels are **not** robot capabilities. |
| Use in this project | **Methodology + hand-authored aux OOS only.** `source=public_aux_oos` rows are original phrases inspired by assistant-style OOS practice; they are **not** a redistribution of CLINC150 utterances. |

## MASSIVE (Amazon / Alexa)

| Field | Finding |
|---|---|
| Relevance | Multilingual slot-and-intent data derived from SLURP-style schemas. English subset can inform slot annotation conventions. |
| Language | 50+ locales; English subset relevant here. |
| License | CC BY 4.0 (confirm on the official dataset card). |
| Class coverage | Smart-speaker domains; no direct robot taxonomy. |
| Use in this project | **Not downloaded or merged** in this pass (disk/RAM/time budget). Schema ideas only. |

## Public speech corpora for ASR

| Dataset | Relevance | Status |
|---|---|---|
| LibriSpeech | Read speech; domain mismatch for short robot commands. | Not used. |
| Common Voice | Crowdsourced; license-friendly but domain mismatch. | Not used. |
| Project `data/raw_audio/` | Intended home for consented robot-command recordings. | Empty / gitignored; **no WER claimed**. |

## What each resource can and cannot establish

| Resource | Can establish | Cannot establish |
|---|---|---|
| Hand-authored robot commands | In-domain intent/entity behavior for configured catalogs | Real-world spoken robustness |
| Synthetic templates/paraphrases | Coverage of patterns, OOS cues, missing slots | Equivalence to natural speech |
| Aux assistant-style OOS | Rejection of unrelated chat-like requests | Coverage of all unsupported physical tasks |
| CLINC150 (if later downloaded under license) | Generic OOS detection baselines | Robot capability grounding |
| SNIPS / MASSIVE | Slot schema / multilingual methods | Robot taxonomy accuracy |
| Whisper pretrained checkpoint | General English ASR starting point | Domain WER without labeled audio |

## Policy for this repository

1. Never claim public assistant labels are robot intents.
2. Keep provenance on every example (`source`, `family_id`, `tags`).
3. Prefer project-specific unsupported requests for capability gating.
4. If a public corpus is added later, store license text and a download script; do not commit redistributed files without license clearance.
