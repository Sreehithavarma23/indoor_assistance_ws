# Robust Speech and Language Upgrade Notes

Project: Intelligent Speech-Controlled Autonomous Mobile Manipulator for Indoor Assistance.

## Current Audit

Baseline architecture before this upgrade:

```text
Audio file / microphone
  -> preprocessing + energy VAD
  -> Faster-Whisper ASR
  -> TF-IDF + Logistic Regression intent classifier
  -> deterministic entity extraction
  -> validation
  -> dialogue / clarification
  -> deterministic planner or constrained LLM planner
  -> validated symbolic TaskPlan
```

Important findings:

- Current ASR backend: `faster-whisper`, configured as `model_size: base`, `device: auto`, `compute_type: default`, `language: en`.
- Cached local ASR checkpoint: `models/whisper/models--Systran--faster-whisper-base`.
- Current NLU backend: TF-IDF unigrams/bigrams plus Logistic Regression, with deterministic catalog entity extraction.
- Current LLM planner does not interpret arbitrary raw language. It only receives validated READY dialogue intent/entities and proposes a JSON `TaskPlan`.
- Dialogue state is stored per `PipelineSession` inside `DialogueManager`.
- Missing entities, ambiguity, unsupported intents, and STOP are handled before planning.
- Plans are validated structurally by `src/planning/plan_validator.py`.
- A separate dry-run adapter now revalidates symbolic plans and catalog IDs; it cannot publish ROS messages.

## Machine Check

Observed on this machine:

- OS: Ubuntu 22.04.5 LTS.
- Python in venv: 3.10.12.
- CPU: AMD Ryzen 5 3500U, 8 logical CPUs.
- RAM: 5.7 GiB total, about 1.0 GiB available during audit.
- Swap: 2.0 GiB total, fully used during audit.
- GPU tooling: `nvidia-smi` and `rocminfo` not available.
- Free disk in repo filesystem: about 52 GiB.

This is suitable for the current `faster-whisper` base CPU/int8 baseline and deterministic NLU. It is not a good default target for unquantized 4B language models or multiple large ASR checkpoints loaded locally.

## Model Candidates

| Model | Role | License | Status |
|---|---|---:|---|
| Current `faster-whisper` base | ASR baseline | Depends on converted checkpoint; currently cached from Systran | Implemented and tested |
| `openai/whisper-large-v3-turbo` | ASR candidate | MIT | Researched only; not downloaded or benchmarked locally |
| `Qwen/Qwen3-ASR-0.6B-hf` | ASR candidate | Apache-2.0 | Researched only; not downloaded or benchmarked locally |
| `Qwen/Qwen3-4B-Instruct-2507` | language-understanding candidate | Apache-2.0 | Researched only; not downloaded or benchmarked locally |

Notes:

- `openai/whisper-large-v3-turbo` is multilingual, listed as 809M parameters, and is faster than full large-v3 because decoder layers were reduced from 32 to 4.
- `Qwen/Qwen3-ASR-0.6B-hf` is Apache-2.0, supports ASR/language identification across many languages/dialects, and requires newer Transformers support.
- `Qwen/Qwen3-4B-Instruct-2507` is Apache-2.0, 4.0B parameters, supports very long context, and should be optional or quantized/hosted on this hardware.

## Upgrade Implemented

Implemented in this pass:

- Added `src/language_understanding/` as a typed schema layer over current NLU, validation, and dialogue.
- Added `LanguageUnderstandingResult` fields for interpreted request, task category, entities, provenance, missing information, ambiguity, clarification state, unsupported capabilities, proposed symbolic task, model backend, model errors, and metadata.
- Added `language_understanding` to serialized `PipelineResult`.
- Extended ASR evaluation to report:
  - backend/model/device/compute type/language,
  - model load time,
  - process memory before/after,
  - audio duration,
  - real-time factor,
  - critical-term error rate for object/location/person/negation/correction terms.
- Updated the ASR manifest example with optional `critical_terms`.

Not implemented in this pass:

- No new pretrained ASR or LLM checkpoint was downloaded.
- No hosted inference dependency was introduced.
- No ROS 2 execution or simulation was run.
- No physical robot or hardware commands were issued.

## Commands

Regression tests:

```bash
source .venv/bin/activate
PYTHONPATH= pytest -q
```

Pipeline offline evaluation:

```bash
source .venv/bin/activate
PYTHONPATH= python -m src.pipeline.evaluation
```

ASR benchmark framework:

```bash
source .venv/bin/activate
PYTHONPATH= python -m src.speech.evaluate \
  --manifest data/asr_eval/manifest.csv \
  --output results/asr/eval.json
```

Manifest format:

```csv
audio_path,reference,critical_terms
data/raw_audio/navigate_kitchen.wav,go to the kitchen,kitchen
```

Do not fill references from model predictions. Use consented recordings with human-authored transcripts.

## Evaluation State

Current audit regression checks:

- Full automated suite: 260 passed, 2 skipped.
- Frozen text evaluations do not include ASR and do not claim real-world speech performance.

ASR real-model benchmarking:

- Framework run with `data/asr_eval/manifest.example.csv` evaluates zero recordings, as intended.
- No WER/CER/critical-term accuracy is claimed without labeled real audio.

## Recommended Next Steps

1. Collect a consented labeled ASR dataset with multiple speakers, microphone distances, noise conditions, and command categories.
2. Run the current Faster-Whisper base baseline and save `results/asr/eval.json`.
3. On stronger hardware or with careful disk/RAM planning, benchmark `openai/whisper-large-v3-turbo` and `Qwen/Qwen3-ASR-0.6B-hf` on the same manifest.
4. Add an optional language-understanding backend behind `LanguageUnderstandingResult` after selecting a local quantized or hosted model.
5. Keep deterministic validation and symbolic planning as mandatory gates after any language model output.
6. Keep the dry-run boundary in place until a household simulation exposes verified location, perception, grasp, and STOP interfaces. The sibling surgical simulator does not provide that household contract.
