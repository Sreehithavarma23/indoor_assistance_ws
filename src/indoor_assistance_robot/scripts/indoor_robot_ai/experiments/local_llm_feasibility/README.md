# Phase 9A Local LLM Feasibility

This directory is an isolated experiment for local language interpretation only.
It does not integrate with the production pipeline, ROS 2, MoveIt 2, robot
hardware, or the deterministic task planner.

Model:

- Repository: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- File: `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- License observed on Hugging Face: Apache-2.0
- Downloaded size: 1,117,320,736 bytes
- SHA-256: `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`

Runtime boundary:

- CPU-only llama.cpp inference through `llama-cpp-python`
- Small context window
- No cloud APIs
- No remote inference service
- No executable robot plans, ROS commands, motor commands, joint commands,
  velocity commands, scripts, or coordinates in model output

Memory notes:

- Process memory is measured from `/proc/self/status` and describes this
  Python process.
- System memory is measured from `/proc/meminfo` and describes the whole OS.
  These values are related but not interchangeable.

Typical command:

```bash
experiments/local_llm_feasibility/.venv/bin/python \
  experiments/local_llm_feasibility/run_feasibility.py \
  --cases experiments/local_llm_feasibility/cases.jsonl \
  --output experiments/local_llm_feasibility/results/diagnostic_report.json \
  --baseline
```
