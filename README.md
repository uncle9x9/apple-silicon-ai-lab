# Apple Silicon AI Lab

Measured, reproducible experiments for running local AI workloads on Apple Silicon.

This repository exists to turn expensive trial-and-error into reusable evidence for people and AI agents. Results are separated into **measured**, **reproduced**, **observed**, **hypothesis**, **unverified**, and **retired** findings so exploratory conclusions are not mistaken for facts.

## Current reference machine

- MacBook Pro, M2 Max
- 30-core GPU
- 32 GB unified memory
- macOS / Metal / MPS

## First case study: MiniMax-H3 + ComfyUI

The first series documents a real MiniMax-H3 video-generation bake-off on an M2 Max 32 GB machine, including:

- a validated Qwen3-VL-4B BF16 + ClipProj baseline;
- Native MiniMax-H3 32B conditioning experiments;
- two-process memory staging to avoid simultaneous encoder + DiT residency;
- unified-memory / swap / jetsam observations;
- measured 56-frame and 124-frame runs;
- EasyCache measurements;
- a four-phase reference-conditioning root-cause investigation;
- exact comparison against the official MiniMax-H3 encoder checkpoint;
- proof that the local GGUF/mmproj vision tower and full 50-layer Qwen3VL/LLM conditioning path are correct and reference-sensitive;
- the remaining blocker: H3 DiT/keyframe-conditioning integration;
- reproducible workflow, downloader, runbook and diagnostic artefacts as they are validated.

See [`experiments/minimax-h3/`](experiments/minimax-h3/README.md).

## Evidence labels

| Label | Meaning |
|---|---|
| **MEASURED** | Captured directly from a real run. |
| **REPRODUCED** | Repeated successfully under the same conditions. |
| **OBSERVED** | Directly seen but not yet controlled/repeated. |
| **HYPOTHESIS** | Plausible explanation awaiting proof. |
| **UNVERIFIED** | Reported or inferred but not independently checked. |
| **RETIRED** | Earlier conclusion superseded by stronger evidence. |

## Repository principles

1. **Measured evidence > agent claims.**
2. Preserve raw measurements and distinguish them from interpretation.
3. Do not present extrapolated runtime as measured runtime.
4. Prefer reproducible workflows and scripts over screenshots alone.
5. Record failed approaches when they teach something reusable.
6. Treat Apple Silicon as a unified-memory system: CPU offload is not the same as freeing physical RAM.
7. Avoid re-running expensive experiments when an existing controlled result already answers the question.
8. On constrained-memory Macs, use **parallel brains, serial GPU**: parallelise analysis, not heavy model residency or renders.

## Planned series

- MiniMax-H3 / ComfyUI video generation
- Apple Silicon GPU / unified-memory profiling
- Local LLM inference with MLX
- Cantonese ASR / dictation workloads
- Model memory staging and process isolation
- Practical agent workflows for local AI experimentation

## Status

The MiniMax-H3 case study is active. The upstream reference-conditioning stack has been exonerated through controlled tensor and full-LLM A/B tests; current work is tracing H3 DiT/keyframe integration. Raw evidence and reproducible files are added progressively rather than reconstructed or fabricated from memory.
