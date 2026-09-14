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
- native MiniMax-H3 32B conditioning experiments;
- two-process memory staging to avoid simultaneous encoder + DiT residency;
- unified-memory / swap / jetsam observations;
- measured 56-frame and 124-frame runs;
- EasyCache measurements;
- reference-image conditioning failures and current root-cause investigation;
- reproducible workflow, downloader, and runbook artefacts.

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

## Planned series

- MiniMax-H3 / ComfyUI video generation
- Apple Silicon GPU / unified-memory profiling
- Local LLM inference with MLX
- Cantonese ASR / dictation workloads
- Model memory staging and process isolation
- Practical agent workflows for local AI experimentation

## Status

This repository is being reconstructed from measured experiment logs and validated artefacts produced on the reference machine. Raw evidence and reproducible files will be added progressively rather than fabricated from memory.
