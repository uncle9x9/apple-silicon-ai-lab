# Apple Silicon AI Lab

Measured, reproducible experiments for running local AI workloads on Apple Silicon.

This repository exists to turn expensive trial-and-error into reusable evidence for people and AI agents. Results are separated into **measured**, **reproduced**, **observed**, **hypothesis**, **unverified**, and **retired** findings so exploratory conclusions are not mistaken for facts.

## Current reference machine

- MacBook Pro, M2 Max
- 30-core GPU
- 32 GB unified memory
- macOS / Metal / MPS

## First case study: MiniMax-H3

The first series documents a real MiniMax-H3 video-generation investigation on an M2 Max 32 GB machine, including:

- a validated Qwen3-VL-4B BF16 + ClipProj baseline;
- Native MiniMax-H3 32B conditioning experiments;
- two-process memory staging to avoid simultaneous encoder + DiT residency;
- unified-memory / swap / jetsam observations;
- measured 56-frame and 124-frame runs;
- EasyCache measurements;
- a five-phase reference-conditioning investigation;
- exact comparison against the official MiniMax-H3 encoder checkpoint;
- proof that the local GGUF/mmproj vision tower and full 50-layer Qwen3VL/LLM conditioning path are correct and reference-sensitive;
- a controlled DiT A/C test showing the final predicted latent is materially reference-sensitive (`cosine=0.8372`, `relative L2=0.549`);
- a resolved DiT QKV-layout audit proving that the production GGUF stores contiguous `[Q_all | K_all | V_all]`, matching ComfyUI's split semantics;
- an independent `antirez/h3.c` validation track for FL2VA vs Ref2VA and low-memory Apple-Silicon performance.

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
9. When independent implementations disagree semantically, resolve the exact checkpoint contract before patching code.
10. Distinguish **model task semantics** from implementation correctness: FL2VA first-frame anchoring and Ref2VA subject-reference conditioning are not interchangeable.

## Planned series

- MiniMax-H3 / ComfyUI / h3.c video generation
- Apple Silicon GPU / unified-memory profiling
- Local LLM inference with MLX
- Cantonese ASR / dictation workloads
- Model memory staging and process isolation
- Practical agent workflows for local AI experimentation

## Status

The MiniMax-H3 implementation investigation is **closed with no actionable code defect found in the exercised path**. Vision loading, Qwen3VL propagation, DiT reference sensitivity and production QKV layout have all been validated.

Current work has moved to an independent **h3.c validation campaign**:

1. reproduce FL2VA first-frame behaviour in h3.c;
2. run genuine Ref2VA subject-reference conditioning;
3. compare ComfyUI FL2VA vs h3.c FL2VA vs h3.c Ref2VA;
4. measure wall time, memory pressure and swap on the M2 Max 32 GB reference machine;
5. keep Turbo/lightx2v optimisation separate until task semantics and baseline correctness are established.

Raw evidence and reproducible files are added progressively rather than reconstructed or fabricated from memory.
