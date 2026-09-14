# MiniMax-H3 Native Reference-Conditioning Investigation

Date: 2026-09-14  
Machine: MacBook Pro, M2 Max, 30-core GPU, 32 GB unified memory  
Status: **active; upstream Qwen3VL conditioning path exonerated, blocker moved downstream to H3 DiT/keyframe integration**

## Problem statement

Native MiniMax-H3 32B Q4 conditioning is computationally viable with staged execution, but earlier Ref2VA runs showed poor reference adherence after frame 0. The investigation therefore focused on locating the first point where reference-image information was lost or corrupted.

The key rule throughout was: **do not patch or optimise until a discriminating experiment identifies the failing subsystem.**

## Phase 1 — GGMLOps vs plain ops

The production Qwen3VL vision tower was run twice with identical weights and input:

- Path A: production `GGMLOps` path;
- Path B: plain `manual_cast` path using fully materialised/dequantised tensors.

Measured L2 norms were bit-identical at every checkpoint:

| Checkpoint | GGMLOps | Plain ops | Ratio |
|---|---:|---:|---:|
| patch embed | 368.679 | 368.679 | 1.0000 |
| block 8 | 413.060 | 413.060 | 1.0000 |
| block 16 | 1502.006 | 1502.006 | 1.0000 |
| block 24 | 4709.692 | 4709.692 | 1.0000 |
| DeepStack merger 0 | 1383.3 | 1383.3 | 1.0000 |
| DeepStack merger 1 | 1250.4 | 1250.4 | 1.0000 |
| DeepStack merger 2 | 408.8 | 408.8 | 1.0000 |
| final merger | 1372.353 | 1372.353 | 1.0000 |

**Conclusion:** `GGMLOps` is not the cause.

## Phase 2 — QKV and structural validation

The real production `Qwen3VLVisionModel` loaded with:

- `missing=0`
- `unexpected=0`

Confirmed:

- mmproj is genuinely loaded;
- DeepStack source layers `8/16/24` map to positions `0/1/2`;
- Qwen3VL merger keys land on `norm`, `linear_fc1`, and `linear_fc2`;
- no stale or orphaned tensor names remain.

A permutation search against an official Qwen3-VL-8B reference strongly favoured identity `[Q,K,V]` ordering:

| Block | Identity | Swap K/V | Per-head interleave |
|---:|---:|---:|---:|
| 0 | 0.99951 | 0.40668 | 0.01679 |
| 8 | 0.99676 | 0.38988 | 0.04773 |
| 16 | 0.98999 | 0.34598 | 0.04448 |
| 24 | 0.98214 | 0.31761 | 0.04741 |

The 8B reference showed much smaller DeepStack/merger magnitudes, but this was later proven to be an invalid cross-model baseline rather than a local bug.

## Phase 3 — Official MiniMax-H3 oracle

The exact official MiniMax-H3 encoder shard was downloaded and SHA-256 verified:

- file: `model-00014-of-00014.safetensors`
- size: `3,270,697,008` bytes
- SHA-256: `e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b`

Local mmproj provenance:

- file: `MiniMax-H3-encoder-mmproj-F16.gguf`
- size: `1,200,333,792` bytes
- SHA-256: `448b7ab3ca0b1f92a5ebca9cc6a4b56e7dbb45b0e313c9d34d8a1154eb49d7ea`
- source: `joeygambino/MiniMax-H3-encoder-GGUF`

Fifty representative tensors were compared across patch embedding, positional embedding, vision blocks, all three DeepStack mergers, and the final merger.

Result on every compared tensor:

- max absolute error: `0.000000`
- mean absolute error: `0.000000`
- relative L2 error: `0.000000`
- cosine similarity: effectively `1.0`

This includes the previously suspicious DeepStack `norm.weight` tensors and all merger FC weights.

**Conclusion:** the local GGUF/mmproj vision-tower conversion is bit-identical to the official MiniMax-H3 checkpoint. The larger DeepStack activation magnitude is intentional H3 behaviour and must not be patched.

## Phase 4 — Full 50-layer Qwen3VL/LLM conditioning trace

A controlled A/B test compared:

- A: prompt + real reference image;
- B: same prompt, image ablated.

Token accounting was internally consistent:

- A: 380 total tokens; 256 image-tagged in `build_image_inputs`, 124 text;
- B: 116 tokens; all text;
- final A token tags: 258 image / 122 text, including the expected vision boundary tokens.

The decisive test compared the literal shared prompt-text suffix between A and B. If the reference image were being ignored, these text hidden states would remain nearly identical through depth.

| Layer | Mean cosine A vs B | Min cosine | Relative L2 difference |
|---:|---:|---:|---:|
| 0 | 0.998 | 0.83 | 0.10 |
| 1 | 0.997 | 0.87 | 0.10 |
| 2 | 0.995 | 0.82 | 0.13 |
| 3 | 0.994 | 0.77 | 0.14 |
| 24 | 0.967 | 0.39 | 0.99 |
| 49 | 0.962 | 0.42 | 0.98 |

The divergence grows coherently with depth, exactly as expected when image information propagates through self-attention into the shared text representation.

The final image-tagged conditioning is finite and non-degenerate:

- 258 image-tagged positions;
- std approximately 29.07;
- L2 approximately 33,397;
- no NaN or Inf observed.

**Conclusion:** the Qwen3VL/50-layer LLM conditioning path is functioning and reference-sensitive.

## Current blocker

The first four phases exonerate the complete upstream reference-conditioning chain:

- GGUF loading;
- mmproj conversion;
- GGMLOps;
- QKV ordering;
- DeepStack mapping;
- vision-tower weights and activations;
- image-token placement and masks;
- DeepStack injection into the LLM;
- 50-layer Qwen3VL/LLM propagation.

The unresolved subsystem is now downstream:

```text
reference image
  -> Qwen3VL vision tower            [exonerated]
  -> 50-layer Qwen3VL/LLM            [exonerated]
  -> final H3 conditioning
  -> MiniMaxH3ImageToVideo
  -> minimax_keyframes / VAE anchor
  -> H3 DiT conditioning consumption [current blocker]
  -> sampled latent
  -> video
```

The next experiment must determine whether changing the reference image materially changes:

1. the conditioning actually passed into H3 DiT;
2. cross-attention or equivalent conditioning consumption;
3. representative early/mid/late DiT blocks;
4. the final predicted latent/noise.

## 32 GB unified-memory constraint

This machine has 32 GB **unified memory**, not dedicated 32 GB GPU VRAM. CPU, GPU and system processes share the same pool.

Operational policy:

- parallelise reasoning, static code inspection and log analysis only;
- run one heavy H3/Qwen/Metal workload at a time;
- serialise all renders;
- preserve Stage A / Stage B process isolation;
- record wall-clock time, memory pressure and swap around heavy runs;
- reduce resolution/frames/steps before adding concurrency.

## Planned diagnostic micro-render

After the DiT trace, run a tiny diagnostic A/B/C render if the numerical evidence warrants it:

- portrait 9:16;
- smallest valid H3-aligned resolution determined from code/RUNBOOK;
- 5 frames if the confirmed constraint is `frames % 17 == 5`;
- 2–4 steps;
- fixed seed, prompt, sampler and all other settings;
- A: reference image 1;
- B: materially different reference image 2;
- C: safe ablated/neutral control if supported;
- renders strictly serial.

This is a **diagnostic smoke test**, not a quality benchmark.

## External acceleration lead — unverified

A community YouTube report described an Apple-Silicon MiniMax-H3 acceleration path using `h3.c` plus a `lightx2v` Turbo patch, with step reduction from 12 to 8 or 4. Reported figures included approximately 18:17 -> 8:11 for a 9-second video and 35:51 -> 14:56 for a 15-second video, together with a warning that a strength coefficient error of 16x produced blurred output.

These claims are **UNVERIFIED** for this repository and were reported under materially different hardware/resource assumptions, including 128 GB RAM and a much larger model footprint. They must remain separate from the current 32 GB correctness investigation until independently reproduced.

## Current conclusion

The native 32B reference path has **not** been proven broken end-to-end. What has been proven is narrower and stronger:

> The entire upstream Qwen3VL reference-conditioning path is correct against the official MiniMax-H3 oracle and is demonstrably reference-sensitive. The remaining investigation must focus on H3 DiT/keyframe integration and, if that is also exonerated, reassess whether the original failure was caused by workflow or sampling configuration rather than a broken conditioning implementation.
