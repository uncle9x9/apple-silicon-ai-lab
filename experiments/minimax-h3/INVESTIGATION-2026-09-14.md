# MiniMax-H3 Native Reference-Conditioning Investigation

Date: 2026-09-14 to 2026-09-15  
Machine: MacBook Pro, M2 Max, 30-core GPU, 32 GB unified memory  
Status: **implementation investigation closed; no actionable code defect found in the exercised path**

## Problem statement

Native MiniMax-H3 32B Q4 conditioning is computationally viable with staged execution, but earlier full-video runs showed poor reference adherence after frame 0. The investigation focused on locating the first point where reference-image information was lost, corrupted or semantically misinterpreted.

The governing rule was: **do not patch or optimise until a discriminating experiment identifies the failing contract.**

That rule eliminated several plausible but incorrect hypotheses and ultimately closed the implementation-bug investigation without finding an actionable defect.

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

## Phase 2 — Qwen3VL QKV and structural validation

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

## Phase 3 — Official MiniMax-H3 encoder oracle

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

The decisive test compared the literal shared prompt-text suffix between A and B.

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

## Phase 5 — Production H3 DiT A/C test

The next test instrumented the actual production H3 DiT path rather than another upstream proxy.

Diagnostic configuration:

- portrait 288x512;
- 5 frames;
- 2 denoising steps;
- serial execution;
- A: real reference image;
- C: ablated/no-image control;
- otherwise identical prompt, seed and sampling configuration.

The real-reference Stage B run took approximately 96 seconds total.

### DiT boundary and block observations

A:

- context shape: `(1, 268, 5376)`;
- 146 image-tagged / 122 text-tagged positions;
- context std: image `2.222`, text `2.220`.

C:

- context shape: `(1, 116, 5376)`;
- no image-tagged positions;
- text std: `2.230`.

Representative checkpoints:

| Checkpoint | A | C |
|---|---:|---:|
| block 0 post std / L2 | 180.9 / 354980 | 243.6 / 365982 |
| block 25 pre std / L2 | 966.4 / 1895972 | 1166.6 / 1753019 |
| block 49 pre std / L2 | 5508.9 / 10808190 | 5088.7 / 7646727 |
| final predicted latent mean / std | -0.0366 / 1.856 | -0.1867 / 2.105 |

Direct final-latent comparison:

- cosine similarity: `0.8372`;
- relative L2 difference: `0.549`;
- no NaN/Inf observed.

This is a substantial, coherent difference rather than an ignored-reference result.

### Static trace

The production path was also traced through:

- `extra_conds`;
- `minimax_token_tags`;
- `minimax_keyframes`;
- `preprocess_text_embeds`;
- packed text/condition/audio/video sequence construction;
- DiT attention;
- keyframe RoPE positioning.

No routing, masking or silent-bypass defect was found in the exercised graph.

### Conditional scheduled-CLIP defect

A separate real defect was identified in a scheduled CLIP branch: with `use_clip_schedule=True`, one path can rebuild the output dictionary without preserving extras such as `minimax_token_tags`.

This did **not** affect the tested workflow. The captured DiT boundary contained populated image tags, empirically ruling that condition out for this run.

**Conclusion:** the tested H3 DiT/keyframe path is reference-sensitive. The original full-video drift is not explained by a silent reference-routing failure.

## Phase 6 — Production DiT QKV checkpoint-layout audit

The independent `antirez/h3.c` implementation exposed a deeper checkpoint-contract question.

h3.c documents the released raw H3 DiT QKV tensor as **per-head grouped/interleaved**:

```text
[h0 Q,K,V][h1 Q,K,V]...
```

Current ComfyUI MiniMax-H3 attention consumes a conventional contiguous layout:

```text
[Q_all | K_all | V_all]
```

This initially looked like a possible implementation defect, but a critical caveat was recognised: Comfy-oriented checkpoints may be repacked before runtime.

### Loader/converter trace

For the exact production GGUF:

- `UnetLoaderGGUF.load_unet()` calls `gguf_sd_loader(unet_path)`;
- the DiT/UNET loader applies no architecture-specific QKV permutation;
- tensor name remains `blocks.0.attn.qkv_proj.weight`;
- `comfy.gguf.orig_shape.*` metadata indicates the generic ComfyUI-GGUF quantisation path.

Therefore the production GGUF preserves the pre-quantisation row order.

### Official raw tensor vs production checkpoint

A pristine official `Ref2VA/transformer` block-0 QKV tensor was retrieved at tensor level.

The production pruned checkpoint is not merely a reordered copy of the vanilla official checkpoint:

- position-wise mean cosine: `0.028`;
- median cosine: `0.001`.

This required a structural self-consistency test rather than direct row equality.

### Structural result

| Tensor | Hypothesis | matched-minus-mismatched `|cosine|` gap | head-0 Q·Kᵀ diagonal/off-diagonal | Verdict |
|---|---|---:|---:|---|
| Official vanilla | contiguous | `0.00026` | `0.96` | wrong grouping |
| Official vanilla | grouped | `0.03116` | **`4.60`** | **correct grouping** |
| Local production GGUF | contiguous | `0.03098` | **`4.61`** | **correct grouping** |
| Local production GGUF | grouped | `0.00008` | `1.40` | wrong grouping |

**Conclusion:** the exact production DiT QKV is already **contiguous `[Q_all | K_all | V_all]`**. ComfyUI's current three-way split is correct for this file. The raw h3.c and ComfyUI behaviours differ because they consume different checkpoint contracts, not because one runtime is necessarily wrong.

No QKV patch should be applied.

Detailed evidence: [`QKV-LAYOUT-AUDIT.md`](QKV-LAYOUT-AUDIT.md).

## 32 GB unified-memory constraint

This machine has 32 GB **unified memory**, not dedicated 32 GB GPU VRAM. CPU, GPU and system processes share the same pool.

Operational policy:

- parallelise reasoning, static code inspection and log analysis only;
- run one heavy H3/Qwen/Metal workload at a time;
- serialise all renders;
- preserve Stage A / Stage B process isolation;
- record wall-clock time, memory pressure and swap around heavy runs;
- reduce resolution/frames/steps before adding concurrency.

h3.c's `--ssd-streaming` mode is especially relevant because it explicitly trades speed for lower DiT residency while keeping original BF16 weights.

## FL2VA vs Ref2VA

The remaining investigation is now semantic rather than implementation-level:

- **FL2VA:** first/last-frame anchoring;
- **Ref2VA:** ordered subject/reference media.

A long-video identity drift under first-frame anchoring does not establish that Ref2VA is broken.

The next independent comparison is:

1. ComfyUI FL2VA;
2. h3.c FL2VA;
3. h3.c Ref2VA.

If FL2VA drifts in both runtimes but Ref2VA preserves identity, the earlier problem was a task-choice mismatch rather than a code defect.

## External performance evidence

A YouTube demonstration reported h3.c running MiniMax-H3 on an M3 Max with 36 GB unified memory using SSD streaming. It reported approximately 181–182 seconds for the H3 DiT portion of a short 20-step run.

This is useful corroboration that low-memory Apple Silicon execution is practical, but it remains **external/unverified benchmark evidence** for this repository because the exact comparable resolution/revision/settings have not been reconstructed.

A separate h3.c + lightx2v/Turbo lead remains optimisation-only evidence and stays outside the correctness critical path.

## Final implementation conclusion

The code-defect investigation is closed for the exercised path:

- vision tower: exonerated;
- Qwen3VL/LLM propagation: exonerated;
- tested DiT reference routing: exonerated;
- production DiT QKV interpretation: exonerated.

No production patch is justified by the current evidence.

The next phase is **independent h3.c validation**, not further generic bug hunting. Its goals are task-semantics comparison (FL2VA vs Ref2VA), independent runtime confirmation, and M2 Max 32 GB memory/performance measurement.
