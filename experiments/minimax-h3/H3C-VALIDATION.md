# h3.c Validation Track

Date: 2026-09-14  
Reference machine: MacBook Pro M2 Max, 30-core GPU, 32 GB unified memory  
Status: **active — h3.c adopted as an independent MiniMax-H3 correctness oracle; DiT QKV layout is the current P0 question**

## Why this track exists

The ComfyUI investigation has now exonerated the complete reference-conditioning path that was previously suspected:

- Qwen3VL vision weights and mmproj loading are correct against the official MiniMax-H3 checkpoint;
- GGMLOps and plain-ops execution are bit-identical;
- Qwen3VL token placement, DeepStack injection and the full 50-layer LLM are reference-sensitive;
- the production H3 DiT path is also reference-sensitive in a controlled real-reference vs ablated test.

The remaining question is therefore no longer simply whether the reference reaches the DiT. The next task is to verify that the DiT itself interprets the released checkpoint semantics correctly.

`antirez/h3.c` is useful here because it is an independent Apple-Silicon implementation using the original MiniMax-H3 BF16 checkpoints rather than the current ComfyUI/GGUF execution stack.

## Phase 5 result — DiT conditioning path is reference-sensitive

A controlled production Stage-B run compared:

- A: real reference image;
- C: reference ablated;
- identical seed, prompt and sampling configuration otherwise.

Diagnostic shape:

- 288x512 portrait;
- 5 frames;
- 2 denoising steps;
- runs executed serially on the 32 GB machine.

Measured final predicted latent:

- cosine similarity A vs C: `0.8372`;
- relative L2 difference: `0.549`;
- no NaN/Inf observed.

Intermediate DiT checkpoints also diverged progressively.

The static trace confirmed that MiniMax conditioning, token tags, keyframe rows and video rows are packed into one sequence and are available to the DiT attention path. No routing or masking defect was found in the tested graph.

A real conditional defect was found in ComfyUI's scheduled CLIP path: when `use_clip_schedule=True`, one branch can rebuild the output dictionary without preserving extras such as `minimax_token_tags`. This did **not** affect the exercised workflow; captured DiT input contained populated image tags.

**Conclusion:** the previously exercised DiT/keyframe path is reference-sensitive. The original long-video identity drift should not be attributed to a silent reference-routing failure without new evidence.

## Independent oracle: antirez/h3.c

Repository:

- upstream: `https://github.com/antirez/h3.c`
- pinned HEAD verified during this investigation: `8974cc055ea9c02fcd14cc27dfda3e1027c05153`
- commit message: `Clarify SSD streaming memory and speed tradeoff`

At that revision, h3.c provides:

- native MiniMax-H3 inference for Apple Silicon;
- prompt-to-video/audio;
- first/last-frame FL2VA conditioning;
- ordered Ref2VA image/video/audio references;
- original-BF16 SSD-streaming mode;
- short 22-frame development presets;
- explicit low-resolution development modes.

### 32 GB relevance

`--ssd-streaming` keeps only a small rotating set of DiT blocks resident and reads subsequent blocks from SSD while the GPU runs the current block. Upstream reports this as an explicit memory/speed trade-off and recommends omitting `--show` on low-memory systems because preview-VAE residency adds substantial memory pressure.

This supports the local Stage-A / Stage-B finding that **weight residency lifetime is a primary engineering constraint on Apple Silicon**, not simply nominal model size.

Operational policy for this repository remains:

> **Parallel brains, serial GPU.**

Only one heavy H3/Qwen/Metal workload or render should run at a time on the 32 GB reference machine.

## Critical new discrepancy: H3 DiT QKV layout

The upstream h3.c README explicitly states that the released H3 DiT checkpoint stores QKV rows **interleaved per attention head** and that an earlier identity interpretation caused noisy diagnostic outputs.

h3.c implements grouped/per-head QKV consumption in its DiT path. Its tests also describe the released H3 checkpoint as emitting QKV interleaved per head.

By contrast, current ComfyUI MiniMax-H3 attention performs a conventional three-way split:

```python
q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
```

This is a real implementation-level semantic difference, but it is **not yet proof of a ComfyUI defect**.

The local production DiT is a converted GGUF. Its conversion pipeline may already reorder the raw per-head-interleaved H3 QKV tensor into contiguous `[Q_all | K_all | V_all]` rows before ComfyUI executes the split above.

Therefore the current P0 question is:

> **What is the actual QKV row layout of the exact production DiT tensor loaded by this workflow?**

Do not infer this from shape, mean, standard deviation, RMS or L2. A pure row permutation preserves aggregate statistics.

## Required P0 validation

For the exact local DiT GGUF used by the successful Stage-B run:

1. Record filename, size, SHA-256, source and converter lineage where recoverable.
2. Trace the complete loader/converter/state-dict path for `blocks.*.attn.qkv_proj.weight`.
3. Extract a real production-loaded QKV tensor after all dequantisation/remapping.
4. Compare two explicit semantic interpretations:
   - contiguous: `[Q_all | K_all | V_all]`;
   - raw grouped: `[h0 Q,K,V][h1 Q,K,V]...`.
5. Use row fingerprints, row-wise cosine matching and exact permutation recovery rather than aggregate norms.
6. Compare against h3.c's grouped-QKV contract and, if required, a minimal official checkpoint tensor/shard.

Decision states:

- **contiguous confirmed**: ComfyUI's current split is correct for this converted file; close the hypothesis and resume FL2VA/Ref2VA oracle renders;
- **interleaved confirmed**: this is a concrete DiT correctness defect; demonstrate it on one block before patching production code;
- **unresolved**: download only the minimum official shard/tensor required for byte-level verification rather than the full H3 repository.

## FL2VA vs Ref2VA: keep task semantics separate

h3.c exposes two distinct conditioning modes:

- first/last-frame anchors for FL2VA;
- ordered `--ref-image` / reference media for Ref2VA.

These must not be treated as interchangeable when evaluating identity preservation.

After QKV layout is resolved, the planned independent oracle campaign is:

1. reproduce the existing first-frame/FL2VA semantics in h3.c;
2. run genuine Ref2VA with the correct Ref2VA checkpoint family;
3. compare ComfyUI FL2VA vs h3.c FL2VA vs h3.c Ref2VA;
4. separate identity-retention conclusions from runtime/optimisation conclusions.

## Minimal h3.c development shapes

Relevant upstream guidance at the pinned revision:

- width and height must be multiples of 32;
- 512x512 is the repeatedly validated development size;
- 256x256 is an explicitly supported fast-preview size with special low-resolution RoPE handling;
- 22 frames is the standard short development clip (~0.917 s at 24 fps);
- 4-step runs are useful for rapid diagnostics but are not final-quality references;
- 20 steps is the normal default path;
- 50 steps is the slow reference-quality oracle path.

For exact portrait 9:16, `288x512` is mechanically aligned to multiples of 32, but should be treated as a diagnostic shape rather than an upstream validated quality point.

## External YouTube evidence

One independent YouTube demonstration reported MiniMax-H3 running through h3.c on an M3 Max with 36 GB unified memory using SSD streaming and the official large checkpoint. It reported approximately 181–182 seconds for the H3 DiT portion of a short 20-step run.

This is useful corroboration that low-memory Apple Silicon execution is practical, but it is **external/unverified benchmark evidence** for this repository because the exact resolution, model revision and full comparable settings were not reconstructed from the transcript alone.

A separate community report about h3.c plus a lightx2v/Turbo patch remains an optimisation lead, not correctness evidence. Turbo work stays out of the critical path until the QKV/task-semantics questions are closed.

## Source references

- h3.c upstream: `https://github.com/antirez/h3.c`
- pinned h3.c commit: `8974cc055ea9c02fcd14cc27dfda3e1027c05153`
- h3.c README section: checkpoint layout and media pipeline
- h3.c grouped-QKV implementation: `h3_dit.c`, `h3_gpu.h`, `h3_shaders.metal`
- h3.c grouped-QKV validation: `tests/test_bf16.c`, `tests/test_real_dit_block.c`
- ComfyUI MiniMax-H3 attention: `comfy/ldm/minimax/model.py`

## Current status

The original reference-conditioning investigation has moved through two distinct conclusions:

1. **Reference routing is functioning** through Qwen3VL and through the tested DiT path.
2. **DiT checkpoint semantics are now under audit**, with QKV row layout the highest-information unresolved question.

No production patch should be made until the actual loaded QKV layout is established by exact tensor-level evidence.
