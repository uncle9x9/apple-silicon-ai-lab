# h3.c Validation Track

Date: 2026-09-14 to 2026-09-15  
Reference machine: MacBook Pro M2 Max, 30-core GPU, 32 GB unified memory  
Status: **active — QKV layout gate resolved; controlled FL2VA/Ref2VA oracle campaign is next**

## Why this track exists

The ComfyUI implementation investigation has now exonerated the complete reference-conditioning path that was previously suspected:

- Qwen3VL vision weights and mmproj loading are correct against the official MiniMax-H3 checkpoint;
- GGMLOps and plain-ops execution are bit-identical;
- Qwen3VL token placement, DeepStack injection and the full 50-layer LLM are reference-sensitive;
- the production H3 DiT path is reference-sensitive in a controlled real-reference vs ablated test;
- the production DiT QKV checkpoint layout has been proven compatible with ComfyUI.

`antirez/h3.c` remains valuable as an independent Apple-Silicon implementation using original MiniMax-H3 BF16 checkpoints rather than the current ComfyUI/GGUF stack.

Its purpose is now **independent validation, task-semantics comparison and low-memory performance measurement**, not generic bug hunting.

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

## Independent oracle: antirez/h3.c

Repository:

- upstream: `https://github.com/antirez/h3.c`
- pinned HEAD verified during this investigation: `8974cc055ea9c02fcd14cc27dfda3e1027c05153`

At that revision, h3.c provides:

- native MiniMax-H3 inference for Apple Silicon;
- prompt-to-video/audio;
- first/last-frame FL2VA conditioning;
- ordered Ref2VA image/video/audio references;
- original-BF16 SSD-streaming mode;
- short 22-frame development presets;
- explicit low-resolution development modes.

### 32 GB relevance

`--ssd-streaming` lowers DiT residency by reading blocks from SSD during execution. This is a deliberate memory/speed trade-off and independently supports the local finding that **weight residency lifetime is a primary engineering constraint on Apple Silicon**, not simply nominal model size.

Operational policy:

> **Parallel brains, serial GPU.**

Only one heavy H3/Qwen/Metal workload or render should run at a time on the 32 GB reference machine.

## Resolved gate — H3 DiT QKV layout

h3.c documents the released raw MiniMax-H3 DiT checkpoint as storing QKV rows **interleaved per attention head**.

Current ComfyUI MiniMax-H3 attention performs a conventional three-way split:

```python
q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
```

The important caveat was that Comfy-oriented checkpoints may already be regrouped before runtime.

The exact production GGUF was therefore audited rather than assuming that all H3 checkpoints share one physical layout.

### Loader trace

For the production DiT GGUF:

- `UnetLoaderGGUF.load_unet()` uses `gguf_sd_loader(unet_path)`;
- the UNET/DiT path applies no MiniMax-specific QKV permutation;
- `blocks.0.attn.qkv_proj.weight` retains its PyTorch key;
- `comfy.gguf.orig_shape.*` metadata is consistent with generic ComfyUI-GGUF quantisation.

The GGUF loader therefore preserves the pre-quantisation row order.

### Structural proof

The official vanilla Ref2VA tensor and the local production tensor were tested independently under both candidate layouts.

| Tensor | Hypothesis | matched-minus-mismatched `|cosine|` gap | Q·Kᵀ diagonal/off-diagonal | Verdict |
|---|---|---:|---:|---|
| Official vanilla | contiguous | `0.00026` | `0.96` | wrong |
| Official vanilla | grouped | `0.03116` | **`4.60`** | **correct** |
| Local production GGUF | contiguous | `0.03098` | **`4.61`** | **correct** |
| Local production GGUF | grouped | `0.00008` | `1.40` | wrong |

**Result:**

- official raw checkpoint: grouped/per-head interleaved;
- local production checkpoint: contiguous `[Q_all | K_all | V_all]`.

The local pruned checkpoint had already been regrouped before generic GGUF quantisation. ComfyUI's current split is therefore correct for the exact production file.

The QKV bug hypothesis is closed. No patch is warranted.

Detailed evidence: [`QKV-LAYOUT-AUDIT.md`](QKV-LAYOUT-AUDIT.md).

## FL2VA vs Ref2VA: the next real question

h3.c exposes two distinct conditioning modes:

- first/last-frame anchors for **FL2VA**;
- ordered `--ref-image` / reference media for **Ref2VA**.

These must not be treated as interchangeable when evaluating identity preservation.

The original symptom — frame 0 matches the reference and later frames drift — may be consistent with FL2VA first-frame anchoring rather than a code defect.

## Controlled h3.c campaign

### Phase A — build and pin

- clone the verified upstream revision;
- record commit SHA, toolchain, macOS version and build command;
- build unmodified;
- do not patch before baseline validation.

### Phase B — model inventory

Before downloading anything large:

- inventory existing local MiniMax-H3 files;
- determine exact FL2VA and Ref2VA files required by the pinned h3.c revision;
- reuse/symlink only when byte identity and semantic compatibility are proven;
- calculate unique additional download bytes first.

### Phase C — h3.c FL2VA

Reproduce the existing first-frame semantics as closely as practical:

- same reference image;
- equivalent prompt;
- portrait orientation;
- short but meaningful temporal length, preferably 22 frames;
- fixed seed where supported;
- `--ssd-streaming`;
- no preview mode on the 32 GB machine.

Run a smoke test first, then a normal 20-step correctness run if the smoke path is valid.

### Phase D — h3.c Ref2VA

Run the same subject/reference using the correct Ref2VA checkpoint and `--ref-image`.

Do not mix Ref2VA references with FL2VA first/last-frame anchors.

### Phase E — compare

Compare:

1. ComfyUI FL2VA;
2. h3.c FL2VA;
3. h3.c Ref2VA.

Evaluate separately:

- frame-0/reference match;
- identity retention over time;
- composition;
- motion;
- hallucination/drift;
- wall-clock time;
- memory pressure;
- swap growth.

Do not claim speed ratios unless resolution, frames, steps, layers and sampling semantics are genuinely comparable.

## Decision rules

### ComfyUI FL2VA ≈ h3.c FL2VA, Ref2VA materially better

Interpret the original problem primarily as **task semantics**: FL2VA anchoring was being asked to provide Ref2VA-like identity preservation.

### h3.c FL2VA materially better than ComfyUI FL2VA

Reopen a focused runtime semantic-differential audit. Do not reopen already exonerated vision/LLM hypotheses generically.

### FL2VA and Ref2VA both drift similarly

Treat the remaining limitation as a model/sampling/reference-distribution issue unless new evidence contradicts that conclusion.

## Minimal h3.c development shapes

Relevant upstream guidance at the pinned revision:

- width and height must be multiples of 32;
- 512x512 is a repeatedly validated development size;
- 256x256 is a fast-preview size with special low-resolution handling;
- 22 frames is the standard short development clip;
- low-step runs are useful for diagnostics but not final quality judgements;
- 20 steps is the normal baseline path.

For exact portrait 9:16, `288x512` is mechanically aligned to multiples of 32, but should be treated as a diagnostic shape rather than an upstream validated quality point.

## External YouTube evidence

One independent YouTube demonstration reported MiniMax-H3 running through h3.c on an M3 Max with 36 GB unified memory using SSD streaming and the official large checkpoint. It reported approximately 181–182 seconds for the H3 DiT portion of a short 20-step run.

This remains **external/unverified benchmark evidence** because the exact resolution, revision and fully comparable settings have not been reconstructed.

A separate h3.c + lightx2v/Turbo path remains an optimisation lead only. Turbo stays outside the baseline until the FL2VA/Ref2VA comparison is established.

## Current status

The implementation-bug investigation is closed for the exercised ComfyUI path.

The active h3.c work is now:

1. independent FL2VA confirmation;
2. genuine Ref2VA identity-preservation test;
3. M2 Max 32 GB memory/runtime characterisation;
4. only then, optimisation work.
