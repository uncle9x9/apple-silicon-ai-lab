# MiniMax-H3 on MacBook Pro M2 Max 32 GB

Status: **active — reference routing exonerated; current P0 is H3 DiT QKV checkpoint semantics, with h3.c as an independent oracle**

This case study records measured MiniMax-H3 experiments on a MacBook Pro M2 Max with 30-core GPU and 32 GB unified memory.

See also:

- [`INVESTIGATION-2026-09-14.md`](INVESTIGATION-2026-09-14.md) — Phase 1–5 evidence;
- [`H3C-VALIDATION.md`](H3C-VALIDATION.md) — h3.c oracle track and QKV-layout gate;
- [`DECISIONS.md`](DECISIONS.md) — engineering decisions;
- [`CONTENT-NOTES.md`](CONTENT-NOTES.md) — concise factual source material for future blog/YouTube content.

## Current conclusion

The original Native 32B reference-adherence failure is **not explained by a broken reference-routing path**.

Controlled experiments now establish:

- production GGMLOps vs plain ops: bit-identical;
- local GGUF/mmproj vision weights vs official MiniMax-H3 encoder shard: bit-identical;
- Qwen3VL vision QKV ordering: correct;
- DeepStack `8/16/24 -> 0/1/2`: correct;
- image-token placement, masks and DeepStack injection: consistent;
- full 50-layer Qwen3VL/LLM path: reference-sensitive;
- the tested H3 DiT path: also reference-sensitive.

A controlled production DiT test at 288x512 / 5 frames / 2 steps compared real reference vs ablated reference and measured:

- final predicted-latent cosine: `0.8372`;
- relative L2 difference: `0.549`;
- progressive divergence through DiT blocks;
- no NaN/Inf.

Therefore the reference is not silently dropped before or inside the tested DiT path.

## Current P0: DiT QKV checkpoint layout

Independent upstream `antirez/h3.c` documents the released MiniMax-H3 DiT checkpoint as storing QKV rows **interleaved per attention head**. h3.c consumes that layout directly.

Current ComfyUI MiniMax-H3 attention performs a conventional three-way split of `qkv_proj(x)` into contiguous Q/K/V blocks.

This is a concrete semantic difference, but **not yet proof of a ComfyUI bug**, because the local GGUF conversion may already reorder the raw checkpoint into contiguous `[Q_all | K_all | V_all]` layout.

The next decisive test is therefore not another full render. It is an exact tensor-level audit of the production-loaded DiT QKV tensor and its converter lineage.

See [`H3C-VALIDATION.md`](H3C-VALIDATION.md).

## Known-good baseline

**Qwen3-VL-4B BF16 + ClipProj v3.1**, using hard Stage A / Stage B process isolation.

Measured 56-frame run:

- 640x352;
- 56 frames @ 24 fps;
- 2.333 s output;
- conditioning: approximately 18–20 s;
- sampling/decode: approximately 454–501 s depending on run;
- peak resident memory observed during bake-off: approximately 30.98 GB.

This remains the safest tested lane while the Native 32B task/checkpoint semantics are being validated.

## Native 32B Q4

Native Q4 conditioning is computationally viable with staged execution.

Earlier full-video runs showed:

- frame 0 reflects the supplied reference;
- later frames lose reference adherence;
- the same unrelated subject appeared under Q2 and Q4.

The earlier hypotheses that the native vision tower, Qwen3VL propagation or DiT reference routing were broken are now retired for the tested path.

Native Q4 remains **experimental** because the current open question is deeper: whether the converted DiT QKV layout matches the semantics assumed by ComfyUI.

## Native 32B Q2

**RETIRED.** Q2 showed no useful advantage over Q4 in the controlled bake-off. It was slower because unsupported IQ2/IQ3 sub-blocks fell back to CPU/NumPy on the tested MPS path, while the same reference-adherence failure reproduced under Q4.

## EasyCache

Measured on Native Q4:

- wall-time improvement: about 10% in the tested run;
- sampling loop: 2/8 steps skipped and approximately 1.33x loop speedup;
- peak memory: approximately 32.02 GB.

Decision: **optional only**, not a default on this 32 GB machine because the memory margin is too small.

## Key architecture finding: process lifetime is part of memory management

Single-process execution kept the text encoder resident while loading the DiT and was killed by macOS under memory pressure. The reliable workaround was:

```text
Process A
  text / vision encoder
      ↓
  conditioning + latent
      ↓
  serialise
      ↓
  process exits

Process B
  reload conditioning
      ↓
  load H3 DiT
      ↓
  sample
      ↓
  VAE decode
      ↓
  MP4
```

On Apple Silicon, CPU and GPU share unified memory. Moving tensors from MPS to CPU does not necessarily release physical RAM. Process lifetime boundaries can therefore be a practical reliability tool.

Operational rule:

> **Parallel brains, serial GPU.** Parallelise code inspection and reasoning, but run only one heavy H3/Qwen/Metal workload or render at a time.

## Measured frame-length scaling

At 640x352 / 8 steps:

| Frames | Video length | Sampling | Total | Status |
|---:|---:|---:|---:|---|
| 56 | 2.333 s | 453.7 s in known-good run | approximately 7.9 min | MEASURED |
| 124 | 5.167 s | 928 s | 16 min 54 s | MEASURED |

Do not treat these values as universal performance claims; they describe this machine and this workflow configuration.

## Phase 5 micro-test

The smallest exact 9:16 mechanically aligned diagnostic used:

- 288x512;
- 5 frames;
- 2 steps;
- real reference vs ablated reference;
- serial execution.

The real-reference Stage B run took approximately 96 s total. This timing is a diagnostic point, not a quality benchmark.

## h3.c validation track

`antirez/h3.c` is now a required independent reference implementation, not merely an acceleration lead.

At pinned upstream HEAD `8974cc055ea9c02fcd14cc27dfda3e1027c05153`, it provides:

- original-BF16 MiniMax-H3 inference on Apple Silicon;
- FL2VA first/last-frame conditioning;
- ordered Ref2VA references;
- SSD streaming for low-memory operation;
- validated 22-frame development presets;
- direct grouped/per-head H3 DiT QKV handling.

The planned sequence is:

1. resolve the exact production GGUF DiT QKV layout;
2. if contiguous, close the QKV hypothesis;
3. then reproduce FL2VA in h3.c;
4. run genuine Ref2VA separately;
5. compare ComfyUI FL2VA vs h3.c FL2VA vs h3.c Ref2VA;
6. only then revisit optimisation such as Turbo/lightx2v.

## FL2VA vs Ref2VA

Do not equate first-frame anchoring with subject-reference conditioning.

- **FL2VA:** first/last-frame anchors;
- **Ref2VA:** ordered subject/reference media.

If FL2VA drifts but Ref2VA preserves identity, the earlier experiment was testing the wrong task family for the intended identity-preservation objective rather than exposing a reference-routing bug.

## Historical corrections

Retired conclusions:

1. **“32B is impossible on 32 GB.”** False as a blanket rule. Simultaneous residency was the main failure mode; staged execution makes Native 32B conditioning viable.
2. **“The native vision tower is numerically wrong.”** False. The local GGUF/mmproj vision path is bit-identical to the official H3 encoder.
3. **“Reference information is being dropped before the DiT.”** False for the tested graph. The Qwen3VL path and final predicted latent both respond materially to the reference.

## Artefacts still to publish

- validated Stage A / Stage B workflow JSONs;
- downloader / profile scripts;
- run scripts;
- concise M2 Max 32 GB runbook;
- raw benchmark logs and exact generation-time reconstruction;
- representative reference-conditioning frames;
- `h3_condio` implementation and audit notes;
- production-loaded DiT QKV layout proof;
- h3.c build/model manifest;
- controlled FL2VA/Ref2VA oracle outputs.
