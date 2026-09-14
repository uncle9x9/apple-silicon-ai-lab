# MiniMax-H3 on MacBook Pro M2 Max 32 GB

Status: **implementation investigation closed; independent h3.c FL2VA/Ref2VA validation is next**

This case study records measured MiniMax-H3 experiments on a MacBook Pro M2 Max with 30-core GPU and 32 GB unified memory.

See also:

- [`INVESTIGATION-2026-09-14.md`](INVESTIGATION-2026-09-14.md) — Phase 1–6 evidence;
- [`QKV-LAYOUT-AUDIT.md`](QKV-LAYOUT-AUDIT.md) — resolved DiT checkpoint-layout audit;
- [`H3C-VALIDATION.md`](H3C-VALIDATION.md) — independent h3.c oracle track;
- [`DECISIONS.md`](DECISIONS.md) — engineering decisions;
- [`CONTENT-NOTES.md`](CONTENT-NOTES.md) — factual source material for future blog/YouTube content.

## Current conclusion

No actionable code defect has been found in the exercised reference-conditioning path.

Controlled experiments establish:

- production GGMLOps vs plain ops: bit-identical;
- local GGUF/mmproj vision weights vs official MiniMax-H3 encoder shard: bit-identical;
- Qwen3VL vision QKV ordering: correct;
- DeepStack `8/16/24 -> 0/1/2`: correct;
- image-token placement, masks and DeepStack injection: consistent;
- full 50-layer Qwen3VL/LLM path: reference-sensitive;
- the tested H3 DiT path: reference-sensitive;
- the exact production DiT QKV layout is **contiguous `[Q_all | K_all | V_all]`**, matching ComfyUI's three-way split.

A controlled production DiT test at 288x512 / 5 frames / 2 steps compared real reference vs ablated reference and measured:

- final predicted-latent cosine: `0.8372`;
- relative L2 difference: `0.549`;
- progressive divergence through DiT blocks;
- no NaN/Inf.

Therefore the original long-video identity drift is no longer best explained by a silent routing, loading or QKV-layout defect. The remaining questions are now **task semantics and model/workflow behaviour**, especially FL2VA first-frame anchoring versus genuine Ref2VA subject/reference conditioning.

## Resolved QKV checkpoint-layout audit

`antirez/h3.c` correctly documents the released raw MiniMax-H3 DiT checkpoint as using per-head interleaved QKV rows. Current ComfyUI consumes a contiguous `[Q_all | K_all | V_all]` layout.

This looked like a possible implementation discrepancy, but the production GGUF was audited directly.

Result:

- official vanilla checkpoint: **grouped/per-head interleaved**;
- local production GGUF: **contiguous `[Q_all | K_all | V_all]`**;
- generic GGUF loading performs no MiniMax-specific permutation;
- therefore the local pruned checkpoint had already been regrouped before quantisation;
- ComfyUI's current `Attention.forward()` split is correct for the file actually used in production.

Structural evidence:

| Tensor | Correct hypothesis | matched-minus-mismatched `|cosine|` gap | Q·Kᵀ diagonal/off-diagonal |
|---|---|---:|---:|
| Official vanilla | grouped | `0.03116` | `4.60` |
| Local production GGUF | contiguous | `0.03098` | `4.61` |

See [`QKV-LAYOUT-AUDIT.md`](QKV-LAYOUT-AUDIT.md).

## Known-good baseline

**Qwen3-VL-4B BF16 + ClipProj v3.1**, using hard Stage A / Stage B process isolation.

Measured 56-frame run:

- 640x352;
- 56 frames @ 24 fps;
- 2.333 s output;
- conditioning: approximately 18–20 s;
- sampling/decode: approximately 454–501 s depending on run;
- peak resident memory observed during bake-off: approximately 30.98 GB.

## Native 32B Q4

Native Q4 conditioning is computationally viable with staged execution.

Earlier full-video runs showed:

- frame 0 reflects the supplied reference;
- later frames lose reference adherence;
- the same unrelated subject appeared under Q2 and Q4.

Earlier hypotheses that the native vision tower, Qwen3VL propagation, DiT routing or DiT QKV interpretation were broken are now retired for the exercised path.

Native Q4 remains **experimental**, not because of an identified implementation defect, but because the intended identity-preservation task may require Ref2VA rather than FL2VA first-frame anchoring.

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

`antirez/h3.c` is retained as a required independent reference implementation, but its role has changed.

It is no longer being used primarily to hunt for a ComfyUI code defect. The QKV discrepancy has been resolved as a checkpoint-contract difference.

The next controlled campaign is:

1. build and pin h3.c unmodified;
2. inventory/reuse local model files before downloading anything large;
3. reproduce FL2VA first-frame semantics in h3.c;
4. run genuine Ref2VA using the correct checkpoint family;
5. compare ComfyUI FL2VA vs h3.c FL2VA vs h3.c Ref2VA;
6. measure wall time, memory pressure and swap on the M2 Max 32 GB machine;
7. keep Turbo/lightx2v optimisation outside the correctness baseline.

Pinned upstream revision already verified during the investigation:

`8974cc055ea9c02fcd14cc27dfda3e1027c05153`

## FL2VA vs Ref2VA

Do not equate first-frame anchoring with subject-reference conditioning.

- **FL2VA:** first/last-frame anchors;
- **Ref2VA:** ordered subject/reference media.

Current highest-value hypothesis:

> If ComfyUI FL2VA and h3.c FL2VA both drift, but h3.c Ref2VA preserves identity, the original experiment was testing the wrong task family for the intended identity-preservation objective rather than exposing a code defect.

## Historical corrections

Retired conclusions:

1. **“32B is impossible on 32 GB.”** False as a blanket rule. Simultaneous residency was the main failure mode; staged execution makes Native 32B conditioning viable.
2. **“The native vision tower is numerically wrong.”** False. The local GGUF/mmproj vision path is bit-identical to the official H3 encoder.
3. **“Reference information is being dropped before the DiT.”** False for the tested graph. The Qwen3VL path and final predicted latent both respond materially to the reference.
4. **“ComfyUI may be misreading raw interleaved H3 DiT QKV.”** False for the exact production checkpoint. The production GGUF is already contiguous and matches ComfyUI semantics.

## Artefacts still to publish

- validated Stage A / Stage B workflow JSONs;
- downloader / profile scripts;
- run scripts;
- concise M2 Max 32 GB runbook;
- raw benchmark logs and exact generation-time reconstruction;
- representative reference-conditioning frames;
- `h3_condio` implementation and audit notes;
- h3.c build/model manifest;
- controlled FL2VA/Ref2VA oracle outputs.
