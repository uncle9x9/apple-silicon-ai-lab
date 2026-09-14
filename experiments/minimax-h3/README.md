# MiniMax-H3 on MacBook Pro M2 Max 32 GB

Status: **active investigation**

This case study records measured MiniMax-H3 / ComfyUI experiments on a MacBook Pro M2 Max with 30-core GPU and 32 GB unified memory.

## Current decision

### Known-good baseline

**Qwen3-VL-4B BF16 + ClipProj v3.1**, using hard Stage A / Stage B process isolation.

Measured 56-frame run:

- 640×352
- 56 frames @ 24 fps
- 2.333 s output
- conditioning: ~18–20 s
- sampling/decode: ~454–501 s depending on run
- peak resident memory during bake-off: ~30.98 GB

This is currently the safest reference-image baseline. Reference adherence is imperfect, but the subject remains broadly on-model instead of collapsing into an unrelated identity.

### Native 32B Q4

Native Q4 conditioning is computationally viable under staged execution, but the current throwaway native vision-conditioning path is **not trusted for reference-image work**.

Observed failure pattern:

- frame 0 correctly reflects the supplied reference through the VAE/keyframe path;
- later frames abandon the reference;
- the same unrelated photorealistic subject appears in both Q2 and Q4 tests;
- therefore the failure is unlikely to be Q2-specific quantisation noise.

Current hypothesis: a correctness defect exists somewhere along the H3 native vision-conditioning chain, including possible vision tensor remapping, deepstack ordering, token/tag metadata, preprocessing, or conditioning serialisation/deserialisation.

### Native 32B Q2

**RETIRED.** Q2 showed no useful advantage over Q4 in the controlled bake-off. It was slower because unsupported IQ2/IQ3 sub-blocks fell back to CPU/NumPy on the tested MPS path, while reference-conditioning failed in the same way as Q4.

### EasyCache

Measured on Native Q4:

- wall-time improvement: about 10% in the tested run;
- sampling loop reported 2/8 skipped steps and ~1.33× loop speedup;
- peak memory reached about 32.02 GB.

Decision: **optional only**, not a default on this 32 GB machine because the memory margin is too small.

## Key architecture finding

The most important engineering result so far is not a model choice but a memory-lifetime strategy.

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

On unified-memory Macs, moving tensors from MPS to CPU does not necessarily release physical memory. Process lifetime boundaries can therefore be a practical reliability tool.

## Measured frame-length scaling

At 640×352 / 8 steps:

| Frames | Video length | Sampling | Total | Status |
|---:|---:|---:|---:|---|
| 56 | 2.333 s | 453.7 s in known-good run | ~7.9 min | MEASURED |
| 124 | 5.167 s | 928 s | 16 min 54 s | MEASURED |

Do not treat these numbers as universal performance claims; they describe this machine and this workflow configuration.

## Reference-conditioning investigation

Next priority: stop expensive end-to-end video bake-offs and perform a conditioning-equivalence investigation across:

1. official / known-good H3 reference path;
2. H3ClipLoaderAny-style GGUF + mmproj path;
3. the throwaway `H3NativeClipLoader` path;
4. the `h3_condio` serialise → deserialise boundary.

Compare, for the same prompt and reference image:

- vision tensor key mapping;
- missing / unexpected keys;
- deepstack ordering;
- image preprocessing shape, dtype and range;
- visual-token counts;
- deepstack feature shapes;
- final conditioning shapes and statistics;
- MiniMax-specific token/tag metadata and positions;
- conditioning before and after serialisation.

The first divergence should be treated as the primary root-cause lead.

## Important historical correction

An earlier rule hard-banned 32B on this 32 GB Mac. That conclusion was too broad and is now **RETIRED**.

The better policy is:

> Native 32B conditioning may be feasible on constrained-memory Apple Silicon when encoder and DiT lifetimes are staged. Do not assume a single-process OOM proves that the model itself cannot run.

## Artefacts to publish next

- validated Stage A / Stage B workflow JSONs;
- downloader / profile script;
- run script;
- concise M2 Max 32 GB runbook;
- benchmark table with raw logs;
- representative reference-conditioning frames;
- `h3_condio` implementation and audit notes;
- root-cause report for the native reference-conditioning bug.
