# MiniMax-H3 on MacBook Pro M2 Max 32 GB

Status: **active investigation — upstream reference-conditioning path exonerated; current blocker is H3 DiT/keyframe integration**

This case study records measured MiniMax-H3 / ComfyUI experiments on a MacBook Pro M2 Max with 30-core GPU and 32 GB unified memory.

See also:

- [`INVESTIGATION-2026-09-14.md`](INVESTIGATION-2026-09-14.md) — full Phase 1–4 root-cause evidence;
- [`DECISIONS.md`](DECISIONS.md) — current engineering decisions;
- [`CONTENT-NOTES.md`](CONTENT-NOTES.md) — concise source material for future blog/YouTube content.

## Current conclusion

The earlier Native 32B reference-adherence failure has **not** been proven to originate in the vision encoder or Qwen3VL conditioning stack. Four controlled phases now exonerate that entire upstream path:

- production GGMLOps vs plain ops: bit-identical;
- local GGUF/mmproj vision weights vs official MiniMax-H3 encoder shard: bit-identical;
- QKV ordering: correct;
- DeepStack `8/16/24 -> 0/1/2`: correct;
- image-token placement, masks and DeepStack injection: internally consistent;
- full 50-layer Qwen3VL/LLM path: demonstrably reference-sensitive.

The current blocker is downstream:

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

## Known-good baseline

**Qwen3-VL-4B BF16 + ClipProj v3.1**, using hard Stage A / Stage B process isolation.

Measured 56-frame run:

- 640x352;
- 56 frames @ 24 fps;
- 2.333 s output;
- conditioning: approximately 18–20 s;
- sampling/decode: approximately 454–501 s depending on run;
- peak resident memory observed during bake-off: approximately 30.98 GB.

This remains the safest tested reference-image baseline while the Native 32B downstream conditioning path is under investigation.

## Native 32B Q4

Native Q4 conditioning is computationally viable with staged execution.

Earlier runs showed:

- frame 0 reflects the supplied reference through the VAE/keyframe path;
- later frames lose reference adherence;
- the same unrelated subject appeared under Q2 and Q4.

The earlier hypothesis that the native vision tower itself was corrupt is now **RETIRED**. The exact official H3 encoder shard proved that the local GGUF/mmproj weights are correct.

Native Q4 should therefore remain **experimental**, not rejected. The next question is whether H3 DiT/keyframe integration consumes the reference-sensitive conditioning correctly.

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

Operational rule for this machine:

> **Parallel brains, serial GPU.** Parallelise code inspection and reasoning, but run only one heavy H3/Qwen/Metal workload or render at a time.

## Measured frame-length scaling

At 640x352 / 8 steps:

| Frames | Video length | Sampling | Total | Status |
|---:|---:|---:|---:|---|
| 56 | 2.333 s | 453.7 s in known-good run | approximately 7.9 min | MEASURED |
| 124 | 5.167 s | 928 s | 16 min 54 s | MEASURED |

Do not treat these values as universal performance claims; they describe this machine and this workflow configuration.

## Current investigation plan

Instrument the production H3 DiT/keyframe path and compare multiple references while holding prompt, seed and sampler constant.

Measure whether changing the reference image materially changes:

1. conditioning actually passed into the DiT;
2. cross-attention or equivalent conditioning consumption;
3. representative early/mid/late DiT blocks;
4. final predicted latent/noise.

If numerical evidence shows a reference-sensitive DiT path, run a minimal diagnostic micro-render:

- portrait 9:16;
- smallest valid H3-aligned resolution determined from code/RUNBOOK;
- 5 frames if `frames % 17 == 5` is confirmed;
- 2–4 steps;
- fixed seed, prompt and sampler;
- materially different references plus a safe ablated control;
- renders strictly serial.

This is a diagnostic smoke test, not a quality benchmark.

## External acceleration lead

A community report describes an Apple-Silicon H3 path using `h3.c` plus a `lightx2v` Turbo patch, with step reduction from 12 to 8 or 4 and materially shorter reported runtimes.

This is currently **UNVERIFIED external evidence** for this repository. The reported setup appears to assume substantially more memory/storage than the 32 GB reference machine, so correctness and acceleration work remain separate until reproduced locally.

## Important historical corrections

Two earlier conclusions are now retired:

1. **“32B is impossible on 32 GB.”** False as a blanket rule. The main failure mode was simultaneous encoder + DiT residency; staged execution makes Native 32B conditioning viable.
2. **“The native vision tower is numerically wrong.”** False. The local GGUF/mmproj path is bit-identical to the official MiniMax-H3 vision weights, and the apparently large DeepStack magnitude is intentional H3 behaviour.

## Artefacts still to publish

- validated Stage A / Stage B workflow JSONs;
- downloader / profile script;
- run script;
- concise M2 Max 32 GB runbook;
- raw benchmark logs and exact generation-time reconstruction;
- representative reference-conditioning frames;
- `h3_condio` implementation and audit notes;
- DiT/keyframe integration trace and final root-cause report.
