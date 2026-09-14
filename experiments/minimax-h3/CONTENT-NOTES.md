# MiniMax-H3 M2 Max 32 GB — Content Notes

Purpose: factual source material for a future blog post, tutorial or YouTube video. Keep measured facts separate from interpretation.

## One-line story

A MiniMax-H3 reference-conditioning failure on a 32 GB M2 Max initially looked like a GGUF/vision-tower defect, but four controlled phases exonerated the entire Qwen3VL path and narrowed the blocker to H3 DiT/keyframe integration.

## What was difficult

- MiniMax-H3 is large relative to a 32 GB unified-memory machine.
- Encoder + DiT simultaneous residency can exceed the practical memory envelope.
- Cross-model activation magnitudes can look suspicious even when both models are correct.
- A working tensor load is not enough: semantic correctness must be tested against an authoritative oracle.
- Long video generations are expensive and can have low diagnostic value.

## Reusable engineering lessons

1. **Use process lifetime as a memory-management tool.** Stage A / Stage B isolation was more reliable than assuming CPU offload frees physical memory on Apple Silicon.
2. **Prefer discriminating tests over repeated renders.** Tensor-level A/B tests eliminated several hypotheses without paying the cost of a full generation.
3. **Do not compare activation magnitudes across different model variants as if they were a correctness oracle.** The earlier 8B comparison falsely made official H3 behaviour look abnormal.
4. **Verify against the exact official checkpoint whenever possible.** The official MiniMax-H3 shard resolved the DeepStack/merger ambiguity immediately.
5. **Prove information flow, not just tensor presence.** The Phase-4 test showed that reference-image information progressively changes the shared prompt-text hidden states through the 50-layer LLM.
6. **On a 32 GB unified-memory Mac: parallel brains, serial GPU.** Parallel code/reasoning agents are useful; concurrent heavy model loads and renders are not.

## Measured highlights

### Known-good 4B + ClipProj baseline

At 640x352 / 56 frames / 8 steps:

- output length: 2.333 s at 24 fps;
- conditioning: approximately 18–20 s;
- sampling/decode: approximately 454–501 s depending on run;
- peak resident memory observed during bake-off: approximately 30.98 GB.

### 124-frame scaling point

At 640x352 / 124 frames / 8 steps:

- output length: 5.167 s;
- sampling: approximately 928 s;
- total: 16 min 54 s.

### EasyCache

On Native Q4 in the tested run:

- wall-time improvement: approximately 10%;
- sampling loop: 2/8 steps skipped, approximately 1.33x loop speedup;
- peak memory: approximately 32.02 GB.

Decision: useful experimentally, but too little memory margin to make it the default on this 32 GB machine.

## Investigation arc

### Phase 1 — GGMLOps

Production GGMLOps and plain-ops vision towers were bit-identical at every measured checkpoint. GGMLOps was exonerated.

### Phase 2 — QKV / mapping

- `missing=0`, `unexpected=0` on the real production model;
- DeepStack 8/16/24 -> 0/1/2 confirmed;
- identity `[Q,K,V]` ordering decisively beat alternative permutations.

### Phase 3 — official H3 oracle

The local H3 mmproj was compared with the SHA-256-verified official MiniMax-H3 encoder shard. Fifty representative tensors were bit-identical. The apparently large DeepStack LayerNorm gains were confirmed as intentional H3 weights.

### Phase 4 — full LLM propagation

Reference-present vs reference-ablated passes showed progressively increasing differences in the identical shared prompt-text suffix:

- layer 0 mean cosine: 0.998;
- layer 24 mean cosine: 0.967;
- layer 49 mean cosine: 0.962;
- relative L2 difference approached 1.0 by the deep layers.

This is direct evidence that image information genuinely propagates through the full Qwen3VL/LLM conditioning path.

## Current blocker

The next subsystem is H3 DiT/keyframe integration:

- final Qwen3VL conditioning -> DiT;
- `minimax_keyframes`;
- reference VAE latent anchor;
- masks/indexing;
- whether image-tagged conditioning materially changes DiT computation and the predicted latent.

## Next efficient test

Instrument the actual DiT path first. Then, if needed, run a minimal diagnostic micro-render:

- portrait 9:16;
- smallest valid H3-aligned resolution;
- 5 frames if `frames % 17 == 5` is confirmed;
- 2–4 steps;
- fixed seed and sampler;
- two materially different references plus an ablated control;
- one render at a time.

The goal is diagnosis, not visual quality.

## External acceleration lead

A community YouTube report described `h3.c` + `lightx2v` Turbo acceleration, step reduction from 12 to 8/4, and large runtime improvements. It also warned that an incorrect strength coefficient by a factor of 16 destroyed image quality.

Treat this as **UNVERIFIED external evidence** until reproduced locally. The reported setup appears to assume substantially more memory and storage than this M2 Max 32 GB workflow, so it should not be mixed into the correctness investigation yet.

## Suggested video structure

1. Why running H3 on 32 GB unified memory is hard.
2. The initial reference-image failure.
3. The wrong hypothesis: vision tower / GGUF.
4. Four experiments that progressively eliminated the upstream stack.
5. Why the 8B comparison was misleading.
6. How the official H3 shard changed the conclusion.
7. The current DiT/keyframe blocker.
8. The practical lesson: stage memory, use small diagnostic tests, and separate correctness from optimisation.
