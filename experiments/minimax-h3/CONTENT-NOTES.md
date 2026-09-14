# MiniMax-H3 M2 Max 32 GB — Content Notes

Purpose: factual source material for a future blog post, tutorial or YouTube video. Keep measured facts separate from interpretation.

## One-line story

A MiniMax-H3 identity-drift problem on a 32 GB M2 Max first looked like a broken vision/conditioning path, but five controlled phases exonerated reference routing through the DiT. The next high-value question is lower-level: whether the converted DiT QKV tensor preserves the checkpoint row semantics expected by the runtime.

## What was difficult

- MiniMax-H3 is large relative to a 32 GB unified-memory machine.
- Encoder + DiT simultaneous residency can exceed the practical memory envelope.
- Cross-model activation magnitudes can look suspicious even when both models are correct.
- A tensor can load with the correct shape while still being semantically misinterpreted.
- Long video generations are expensive and can have low diagnostic value.
- FL2VA first-frame anchoring and Ref2VA subject-reference conditioning are different tasks and must not be judged as if they were equivalent.

## Reusable engineering lessons

1. **Use process lifetime as a memory-management tool.** Stage A / Stage B isolation was more reliable than assuming CPU offload frees physical memory on Apple Silicon.
2. **Prefer discriminating tests over repeated renders.** Tensor-level and tiny A/C tests eliminated several hypotheses without paying the cost of a full generation.
3. **Do not compare activation magnitudes across different model variants as if they were a correctness oracle.** The earlier 8B comparison falsely made official H3 behaviour look abnormal.
4. **Verify against the exact official checkpoint whenever possible.** The official MiniMax-H3 shard resolved the DeepStack/merger ambiguity immediately.
5. **Prove information flow, not just tensor presence.** Reference-present vs ablated tests showed that image information changes both the full LLM conditioning and the final DiT-predicted latent.
6. **On a 32 GB unified-memory Mac: parallel brains, serial GPU.** Parallel code/reasoning agents are useful; concurrent heavy model loads and renders are not.
7. **A pure tensor permutation can preserve mean/std/L2 while destroying attention semantics.** When layout is in question, recover exact row permutations rather than trusting aggregate statistics.
8. **Use independent implementations as semantic oracles.** `antirez/h3.c` is valuable because it consumes the original BF16 checkpoint through a different Apple-Silicon runtime.

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

### Phase 5 micro-test

At 288x512 / 5 frames / 2 steps:

- real-reference Stage B: approximately 96 s total;
- final predicted-latent cosine, real reference vs ablated: `0.8372`;
- relative L2 difference: `0.549`;
- no NaN/Inf.

This is diagnostic evidence, not a quality benchmark.

## Investigation arc

### Phase 1 — GGMLOps

Production GGMLOps and plain-ops vision towers were bit-identical at every measured checkpoint. GGMLOps was exonerated.

### Phase 2 — Qwen3VL QKV / mapping

- `missing=0`, `unexpected=0` on the real production model;
- DeepStack 8/16/24 -> 0/1/2 confirmed;
- identity `[Q,K,V]` ordering decisively beat alternative permutations for the vision tower.

### Phase 3 — official H3 encoder oracle

The local H3 mmproj was compared with the SHA-256-verified official MiniMax-H3 encoder shard. Fifty representative tensors were bit-identical. The apparently large DeepStack LayerNorm gains were confirmed as intentional H3 weights.

### Phase 4 — full LLM propagation

Reference-present vs reference-ablated passes showed progressively increasing differences in the identical shared prompt-text suffix:

- layer 0 mean cosine: 0.998;
- layer 24 mean cosine: 0.967;
- layer 49 mean cosine: 0.962;
- relative L2 difference approached 1.0 by the deep layers.

This is direct evidence that image information genuinely propagates through the full Qwen3VL/LLM conditioning path.

### Phase 5 — actual H3 DiT

A real-reference vs ablated production DiT run showed substantial divergence through DiT blocks and in the final predicted latent.

Conclusion: the tested reference path is not silently dropped at the DiT boundary.

A scheduled-CLIP branch can drop extras such as `minimax_token_tags` when `use_clip_schedule=True`; this is a real conditional defect but was empirically absent from the exercised workflow.

## Why h3.c changed the investigation

Pinned upstream: `antirez/h3.c` @ `8974cc055ea9c02fcd14cc27dfda3e1027c05153`.

h3.c documents the released MiniMax-H3 **DiT** QKV tensor as per-head interleaved and consumes that grouped layout directly. Current ComfyUI MiniMax attention instead performs a conventional three-way Q/K/V split.

This does **not** yet prove ComfyUI is wrong: the local GGUF conversion may already reorder the raw checkpoint into the layout ComfyUI expects.

The current P0 test is therefore exact and cheap relative to a full render:

> Recover the real production-loaded DiT QKV row layout and converter lineage.

Use row hashes, row-wise cosine matching and exact permutation recovery. Do not use mean/std/RMS/L2 as the discriminator.

## FL2VA vs Ref2VA

This is now part of the story, not a footnote.

- **FL2VA:** first/last-frame anchor semantics.
- **Ref2VA:** ordered subject/reference media.

If both ComfyUI FL2VA and h3.c FL2VA drift but h3.c Ref2VA preserves identity, the earlier failure was a task-choice mismatch rather than a broken reference path.

## h3.c and 32 GB Apple Silicon

h3.c's `--ssd-streaming` mode uses the original BF16 checkpoint and trades speed for lower DiT residency by streaming blocks from SSD.

This independently supports the local finding that the key physical constraint is not only model size but **residency lifetime**.

For this machine:

- no concurrent H3/Qwen heavy runs;
- no concurrent renders;
- omit preview-heavy features during low-memory validation;
- measure memory pressure, swap and wall time around each heavy run.

## External YouTube evidence

One YouTube demonstration showed h3.c on an M3 Max with 36 GB unified memory using SSD streaming. It reported approximately 181–182 s for the H3 DiT portion of a short 20-step run.

Treat this as **external/unverified benchmark evidence** until the exact resolution, revision and comparable settings are reconstructed.

A separate community h3.c + lightx2v/Turbo acceleration path remains a later optimisation topic, not part of the correctness proof.

## Suggested video structure

1. Why H3 on 32 GB unified memory is hard.
2. The initial identity-drift symptom.
3. The first wrong hypothesis: GGUF / vision corruption.
4. Phase 1–3: progressively proving the vision stack correct.
5. Phase 4: proving the image genuinely changes the 50-layer LLM.
6. Phase 5: proving the reference also changes the actual DiT output.
7. The surprise: an independent h3.c implementation exposes a deeper checkpoint-layout question.
8. Why shape/statistics are not enough when QKV rows may be permuted.
9. FL2VA vs Ref2VA: are we asking the model to solve the right task?
10. The Apple-Silicon lesson: stage memory, use tiny discriminating tests, and separate correctness from optimisation.
