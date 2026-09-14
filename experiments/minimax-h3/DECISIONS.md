# MiniMax-H3 Decision Log

## D001 — Keep 4B BF16 + ClipProj as the current default

**Status:** Accepted

Reason: it is the safest tested lane for reference-image work while the Native 32B downstream DiT/keyframe path remains under investigation.

## D002 — Retire Native 32B Q2

**Status:** Accepted

Reason: Q2 was slower on the tested MPS path because unsupported IQ2/IQ3 sub-blocks fell back to CPU/NumPy, while the same reference-adherence failure reproduced under Q4. No measured advantage remained.

## D003 — Keep Native 32B Q4 as experimental

**Status:** Accepted

Reason: Q4 is computationally viable under staged execution and is now proven correct through the entire Qwen3VL conditioning stack. Reference-driven use remains experimental until H3 DiT/keyframe integration is verified.

## D004 — Do not enable EasyCache by default on 32 GB

**Status:** Accepted

Reason: the tested run improved wall time by about 10%, but peak memory reached approximately 32.02 GB, leaving effectively no safety margin.

## D005 — Preserve hard Stage A / Stage B process isolation

**Status:** Accepted

Reason: the single-process workflow was killed under memory pressure when the encoder remained resident alongside the DiT. Process boundaries reliably released memory and enabled successful runs.

## D006 — Retire the blanket “32B banned” rule

**Status:** Accepted

Reason: the failure mode was simultaneous residency, not proof that Native 32B conditioning can never fit. The replacement policy is to permit validated memory-staged Native 32B execution while preventing encoder + DiT co-residency on constrained-memory Macs.

## D007 — Stop broad full-length bake-offs during root-cause work

**Status:** Accepted, refined

Reason: repeated multi-minute generations have low diagnostic value while the failing subsystem is unknown. Prefer instrumented numerical tests first, followed by tiny controlled renders only when they can discriminate a hypothesis.

## D008 — Retire the native vision-tower corruption hypothesis

**Status:** Accepted

Reason: the local GGUF/mmproj vision tower was compared against the SHA-256-verified official MiniMax-H3 encoder shard. Fifty representative tensors were bit-identical, including the previously suspicious DeepStack LayerNorm and merger weights. GGMLOps and plain-ops execution were also bit-identical.

## D009 — Exonerate the full Qwen3VL/50-layer LLM conditioning path

**Status:** Accepted

Reason: in a controlled reference-present vs reference-ablated test, the identical prompt-text hidden states diverged progressively with depth. Mean cosine fell from 0.998 at layer 0 to 0.962 at layer 49, while relative L2 difference approached 1.0. Image information therefore propagates coherently through the full LLM conditioning path.

## D010 — Move the blocker to H3 DiT/keyframe integration

**Status:** Accepted

Reason: all upstream reference-conditioning stages are now exonerated. The next investigation must trace final Qwen3VL conditioning, `minimax_keyframes`, reference VAE latent anchoring, masks/indexing, DiT conditioning consumption, and final predicted latent/noise.

## D011 — Treat 32 GB unified memory as a hard orchestration constraint

**Status:** Accepted

Reason: CPU, GPU and system processes share the same physical memory pool. Parallel analysis is useful, but concurrent heavy H3/Qwen model loads or renders can invalidate timing and trigger memory pressure. Policy: **parallel brains, serial GPU**.

## D012 — Use minimal diagnostic micro-renders after numerical tracing

**Status:** Accepted

Reason: once the DiT path is instrumented, the fastest end-to-end discriminator is a tiny fixed-seed A/B/C render using the smallest valid H3-aligned portrait 9:16 resolution, minimal valid frame count, and 2–4 steps. This is a diagnostic smoke test, not a quality benchmark.

## D013 — Keep external Turbo/lightx2v acceleration separate from correctness work

**Status:** Accepted

Reason: community reports of `h3.c` + `lightx2v` Turbo acceleration are currently unverified on this 32 GB machine and appear to assume materially different memory/storage resources. Reproduce them only after reference-conditioning correctness is established.
