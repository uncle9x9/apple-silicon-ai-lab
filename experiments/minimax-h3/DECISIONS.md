# MiniMax-H3 Decision Log

## D001 — Keep 4B BF16 + ClipProj as the current default

**Status:** Accepted

Reason: it remains the safest tested lane for reference-image work while the Native 32B DiT/task semantics are being validated.

## D002 — Retire Native 32B Q2

**Status:** Accepted

Reason: Q2 was slower on the tested MPS path because unsupported IQ2/IQ3 sub-blocks fell back to CPU/NumPy, while the same reference-adherence failure reproduced under Q4. No measured advantage remained.

## D003 — Keep Native 32B Q4 as experimental

**Status:** Accepted

Reason: Q4 is computationally viable under staged execution and is proven reference-sensitive through the Qwen3VL and tested DiT path. It remains experimental until the exact DiT checkpoint semantics and FL2VA/Ref2VA task choice are validated.

## D004 — Do not enable EasyCache by default on 32 GB

**Status:** Accepted

Reason: the tested run improved wall time by about 10%, but peak memory reached approximately 32.02 GB, leaving effectively no safety margin.

## D005 — Preserve hard Stage A / Stage B process isolation

**Status:** Accepted

Reason: the single-process workflow was killed under memory pressure when the encoder remained resident alongside the DiT. Process boundaries reliably released memory and enabled successful runs.

## D006 — Retire the blanket “32B banned” rule

**Status:** Accepted

Reason: the failure mode was simultaneous residency, not proof that Native 32B conditioning can never fit. Permit validated memory-staged execution while preventing encoder + DiT co-residency on constrained-memory Macs.

## D007 — Stop broad full-length bake-offs during root-cause work

**Status:** Accepted, refined

Reason: repeated multi-minute generations have low diagnostic value while the failing semantic contract is unknown. Prefer exact tensor or minimal numerical tests first, followed by tiny controlled renders only when they can discriminate a hypothesis.

## D008 — Retire the native vision-tower corruption hypothesis

**Status:** Accepted

Reason: the local GGUF/mmproj vision tower was compared against the SHA-256-verified official MiniMax-H3 encoder shard. Fifty representative tensors were bit-identical, including the previously suspicious DeepStack LayerNorm and merger weights. GGMLOps and plain-ops execution were also bit-identical.

## D009 — Exonerate the full Qwen3VL/50-layer LLM conditioning path

**Status:** Accepted

Reason: in a controlled reference-present vs reference-ablated test, the identical prompt-text hidden states diverged progressively with depth. Mean cosine fell from 0.998 at layer 0 to 0.962 at layer 49, while relative L2 difference approached 1.0. Image information therefore propagates coherently through the full LLM conditioning path.

## D010 — Move the investigation downstream to H3 DiT/keyframe integration

**Status:** Superseded by D014

Reason: this was the correct next step after Phase 4, but Phase 5 subsequently demonstrated that the tested DiT path is also materially reference-sensitive.

## D011 — Treat 32 GB unified memory as a hard orchestration constraint

**Status:** Accepted

Reason: CPU, GPU and system processes share the same physical memory pool. Parallel analysis is useful, but concurrent heavy H3/Qwen model loads or renders can invalidate timing and trigger memory pressure. Policy: **parallel brains, serial GPU**.

## D012 — Use minimal diagnostic micro-renders after numerical tracing

**Status:** Accepted

Reason: a 288x512 / 5-frame / 2-step controlled A/C run provided a high-information DiT discriminator at far lower cost than a full bake-off.

## D013 — Keep Turbo/lightx2v acceleration separate from correctness work

**Status:** Accepted

Reason: external Turbo claims remain optimisation evidence, not correctness evidence. Reproduce them only after checkpoint/task semantics are established.

## D014 — Exonerate silent reference-routing failure in the tested DiT path

**Status:** Accepted

Reason: a controlled real-reference vs ablated production Stage-B test measured final predicted-latent cosine `0.8372` and relative L2 difference `0.549`, with progressive divergence through DiT blocks and no NaN/Inf. Static tracing independently confirmed that conditioning, token tags and keyframe/video rows are threaded into the packed DiT sequence. The reference materially influences the tested DiT computation.

## D015 — Adopt antirez/h3.c as a required independent correctness oracle

**Status:** Accepted

Reason: h3.c is an independent Apple-Silicon implementation using the original MiniMax-H3 BF16 checkpoints, with FL2VA, Ref2VA and low-memory SSD-streaming support. It is now part of the validation plan, not merely an acceleration lead.

Pinned upstream revision for this investigation: `8974cc055ea9c02fcd14cc27dfda3e1027c05153`.

## D016 — Make H3 DiT QKV row layout the P0 gate before more full renders

**Status:** Accepted

Reason: h3.c documents the released MiniMax-H3 DiT checkpoint as storing QKV rows interleaved per attention head, whereas current ComfyUI MiniMax attention performs a conventional contiguous three-way split. This is a real semantic difference but not yet proof of a ComfyUI defect because the local GGUF converter may already reorder the tensor.

Required evidence: recover the actual production-loaded QKV row layout using converter tracing, row fingerprints, row-wise cosine matching and exact permutation recovery. Aggregate norms are insufficient because pure row permutations preserve them.

## D017 — Keep FL2VA and Ref2VA as separate correctness questions

**Status:** Accepted

Reason: first/last-frame anchoring and subject/reference conditioning are distinct MiniMax-H3 task families. Identity drift under FL2VA does not prove Ref2VA is broken. After QKV layout is resolved, compare ComfyUI FL2VA, h3.c FL2VA and h3.c Ref2VA separately.

## D018 — Do not patch production code until the QKV contract is proven

**Status:** Accepted

Reason: if the local GGUF is already reordered to contiguous `[Q_all | K_all | V_all]`, ComfyUI's current split is correct for that file. If the GGUF still preserves raw per-head interleaving, then and only then is there a concrete DiT correctness defect to reproduce on a single block before patching.
