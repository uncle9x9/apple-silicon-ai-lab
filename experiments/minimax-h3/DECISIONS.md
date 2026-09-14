# MiniMax-H3 Decision Log

## D001 — Keep 4B BF16 + ClipProj as the current default

**Status:** Accepted

Reason: it is the only tested lane that preserved the supplied reference subject beyond frame 0, even though quality still degraded.

## D002 — Retire Native 32B Q2

**Status:** Accepted

Reason: Q2 was slower on the tested MPS path because unsupported IQ2/IQ3 sub-blocks fell back to CPU/NumPy, while the same reference-conditioning failure reproduced under Q4. No measured advantage remained.

## D003 — Keep Native 32B Q4 as experimental

**Status:** Accepted

Reason: Q4 ran faster than Q2 and avoided the same CPU fallback behaviour, but its reference-image path is currently incorrect. Text-only use may remain useful; reference-driven use is blocked pending root-cause analysis.

## D004 — Do not enable EasyCache by default on 32 GB

**Status:** Accepted

Reason: the tested run improved wall time by about 10%, but peak memory reached approximately 32.02 GB, leaving effectively no safety margin.

## D005 — Preserve hard Stage A / Stage B process isolation

**Status:** Accepted

Reason: the single-process workflow was killed under memory pressure when the encoder remained resident alongside the DiT. Process boundaries reliably released memory and enabled successful runs.

## D006 — Retire the blanket “32B banned” rule

**Status:** Accepted

Reason: the failure mode was simultaneous residency, not proof that native 32B conditioning can never fit. The replacement policy is to permit validated memory-staged native 32B execution while preventing encoder + DiT co-residency on constrained-memory Macs.

## D007 — Stop broad video bake-offs until conditioning equivalence is proven

**Status:** Accepted

Reason: repeated 7–9 minute generations now have low information value. The next investigation should compare conditioning tensors and metadata across known-good and native paths to locate the first semantic divergence.
