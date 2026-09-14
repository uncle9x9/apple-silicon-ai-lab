# MiniMax-H3 DiT QKV Layout Audit

Date: 2026-09-15  
Machine: MacBook Pro, M2 Max, 30-core GPU, 32 GB unified memory  
Status: **resolved — production DiT QKV is contiguous and matches ComfyUI semantics**

## Question

`antirez/h3.c` documents the released MiniMax-H3 DiT checkpoint as storing QKV rows interleaved per attention head, while ComfyUI consumes `qkv_proj(x)` as three contiguous blocks:

```text
[Q_all | K_all | V_all]
```

This looked like a possible correctness defect. The key caveat was that Comfy-oriented checkpoints may already be repacked before runtime.

The audit therefore asked only one question:

> What QKV row layout is actually present in the exact production DiT GGUF used by this workflow?

## Loader and conversion trace

For the production GGUF path:

- `UnetLoaderGGUF.load_unet()` calls `gguf_sd_loader(unet_path)`;
- the UNET/DiT loader applies no MiniMax-specific QKV permutation;
- the tensor is stored as `blocks.0.attn.qkv_proj.weight`, preserving the PyTorch key;
- `comfy.gguf.orig_shape.*` metadata indicates the generic ComfyUI-GGUF quantisation path;
- unlike the CLIP/LLM loader, there is no architecture-specific `llama_permute()` equivalent in this DiT path.

Therefore the GGUF runtime consumes whatever row order was already present before generic quantisation.

## Official oracle

A pristine official MiniMax-H3 `Ref2VA/transformer` block-0 QKV tensor was retrieved at tensor level rather than downloading the full model.

The official tensor is structurally consistent with the released raw layout described by h3.c: **per-head grouped/interleaved QKV**.

The production local checkpoint is a distinct pruned/trained checkpoint, not merely a row-permuted copy of the vanilla official tensor. Position-wise local-vs-official cosine was near noise level:

- mean cosine: `0.028`
- median cosine: `0.001`

This ruled out using direct row equality between the two independently trained checkpoints as the discriminator.

## Structural self-consistency test

The decisive test evaluated which candidate grouping exposes the learned same-head Q/K structure expected from jointly trained attention projections.

Two hypotheses were tested independently on each tensor:

1. **contiguous** — `[Q_all | K_all | V_all]`;
2. **grouped** — `[h0 Q,K,V][h1 Q,K,V]...`.

Aggregate statistics were intentionally not used as the discriminator because a row permutation preserves shape, dtype, mean, standard deviation, RMS and L2 norm.

| Tensor | Hypothesis | matched-minus-mismatched `|cosine|` gap | head-0 Q·Kᵀ diagonal/off-diagonal | Verdict |
|---|---|---:|---:|---|
| Official vanilla | contiguous | `0.00026` | `0.96` | wrong grouping |
| Official vanilla | grouped | `0.03116` | **`4.60`** | **correct grouping** |
| Local production GGUF | contiguous | `0.03098` | **`4.61`** | **correct grouping** |
| Local production GGUF | grouped | `0.00008` | `1.40` | wrong grouping |

The result is a clean mirror image:

- official raw checkpoint: grouped/per-head interleaved;
- production local checkpoint: contiguous `[Q_all | K_all | V_all]`.

## Conclusion

**Production DiT QKV layout = contiguous.**

ComfyUI's current three-way split is correct for the exact production file used in these experiments. The h3.c/ComfyUI difference is therefore a checkpoint-contract difference, not an implementation bug.

The production checkpoint had already been regrouped before generic GGUF quantisation. No QKV patch should be applied.

## Reusable lesson

> Same model family, tensor name, shape, dtype and aggregate statistics do not guarantee the same physical row layout.

For MiniMax-H3, runtime semantics must be matched to checkpoint semantics:

```text
Raw upstream checkpoint
  grouped/per-head QKV
  -> h3.c consumes grouped layout directly

Comfy-oriented production checkpoint
  pre-regrouped contiguous Q|K|V
  -> generic GGUF quantisation preserves row order
  -> ComfyUI three-way split is correct
```

## Decision impact

This closes the final active code-defect hypothesis from the reference-adherence investigation.

The project now moves from **root-cause debugging** to **independent validation and task-semantics comparison**:

1. ComfyUI FL2VA baseline;
2. h3.c FL2VA oracle;
3. h3.c Ref2VA oracle;
4. M2 Max 32 GB memory/runtime comparison;
5. optimisation work only after correctness/task semantics are separated.
