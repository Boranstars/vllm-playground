# Qwen3 Week 2 figures — evidence and mapping

These figures describe the local **v3.1** short-request experiment. `[T]` means directly observed in the trace/SQLite export; `[S]` means a semantic interpretation supported by the installed source and model config. Neither figure is a duration chart. Grouping and arrow spacing are schematic.

## Local evidence used

- Primary: [week2-vllm-short-v3.1-analysis.txt](../../results/nsys/week2/week2-vllm-short-v3.1-analysis.txt). NVTX ranges: lines 8–53; CUDA API trace: lines 57–1233; CUDA launch/GPU execution correspondence: line 1237 onward.
- Targeted confirmation: [week2-vllm-short-v3.1.sqlite](../../results/nsys/week2/week2-vllm-short-v3.1.sqlite), opened read-only. Only relevant NVTX ranges, the Prefill request's graph/cache/attention calls, selected CorrIDs, and request-scoped `sem_wait` were queried. No old v2 trace was used.
- Experiment: [week2_vllm_nsys.py](../week2_vllm_nsys.py): `Qwen/Qwen3-0.6B`, `max_model_len=4096`, single fixed prompt, greedy `temperature=0`, `max_tokens=16`, `ignore_eos=True`, prefix caching disabled; one identical warmup before `start_profile()`, CUDA profiler configured, outer `week2_vllm_generate` NVTX, `stop_profile()` in `finally`; no forced eager execution or speculative decoding.
- Cached model config: `/home/boran/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/config.json`.
- Installed vLLM **0.31.0** sources under `.venv/lib/python3.12/site-packages/vllm/`:
  - `model_executor/models/qwen3.py`: lines 105–170 (packed QKV, Q/K head RMSNorm, RoPE, attention, O); lines 222–247 (residual/norm dataflow); lines 276–278 (packed module mapping).
  - `model_executor/models/qwen2.py`: lines 81–115 (`Qwen2MLP`, reused as `Qwen3MLP`: packed gate/up, `SiluAndMul`, down).
  - `model_executor/layers/linear.py`: `MergedColumnParallelLinear` (line 673 onward), `QKVParallelLinear` (line 998 onward).
  - `model_executor/layers/layernorm.py`: lines 70–94, residual-aware `fused_add_rms_norm`.
  - `v1/attention/backends/flash_attn.py`: eager-mode note at lines 1278–1285, FlashAttention path, cache scatter/update at lines 1517–1541.
  - `config/compilation.py`: lines 607–638, documented `FULL_AND_PIECEWISE` default: full Decode graph, piecewise Prefill graph.
  - `v1/cudagraph_dispatcher.py`: uniform Decode versus piecewise dispatch paths.

## Direct observations [T]

| Evidence | Observed fact |
|---|---|
| Parent NVTX | `week2_vllm_generate`, PID 77177, 124401.492–242828.642 µs. EngineCore ranges/CUDA API calls belong to PID 77323. The timeline Host lane denotes EngineCore. |
| Prefill NVTX | `execute_context_1(30)_generation_0(0)`, 126312.436–131524.121 µs. |
| Decode NVTX | 15 instances of `execute_context_0(0)_generation_1(1)` follow Prefill. The subsequent zero-token context is not counted as Decode. |
| Prefill structure | 29 `cudaGraphLaunch` calls and 28 eager `reshape_and_cache_flash` / `flash_fwd_splitkv` pairs in the Prefill Host range. |
| Initial graph, CorrID 458 | Four kernel execution rows: embedding/norm, a CUTLASS GEMM, two fused Triton kernels. |
| CorrIDs 464 / 507 | Direct `cudaLaunchKernel` calls to cache / attention, respectively; kernel `graphNodeId` is NULL. Next eager pair: CorrIDs 519 / 562. |
| Prefill graph, CorrID 513 | `graphId=8276`; 11 kernel execution rows, all with graph-node IDs 35545149341696–35545149341706. The next representative graph, CorrID 568, also has 11 kernel rows. |
| Decode graph, CorrID 2439 | `graphId=8984`; 337 kernel execution rows, with 337 distinct kernel `graphNodeId` values. Includes 84 GEMV-family kernels, 28 CUTLASS GEMMs, 28 cache kernels, 28 `flash_fwd_splitkv` plus 28 combine kernels, 28 activation kernels, and repeated norm/preprocess kernels. These are counts of kernel rows, not a count of every kind of CUDA graph node. |
| Next Decode graph, CorrID 2907 | Another replay with 337 kernel execution rows. |
| Host / GPU asynchrony | First Decode Host NVTX is 132165.334–133249.143 µs. Its `cudaGraphLaunch` is 132581.791–133224.076 µs; corresponding GPU kernels execute at 134170.634–139008.495 µs. Host `cudaEventSynchronize` occurs at 133642.607–134192.747 µs and does not delimit completion of all kernels in that graph. |
| CPU coordination | 47 `sem_wait` calls start inside the outer request's NVTX interval in OSRT records. This records CPU waiting, not GPU computation. |

Representative Prefill segment **CorrID 513**, in observed GPU execution order:

1. CUTLASS GEMM → `cublasLt::splitKreduce_kernel`.
2. `triton_per_fused_fused_add_rms_norm_0`.
3. CUTLASS GEMM.
4. `triton_poi_fused_mul_silu_slice_1`.
5. CUTLASS GEMM → `cublasLt::splitKreduce_kernel`.
6. `triton_per_fused_fused_add_rms_norm_2`.
7. CUTLASS GEMM → two `triton_red_fused_add_cat_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_*` kernels.

The diagrams shorten long demangled symbols and omit template arguments.

## Source-supported semantic mappings [S]

- Config: **28 layers**, hidden size **1024**, **16 Q heads**, **8 KV heads**, head dimension **128**, MLP intermediate size **3072**, SiLU activation, bf16 checkpoint. `16 × 128 = 2048` is the Q width; it is not equal to this model's hidden size. At TP=1, packed QKV width is `2048 + 1024 + 1024 = 4096`, O maps `2048 → 1024`, packed gate/up maps `1024 → 6144`, and down maps `3072 → 1024`.
- Q/K/V semantic projections are implemented by a packed `QKVParallelLinear`, then split. Q and K have per-head RMSNorm before RoPE. Gate/up are implemented by one merged projection followed by `SiluAndMul`.
- The order and model dataflow support mapping CorrID 513 to **O(i) → residual/post-attention norm → gate/up → SiLU×up → down → residual/input norm(i+1) → QKV/preprocess(i+1)**. The layer assignments are semantic mapping, not layer annotations recorded by the trace. Therefore **graph boundary ≠ Transformer layer boundary**.
- The Decode graph contains repeated packed-QKV/preprocess → KV-cache/FlashAttention → O → norm → MLP patterns. GEMV/GEMM roles are assigned from source order and packing; generic linear kernel names do not independently identify a weight matrix.
- FlashAttention-family kernel names plus the installed backend source support the cache-update/attention relationship. The observed Decode combine kernel is part of the attention implementation, not another Transformer attention layer.
- The 16-token result consists of the token sampled from the Prefill forward plus 15 subsequent single-token Decode forwards. `generation_1(1)` is vLLM's context label, not an ordinal Decode-step number.
- Residual addition may be carried and fused with the following RMSNorm; the last following norm is the model's final norm. A semantic block boundary need not allocate a separate residual-add kernel.

## Deliberately generalized / omitted

- No one-to-one assignment of every fused Triton instruction or CUDA graph node to an architectural operation. Kernel names containing many operator fragments are kept as a **preprocess path**, rather than asserting the exact fusion partition. A `gemm_relu` template substring is not treated as evidence of a model ReLU; the source specifies SiLU.
- `splitKreduce` is shown only where observed, not as a required stage of every linear operation. Diagram rows sometimes group several kernel launches; graph execution counts do not equal operation or layer counts.
- The figures do not imply every Prefill segment is identical: the initial embedding/QKV segment and terminal segment differ from the representative cross-layer interior segment.
- The Decode `×28` box denotes the repeated model pattern. Prefix/terminal work and the first/final norm differ; the full replay is not exactly 28 identical copies of a fixed kernel list.
- GPU kernel execution rows with graph-node IDs confirm graph membership. They do not provide a full CUDA graph dependency topology or a complete count of non-kernel graph nodes.
- Host and GPU widths are **not to scale**. The figures show submission/replay structure, not relative latency, utilization, or an event-by-event queueing model. Runtime waits are not asserted to fence the whole model graph. Parent `generate()` encompasses work beyond the model graph.
- No explanation of occasional long-kernel jitter is included. No new profiling, benchmark or inference experiment was run to make these figures.

## Outputs

- [Overview — draw.io](qwen3_prefill_decode_overview.drawio), [SVG](qwen3_prefill_decode_overview.svg), [PNG](qwen3_prefill_decode_overview.png).
- [Runtime timeline — draw.io](qwen3_prefill_decode_timeline.drawio), [SVG](qwen3_prefill_decode_timeline.svg), [PNG](qwen3_prefill_decode_timeline.png).

Both files use the same semantic color palette. Gray dashed containers denote CUDA Graph regions; red outlines denote graph-external eager kernels. All exports are generated with the local draw.io desktop CLI and retain editable diagram data.
