# NVFP4 Tensor-Core Microbenchmarks (Phase 0) — Measured

Executed 2026-11-04 on the RTX 5090 box after the GPU was released (priority flag
absent, no compute processes). All numbers below are measured on this machine;
anything not measured is marked `[UNVERIFIED]`.

**Verdict:** the FP4 GEMM primitive is a clear go — **1015–1094 TFLOP/s sustained**,
**4.75–5.13×** the bf16 GEMM on identical shapes, correct in forward, dgrad and
wgrad. An **on-the-fly quantized training prototype is a no-go as measured**:
the naive quantizer costs **15–34× the GEMM time**, turning the 5× kernel win into a
**0.14–0.30× end-to-end loss** versus plain bf16. The blocker is quantization, not
the tensor cores.

## 1. Execution summary

| item | value |
| --- | --- |
| command | `ssh user@<other-host> 'cd /home/user/runs/nvfp4_bench && CASES="…" OUT=…/results/final2 bash run_bench.sh'` |
| results | `/home/user/runs/nvfp4_bench/results/final2/` (`summary.txt`, per-case `*.json`, `*.stderr`) |
| GPU wall time | ≈ 142 s total across 8 GPU-touching invocations (including two discarded first-draft matrices and the probe scripts); the final 12-case matrix alone 59 s |
| GPU state | idle before/after (0 % util, no compute apps); `gpu_bench.flag` created and removed; `gpu_priority.flag` never created or touched |
| trainer | not run; nothing installed; production code untouched |

Case counts: **9 ok, 2 fail (padding probes), 1 rejected (constraint probe)**.
Every successful case selected the same variant: `v2_two_level_swizzled`
(`F.scaled_mm`, `[BlockWise1x16, TensorWise]`, e4m3 block scales + FP32 global,
scales pre-blocked with `SWIZZLE_32_4_4`). The unswizzled and classic variants
were never needed — so the swizzled blocked layout is the supported path.

## 2. Environment

`torch 2.14.1+cu130`, Python 3.13.5, device capability `(12, 0)`, RTX 5090 32 GB,
driver 621.01 / CUDA UMD 13.5, power cap **540 W** (default 600 W).
Full inventory: `docs/nvfp4_env_audit.md`.

## 3. Methodology

- **Quantization**: two-level NVFP4 — per-16-element `float8_e4m3fn` block scales
  plus one FP32 global scale per operand; operands packed as
  `torch.float4_e2m1fn_x2`, mat1 row-major, mat2 column-major, both packed along
  the contraction axis. Scales converted to cuBLAS's 128×4 blocked layout.
- **Reference**: bf16 matmul of the *dequantized* operands (plus an fp32 matmul for
  error decomposition). Because operands and scales are random, a wrong scale
  layout cannot hide behind uniform scales — it would collapse `cos`.
- **Timing**: 0.8 s warmup, then best of 3 time-boxed windows (0.6 s or 50 iters
  each). Best-of-window is required because the card is bimodal (see §7); the
  spread is recorded per case in `fp4.ms_windows`.
- **Baseline**: `torch.matmul` in bf16 on the same shapes, same warmup/window
  protocol.

## 4. Measured results

`rows × contraction × cols`; `quant` = time to produce the FP4 operands from bf16
(with the naive PyTorch quantizer); `e2e` = `quant + gemm`.

| case | GEMM shape | status | cos | rel_L2 | max_abs | nonfinite | fp4 ms | fp4 TFLOP/s | quant ms | fp4+quant ms | bf16 ms | bf16 TFLOP/s | speedup (gemm) | speedup (e2e) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fwd_3072` | 4352×3072×3072 | ok | 0.999995 | 3.249e-3 | 2.0 | 0 | 0.0762 | 1077.6 | 1.617 | 1.693 | 0.3804 | 216.0 | **4.99×** | 0.22× |
| `fwd_16384` | 4352×3072×16384 | ok | 0.999995 | 3.302e-3 | 2.0 | 0 | 0.4172 | 1050.1 | 6.392 | 6.809 | 2.0467 | 214.0 | **4.91×** | 0.30× |
| `fwd_8704` | 8704×3072×3072 | ok | 0.999994 | 3.339e-3 | 2.0 | 0 | 0.1619 | 1014.7 | 2.678 | 2.840 | 0.7693 | 213.6 | **4.75×** | 0.27× |
| `dgrad_3072` | 4352×3072×3072 | ok | 0.999995 | 3.249e-3 | 2.0 | 0 | 0.0751 | 1094.4 | 1.642 | 1.717 | 0.3849 | 213.4 | **5.13×** | 0.22× |
| `dgrad_16384` | 4352×3072×16384 | ok | 0.999995 | 3.302e-3 | 2.0 | 0 | 0.4077 | 1074.5 | 6.518 | 6.926 | 2.0133 | 217.6 | **4.94×** | 0.29× |
| `wgrad_3072` | 3072×4352×3072 | ok | 0.999995 | 3.251e-3 | 2.0 | 0 | 0.0759 | 1082.2 | 2.569 | 2.645 | 0.3745 | 219.3 | **4.93×** | 0.14× |
| `wgrad_16384` | 3072×4352×16384 | ok | 0.999995 | 3.304e-3 | 2.0 | 0 | 0.4138 | 1058.6 | 8.843 | 9.257 | 2.0418 | 214.6 | **4.93×** | 0.22× |
| `fwd_m4353` | 4353×3072×3072 | ok | 0.999995 | 3.249e-3 | 2.0 | 0 | 0.0766 | 1071.9 | 1.705 | 1.782 | 0.3867 | 212.4 | 5.05× | 0.22× |
| `fwd_n3088` | 4352×3072×3088 | ok | 0.999995 | 3.249e-3 | 2.0 | 0 | 0.0786 | 1049.9 | 1.658 | 1.737 | 0.3966 | 208.2 | 5.04× | 0.23× |
| `fwd_k3080` | 4352×3080×3072 | **rejected** | — | — | — | — | — | — | — | — | — | — | — | — |
| `fwd_n3080` | 4352×3072×3080 | **fail** | — | — | — | — | — | — | — | — | — | — | — | — |
| `fwd_k3088` | 4352×3088×3072 | **fail** | — | — | — | — | — | — | — | — | — | — | — | — |

Kernel for every successful case (identical):

```
cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn_align32_o_vs16_bias_bf16_relu
```

i.e. a CUTLASS3 sm120 block-scaled `e4m3 × e2m1` tensor-op GEMM — real FP4 tensor
cores, one kernel family across forward/dgrad/wgrad.

## 5. Correctness

All seven training-shaped cases plus both padding-remedy cases:

- `cos` = 0.999995 (0.999994 for the 8704-row case), far above the 0.99 gate;
- `rel_L2` = 3.25e-3 … 3.34e-3 against the dequantized-bf16 reference, and
  **1.66e-3 against an fp32 matmul of the same operands** — i.e. the residual is
  the two-level NVFP4 quantization error (per-block E4M3 scales, E2M1 data), not a
  layout or kernel defect;
- `max_abs` = 2.0 on outputs with |max| 290–390 (≈0.5–0.7 %), `nonfinite` = 0.

Orientations matter: dgrad (`dy @ W`) and wgrad (`x^T @ dy`) both pass with the
same kernel and the same error profile as forward, so the blocked-scale path
supports all three training GEMM shapes without transposing data beyond the
operand packing already required.

Caveat on the metric: an earlier acceptance rule used
`torch.isfinite(out).float().mean() == 1.0`, which rounds to 0.99999994 at 13.4 M
elements and produced one **false negative** (`fwd_m4353`). The rule now counts
non-finite elements as an exact integer; the reported table uses the fixed check.

## 6. Padding and divisibility constraints (measured)

| dimension | rule | evidence |
| --- | --- | --- |
| **M** (rows) | **unconstrained**; scale rows pad to a multiple of 128 | 4353 works, 0.0766 ms vs 0.0762 ms for 4352 (+0.5 %); scale block rows 4353 → 4480 |
| **K** (contraction) | **must be a multiple of 32** | 3072 ✔, 3104 ✔, 3136 ✔, 3168 ✔, 3200 ✔ / 3088 ✘, 3120 ✘ (`3088 % 32 = 16`) |
| **N** (cols) | **must be a multiple of 16** | 3072 ✔, 3088 ✔ / 3080 ✘, 3096 ✘ |
| K packing | must be even (implied by the above) | quantizer refuses odd last dim |
| K % 16 ≠ 0 | refused before any kernel runs | `quantize: cols=3080 must be divisible by block=16` |

The K rule is not K % 64 and not "blocks divisible by 4" — those hypotheses are
falsified by 3168 (198 blocks, works) and 3088 (193 blocks, fails). The failing
value differs from the working one by `K % 32 = 16`, and the two probes 3104
(works) and 3120 (fails) bracket it on both sides.

Exact failures:

| case | path | error text |
| --- | --- | --- |
| `fwd_k3080` | quantizer | `ValueError: cols=3080 must be divisible by block=16` (case status `rejected`) |
| `fwd_k3088` | swizzled v2 | `RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmulAlgoGetHeuristic(...)` |
| `fwd_k3088` | classic | `RuntimeError: Expected trailing dimension of mat1 to be divisible by 16 but got mat1 shape: (4352x1544).` |
| `fwd_k3088` | unswizzled v2 | `ValueError: For Blockwise scaling scale_a should have 852992 elements, got: 839936` |
| `fwd_n3080` | swizzled v2 / classic | `ValueError` / `RuntimeError`: `mat2 shape (1536x3080) must be divisible by 16` |
| `fwd_n3080` | unswizzled v2 | `ValueError: For Blockwise scaling scale_b should have 614400 elements, got: 591360` |

Practical rule: pad K up to a multiple of 32 and N up to a multiple of 16; leave M
alone unless a 128-row tile boundary matters for occupancy. Padding is cheap
(`fwd_n3088` costs the same per FLOP as `fwd_3072`; `fwd_m4353` is +0.5 %).

## 7. Anomalies observed

1. **Bimodal clocks.** Idle SM clock is 195 MHz; under load the card ramps to
   2422–2437 MHz, pins at the **540 W cap**, 93 % util, 63 °C. A cold or
   pre-ramp measurement lands ~1.6× slower (0.107 ms vs 0.068 ms on an 82 GFLOP
   GEMM). First drafts of the matrix caught the slow state for individual cases
   (e.g. `dgrad_3072` 0.068 ms in one run, 0.107 ms in another).
   Fixed by the 0.8 s warmup + best-of-3 windows now in `bench_nvfp4.timeit`.
2. **Transient above sustained.** The first case of a cold process can measure
   ~0.0685 ms (~1200 TFLOP/s) — a pre-power-limit burst. The sustained,
   warmed numbers in §4 (1015–1094 TFLOP/s) are the ones to plan with; the
   1200 figure is not reproducible under sustained load at 540 W.
3. **Metric false negative** (see §5): float32 `mean()` over 13.4 M elements
   rounded below 1.0; replaced with an exact non-finite count.
4. **Probe timing pollution.** A first probe called `to_blocked()` and built the
   global-scale tensors *inside* the timed lambda, reporting ~420 TFLOP/s. Those
   numbers were discarded; the reported runs pre-hoist everything a real loop
   would prepare once (weights) and time only the GEMM.
5. **Quantizer cost scales with non-contiguity.** `wgrad` quantizes `x.T`
   (non-contiguous): 2.57 ms for 3072×4352 vs 1.62 ms for the contiguous
   4352×3072 case of the same element count.
6. **Kernel name contains `relu`.** `…_bias_bf16_relu` is the CUTLASS epilogue
   template family, not an applied activation: a direct check shows the output
   preserves signs exactly (`min` = −98.0, matching the reference's −98.0;
   49.95 % negative outputs vs 49.94 % in the reference; `cos` 0.999996).

## 8. Go / no-go against the pre-registered criteria

Pre-registered in the previous revision of this document, before measuring:

| # | criterion | measured | verdict |
| --- | --- | --- | --- |
| 1 | forward, dgrad, wgrad all pass `cos ≥ 0.99`, `rel_L2 ≤ 0.10`, finite | cos 0.999995, rel_L2 3.25–3.30e-3, 0 non-finite, all three orientations | **PASS** |
| 2 | pure-GEMM speedup ≥ 3× on the two 16384-wide cases | 4.91× (fwd), 4.94× (dgrad), 4.93× (wgrad) | **PASS** |
| 3 | `fp4 + quantize` ≥ 1.5× faster than bf16 | 0.30×, 0.29×, 0.22× | **FAIL** |
| 4 | non-multiple shapes have a documented, cheap padding rule | K % 32, N % 16, M free; padding cost ≤ 0.5 % | **PASS** |

**Decision: GO for the FP4 GEMM primitive; NO-GO for a custom NVFP4 autograd
prototype in its current form.** Criterion 3 fails by 5–7×, and the failure is
entirely attributable to the quantization implementation (1.6–8.8 ms per step
versus 0.08–0.42 ms of GEMM), not to the tensor cores.

What that means concretely:

- A prototype that quantizes activations per step with this PyTorch quantizer
  cannot beat bf16, regardless of how good the autograd plumbing is.
- Pre-quantizing the base weights (one-off, offline) removes only part of the
  cost: forward still needs the activation (`x` / `dy`) quantized per step, which
  is the same 1.6–2.6 ms measured here.
- Backward remains bf16 in any case: torch exposes no FP4 backward (`_scaled_mm`
  output has `requires_grad=False`, and fp4×bf16 hard-fails), so an autograd
  prototype must dequantize or use a straight-through estimator.

## 9. Required next step before any autograd work

A fused `bf16 → (packed FP4, e4m3 block scales, global scale, blocked layout)`
CUDA/Triton kernel, benchmarked against the same reference. Target: quantization
within ~1× of the GEMM it feeds (i.e. ≲0.4 ms for the 16384-wide cases), which
would put e2e at ≥2.5× over bf16. Whether a fused kernel can reach that on this
card is `[UNVERIFIED]` — the measured 1.6–8.8 ms is a pure-PyTorch elementwise
pipeline and is not evidence about a fused implementation's ceiling.

If that spike succeeds, the prototype shape is: pre-quantized FP4 weights,
per-step fused activation quantization, FP4 forward/dgrad/wgrad, bf16 backward
for the LoRA adapters.

## 10. Unverified

- Throughput of a *fused* quantization kernel on this card — `[UNVERIFIED]`.
- Whether the `K % 32` / `N % 16` rules hold for every cuBLASLt algorithm choice
  and every future driver; they are measured on this driver/GPU only
  (RTX 5090, CUDA UMD 13.5, cuBLAS 13.1.1.3) — `[UNVERIFIED]` beyond that.
- Whether FP4 dgrad/wgrad kernels would be numerically acceptable inside a real
  LoRA training loop (per-step error accumulation, ~3.3e-3 per GEMM here) —
  `[UNVERIFIED]`; no training was run.
- Numbers on a card without the 540 W power cap (the sustained rate here is
  power-limited, not thermally limited at 63 °C) — `[UNVERIFIED]`.

## Appendix: reproduction

```bash
# full matrix, one file per case, ~60 s of GPU time
ssh user@<other-host> 'cd /home/user/runs/nvfp4_bench && \
  CASES="fwd_3072 fwd_16384 fwd_8704 dgrad_3072 dgrad_16384 wgrad_3072 wgrad_16384 \
         fwd_m4353 fwd_k3080 fwd_n3080 fwd_n3088 fwd_k3088" \
  OUT=/home/user/runs/nvfp4_bench/results/final2 bash run_bench.sh'

# divisibility rules at real sizes (~2 s)
ssh user@<other-host> 'cd /home/user/runs/nvfp4_bench && NVFP4_GPU_OK=1 \
  /home/user/MOSS-Lora-Trainer/venv/bin/python probe_extra.py --rules-only'

# CPU-only checks (no GPU)
ssh user@<other-host> 'cd /home/user/runs/nvfp4_bench && \
  /home/user/MOSS-Lora-Trainer/venv/bin/python test_quant_cpu.py && \
  /home/user/MOSS-Lora-Trainer/venv/bin/python validate_shapes_cpu.py'
```

| file (in `/home/user/runs/nvfp4_bench/`) | role |
| --- | --- |
| `nvfp4_quant.py` | quantization, blocked-scale layout, dequant reference |
| `bench_nvfp4.py` | per-case benchmark, guards, variant selection, timing |
| `run_bench.sh` | coordination runner (priority flag, per-case timeout) |
| `summarize.py` | JSON → markdown table |
| `probe_extra.py` | divisibility rules, timing repeats, M-row cost |
| `load_gemm.py` | sustained-load driver used for the clock sampling |
| `test_quant_cpu.py`, `validate_shapes_cpu.py` | CPU-only verification |
| `results/final2/` | the measured artifacts quoted above |
