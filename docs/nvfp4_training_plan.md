# NVFP4 training for MOSS-Image LoRA: feasibility and integration plan

**Status:** design report, no training-code changes made.
**Scope:** single RTX 5090 32 GB (sm_120), WSL2, torch 2.14.1+cu130; current trainer in this repo; reference trainer `KohakuBlueleaf/KohakuMossTrainer` at commit `b06ac60` (cloned to `/tmp/KohakuMossTrainer`).
**Measurement caveat:** the workstation used for this report has `torch 2.14.1+cpu`, no CUDA device, and no local model checkpoint. No GPU measurement was possible. Memory numbers below are analytical, calibrated against the supplied 18.13 GB rank-96/batch-4 measurement and against `docs/RESEARCH.md`. Items marked **[INFERENCE]** are estimates, not observations.

---

## Executive summary

1. **The reference does not implement NVFP4 training.** It implements frozen-base **int8**, **FP8 E4M3**, and **MXFP8** training, plus a **W4A8 int4/int6 ComfyUI storage format that decodes to int8**. `nvfp4` appears in the reference only as an *unsupported* ComfyUI quant record that the loader explicitly refuses (`tests/quant/test_comfy_refusal.py:22,51,92`). There is no FP4 tensor, scale, kernel, recipe, or training flag anywhere in the tree.
2. **Native NVFP4 tensor-core training is not supported on sm_120 by the current training stack.** TransformerEngine documents NVFP4 training for **SM 10.0 / SM 10.3 only** (`nvfp4.rst`, “Supported devices”); on RTX 50 the default recipe fails because `cvt.rs.satfinite.e2m1x4.f32` is absent, the RHT/SR workarounds are required, and the backward weight-gradient GEMM returns `CUBLAS_STATUS_NOT_SUPPORTED` (`NVIDIA/TransformerEngine#2255`). PyTorch 2.14.1 has the NVFP4 `_scaled_mm` plumbing, but that does not create a supported sm_120 training path.
3. **On our trainer, NVFP4 can reduce the frozen base only.** At rank 96, batch 4, the measured 18.13 GB decomposes to ~9.47 GiB base + ~3.32 GiB LoRA/AdamW + ~4.18 GiB checkpointed block inputs + ~1.06 GiB overhead. NVFP4 base storage would replace 9.47 GiB with ~2.66 GiB (save ~6.80 GiB), but leaves LoRA state, activations, VAE, and sampling untouched.
4. **Batch 8 is probably not failing for capacity reasons.** The same model predicts rank-96/batch-8 at ~22.2 GiB, comfortably inside 31.3 GiB usable. If native batch 8 really raises `CUDA driver error: unknown error` rather than `CUDA out of memory`, NVFP4 will not fix it; the rank-64 crash and batch-8 crash should be diagnosed first as a kernel/allocator bug.
5. **Recommendation: prototype first, but prototype the supported path (FP8 weight-only base), not NVFP4 tensor cores.** Native NVFP4 training on sm_120 is a **no-go today**. A bounded 1–2 day FP8 storage prototype plus a 1–2 hour NVFP4 microbenchmark is worth running; a full NVFP4 integration is not.

---

## 1. What KohakuMossTrainer actually implements

### 1.1 There is no NVFP4 implementation

Evidence from the actual tree:

```python
# src/kohakumoss/quant/comfy_format.py:23-24
RECORD = "comfy_quant"
FORMATS = {"int8_tensorwise": "int8", "float8_e4m3fn": "fp8", "asym_w4a8_int8": "w4a8"}
```

Any other record is rejected. The test suite uses `nvfp4` as the canonical *unknown* format:

```python
# tests/quant/test_comfy_refusal.py:22
UNKNOWN = {"format": "nvfp4", "group_size": 16}

# tests/quant/test_comfy_refusal.py:47-51
def check_refused(state: dict) -> None:
    with pytest.raises(ValueError, match="blk.lin.comfy_quant.*nvfp4"):
        split_quantized(state)
```

and again at the file-loader level:

```python
# tests/quant/test_comfy_refusal.py:92
assert "ComfyUI quant format 'nvfp4'" in str(err), err
```

The full-history search (`git log --all`, 170 commits) contains no NVFP4 commit; `git ls-remote` shows only `main`. The string `nvfp4` occurs in exactly three files: the two refusal tests and the ComfyUI format table test. There is no NVFP4 code to port.

### 1.2 What it does implement instead

Three families (`docs/internals/quant-kernels.md:1-27`):

| value | family | arch | stored bytes / weight | trainable base |
|---|---|---|---|---|
| `compute` | none (16-bit) | every | 2 | yes |
| `int8` | weight-only | every (eager on CPU) | 1 + per-row fp32 scale | no |
| `fp8` | weight-only | sm89+ | 1 + per-row fp32 scale | no |
| `quant` | fp8 where available, else int8 | every | 1 + per-row fp32 scale | no |
| `fp8_compute` | fp8 compute | sm89+ | ~1.0 (+ W^T copy for dgrad) | yes, via fp8 copies |
| `mxfp8` | fp8 compute, 1×32 e8m0 | sm89+ (native block MMA on sm100+) | ~1.03 | yes, via fp8 copies |

**Weight-only int8/FP8.** `QuantWeight` stores `q` and a per-output-row fp32 `scale`; the weight is `q * scale[:, None]` (`src/kohakumoss/quant/weights.py:29-44`). `int8` uses `round(w / (row_amax/127))`; `fp8` uses `fp8(w / (row_amax/448))`. The linear dequantizes inside the GEMM and saves only `q`/`scale` for backward; the base is frozen, so there is no weight gradient (`docs/internals/quant-kernels.md:63-78`). This is the closest analogue to a “QLoRA-style” path and is the one that maps onto our trainer.

**FP8/MXFP8 compute.** `ComputeWeight` quantizes `q`, `scale`, an optional vendor-swizzled `aux`, and an optional `q_t` for the input-gradient GEMM (`src/kohakumoss/quant/compute.py:87-140`). `ComputeCache.refresh()` re-quantizes a *trainable* weight after every optimizer step; `fp8_linear_train` runs FP8 forward and FP8 input gradient with a 16-bit weight gradient (`compute.py:163-206`). This is full-finetune machinery, not needed for LoRA.

**W4A8 is not FP4.** The ComfyUI `asym_w4a8_int8` format stores 4- or 6-bit *integer* codes plus a 16-level codebook and scales, then decodes onto an **int8 grid** before an int8 GEMM (`src/kohakumoss/quant/w4a8.py:1-8`, `decode_codes`). It never creates an E2M1 value or uses FP4 tensor cores. It is an inference/storage format, not a training format.

### 1.3 Dependencies and hardware gating

- `pyproject.toml:13-30` depends on `torch>=2.13,<2.15`, `numpy`, `lycoris-lora>=4.1.3`, `diffusers>=0.38`, `transformers>=5.3`. There is **no** `transformer_engine`, `torchao`, CUTLASS, or custom CUDA extension dependency.
- Kernels are Triton/TileLang/`torch.compile`/eager (`README.md:154-161`). On sm_120 the tested set is fp8 and MXFP8; sm_100 is “supported, not tested”; there is no FP4 row.
- The quantization code is pure PyTorch plus Triton kernels for weight-only int8/fp8 and fp8 compute.

### 1.4 How it integrates with MOSS-Image

It is a **separate implementation**, not a fork of this repo: package `kohakumoss`, own config engine (`kogine`), own model builder (`MossLinear` subclasses `nn.Linear`), own dataset/cache, checkpoint manager, sampler, and export. `grep -rn "moss_lora"` over the reference returns nothing. It reads both the HF MOSS-Image-Base layout (`configs/moss/weights/weights_hf.py`) and ComfyUI single-file layouts (`weights_comfy.py`); LoRA/LoHa/LoKr etc. are built through `lycoris-lora`; exports are ComfyUI/LyCORIS adapter files (`docs/guides/export-to-comfy.md`). A merged export from an int8 base is explicitly “the dequantized base plus dW in bf16” (`export-to-comfy.md`).

Its VRAM management, not NVFP4, is where the large wins are:

- Named memory profiles (`src/kohakumoss/memory/profiles.py:40-75`): `balanced` = block checkpointing + **16 resident blocks, the rest streamed**; `low` = 4 resident, activation offload, streamed optimizer state; `minimal` = 2 resident, activation/gradient offload, host states.
- `CHECKPOINT="partial:0.2"` checkpoints only a fraction of blocks; `checkpoint_mode()` parses it (`profiles.py:77-80`), used by `recipes_lora_5090.py:20`.
- Measured with latent+text cache, LoRA r32 at 1 MP: **6.63 GiB** step peak and 1.14 samples/s on the RTX 5090 (`README.md` release table). The speed recipe (`MAX_TOKENS=4608`, `GRAD_ACC=64`, `partial:0.2`) measures **40.26 s/step for 294,912 tokens, 6545 tok/s, 30.0 GiB** (`docs/performance/training-5090.md`, “LoRA r32”). Those are different memory/time tradeoffs: the 6.63 GiB row uses the `balanced` profile (16 resident blocks, the rest streamed); the 30 GiB recipe uses a manual profile, which **[INFERENCE]** keeps more of the base resident for speed.
- No NVFP4 number exists in any table.

**Takeaway:** the reference's relevant contribution to our problem is quantized *frozen-base storage* (int8/FP8) plus block streaming/partial checkpointing, not FP4.

---

## 2. Applicability to our trainer and RTX 5090 / sm_120

### 2.1 What is directly applicable

- **Concept:** store the frozen 5.08 B-parameter DiT base below 16 bits, dequantize per layer inside the forward/backward, keep LoRA A/B in fp32 for export. This is exactly `QuantWeight` semantics and is hardware-agnostic on sm_120 for int8/FP8.
- **FP8 is tested on sm_120** by the reference (`README.md:154-161`) and needs only native `float8_e4m3fn` casts, which torch 2.14.1 supports.
- **Partial block checkpointing** is applicable independently of quantization and would cut our 4.18 GiB checkpoint-input cost at batch 4; it does not require new hardware.

### 2.2 NVFP4-specific blockers on RTX 5090 / sm_120

NVFP4 is E2M1 values (±6 max) with one E4M3 scale per 16-element block and one FP32 global scale; training recipes additionally use stochastic rounding (SR) for gradients and a random Hadamard transform (RHT) for the wgrad GEMM (`nvfp4.rst`, “Data Format”, “Stochastic Rounding”, “Random Hadamard Transform”). It is not just “int4 storage”; the training recipe depends on these pieces.

1. **TransformerEngine gates NVFP4 training to SM 10.0/10.3.** Its own “Supported devices” section says:
   - Training: **SM 10.0, SM 10.3**
   - Inference: **SM 10.0+**
   (`transformer_engine/docs/features/low_precision_training/nvfp4/nvfp4.rst`, “Supported devices”). SM 12.0 is absent from training.
2. **The default recipe cannot run on sm_120 as-is.** `cvt.rs.satfinite.e2m1x4.f32` (stochastic FP4 conversion) is not supported on `sm_120a`; the RTX 50 thread in `NVIDIA/TransformerEngine#2255` shows the initial `FP4 cvt PTX instructions are architecture-specific` failure. The accepted workaround is `NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)`, which changes the recipe (no SR, no RHT) and is exactly the part that protects training quality.
3. **Backward is not a supported path.** In the same issue, a user reports forward success after the workaround but `loss.backward()` fails in the wgrad GEMM with `CUBLAS_STATUS_NOT_SUPPORTED: Unable to find suitable cuBLAS GEMM algorithm` on sm_120. For LoRA the base **wgrad is not needed**, but the base **dgrad** (input gradient) is still required; no report demonstrates that path at production shapes. A later comment describes the TMEM/UMMA absence on sm_120 as “not patchable” via compile flags and measures the recipe workaround at 1.77× bf16 for a single GEMM (M=1024, K=N=4096), not end-to-end training.
4. **PyTorch has plumbing but not a proven sm_120 training kernel.** torch 2.14.1's meta function understands the NVFP4 recipe (`_meta_registrations.py:7203-7215`: block size 16, E4M3 scales, swizzled `SWIZZLE_32_4_4`), and `aten/src/ATen/native/cuda/ScaledBlas.cpp:1187-1230` dispatches `_scaled_nvfp4_nvfp4` to `_scaled_gemm` (cuBLASLt) with no SM100-only check (`ScaledBlasDeviceUtils.cpp`: `major >= 9`). But the actual cuBLASLt algorithm must exist for sm_120; the TE backward failure is direct evidence that it does not for at least one GEMM orientation. cuBLAS's block-scaled FP4 section requires **TN layout**, `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, and documents the 128-element block-scaling extension for **10.x and 11.0**; the surrounding TN-layout note explicitly includes 12.x for FP8, while the FP4 section does not (`cuBLAS docs §3.1.4.4`). **[INFERENCE]** This is consistent with FP4 block-scaled GEMM not being a supported 12.x path, but it is a documentation inference, not a runtime test. **[INFERENCE]** The consumer-Blackwell FP4 GEMM is a library-support gap, not a silicon gap: PTX does define `mma.sync.aligned.m16n8k64...kind::block_scale...e2m1.e2m1` with `kind::mxf4nvf4`, and `cvt.rn.satfinite.e2m1x2.f32` is listed for `sm_120a`; only the stochastic `cvt.rs` form is missing.
5. **`torch._scaled_mm` cannot mix bf16 activations with FP4 weights.** Block-scaled FP4 requires both operands FP4 and TN layout. A LoRA base layer would therefore need a quantization kernel for every activation (`x` and `dy`), not just a weight-only storage change. The reference's FP8 weight-only path avoids this entirely by dequantizing the weight into a 16-bit GEMM.
6. **torchao's NVFP4 training linear is a prototype with hard shape constraints.** `torchao/prototype/moe_training/nvfp4_training/nvfp4_linear.py` uses `F.scaled_mm` with NVFP4 scales and requires `M % 128 == 0`, `K % 128 == 0`, `N % 128 == 0`. Our per-sample joint sequence is 4096 image tokens + a variable caption length (often 4.3k–4.6k total); it would need padding to 128 and a masked path. It is not installed in our venv.
7. **Our model is not a TE model.** The vendored `_Block` and `MOSSImageTransformer` use `nn.Linear` and read `.weight.dtype` (`moss_lora/model/moss_image.py:166-183`, `:287-301`). TE's `fp8_autocast`/NVFP4 recipe operates on `te.Linear` modules; adopting it means replacing the module class across the 42 blocks (and preserving our state-dict keys, `LoRALinear` wrapper, `torch.utils.checkpoint` calls, and the per-sample loop). That is a rewrite of the model plumbing, not a patch.
8. **Our environment has no TE/torchao.** `.venv` contains torch 2.14.1+cpu, no `transformer_engine`, no `torchao`, no Triton. On the target cu130 venv, adding TE requires a source build with `NVTE_CUDA_ARCHS=120a` and CUDA 13; even then the training path is documented as SM10.0/10.3 only.
9. **Export/deployment mismatch is inherent.** Our export is adapter-only (`moss_lora/export_comfy.py`, `COMFY_OUTPUT_NAME`), and `merge.py` loads a bf16 base and merges in bf16 (`moss_lora/merge.py:36-43`). If the adapter is trained against an NVFP4 base, ComfyUI will apply it to a bf16 (or FP8/int8) base. The adapter's correction includes the quantization residual `W - W_q`, which disappears at deployment. FP8's residual is small; NVFP4's is much larger. This is the single biggest quality risk and is independent of kernel support.

### 2.3 Verdict table

| Component | sm_120 support | Evidence |
|---|---|---|
| FP8 E4M3 weight-only / compute | **yes, tested** | reference `README.md:154-161`; `quant/weights.py`, `quant/compute.py` |
| int8 weight-only / compute | **yes** | same |
| MXFP8 block-scaled compute | **yes, tested** | `README.md:154-161`; `quant/compute.py:18-23` |
| W4A8 (int4 codes → int8 GEMM) | inference/storage only | `quant/w4a8.py:1-8` |
| NVFP4 storage only (bf16 compute) | technically possible with a custom unpack; no reference code | **not implemented** |
| NVFP4 tensor-core forward | partial via TE workaround (RTN, no RHT/SR); via `_scaled_mm` unverified | `#2255`; `ScaledBlas.cpp:1187` |
| NVFP4 tensor-core backward (dgrad) | unproven; wgrad known unsupported | `#2255` |
| NVFP4 full training recipe (SR+RHT) | **no** on sm_120 | TE docs SM10.0/10.3; `#2255` |

---

## 3. Ranked integration options

### Option 1 (recommended prototype): weight-only quantized base, bf16 compute

**Idea:** keep the DiT math in bf16, but store the frozen base as FP8 (first) or NVFP4 codes (second), dequantizing one layer at a time inside the forward/backward. This is `QuantWeight` semantics without the Triton kernel dependency and without FP4 tensor cores. It is the lowest-risk way to buy VRAM.

**Affected files/functions**

| File | Change |
|---|---|
| `moss_lora/quant_base.py` (new) | `QuantBaseLinear`: buffers `q` (`int8`/`float8_e4m3fn`/packed uint8 NVFP4), `scale`; `compute_dtype`; `forward(x)` dequantizes `w` then `F.linear(x, w)`; `nbytes()`; deterministic `quantize_weight(w, fmt, group)`. |
| `moss_lora/lora.py:40-78` | `LoRALinear.__init__` accepts a quantized base; `forward` uses `self.base(x)` and `self.base.compute_dtype`; `.weight` property must not dequantize on every block access (raise for quantized base, or return a cached tensor only for merge). |
| `moss_lora/model/moss_image.py:166-183` | `_Block.forward` currently uses `dtype = self.proj['q'].weight.dtype`; replace with `self.proj['q'].compute_dtype`. |
| `moss_lora/model/moss_image.py:287-301` | `video_embedder`, `final_norm_video`, `final_linear_video` read `.weight.dtype`; leave these three unquantized (they are <1.5% of parameters) or add `compute_dtype`. |
| `moss_lora/trainer.py:98-107` | after `from_pretrained` and before `inject_lora`, call `quantize_base(transformer, fmt)`; keep `requires_grad_(False)`; `train()` already freezes first. |
| `moss_lora/trainer.py` (`save_checkpoint`), `moss_lora/lora.py:adapter_config_payload` | record `base_precision`, `base_quant_group`, and a hash of the quantized codes in `adapter_config.json` for reproducible resume. |
| `moss_lora/config.py:ModelConfig` | add `base_precision: str = "bf16"` and `base_quant_group: int = 16`; strict validation; YAML plumbing. |
| `moss_lora/merge.py:36-43` | unchanged for bf16 base; add a warning that the adapter was trained on a quantized base; verification compares merged bf16 output to trainer output. |
| `moss_lora/sample.py:load_sampling_transformer` | add `--base-precision` to sample with the same quantized base (training parity) or bf16 (deployment parity). |

**Config/API**

```yaml
model:
  base_precision: bf16   # bf16 | fp8 | int8 | nvfp4
  base_quant_group: 16   # nvfp4 only
```

No change to `TrainConfig`, optimizer, sampler, or checkpoint format beyond the adapter metadata. `lora_state_dict` remains fp32; ComfyUI export keys are unchanged.

**Cost/risk:** ~1–3 days including tests. FP8 path uses a native `.to(bfloat16)` cast; NVFP4 path needs a packed-nibble unpack kernel (pure PyTorch bit ops are correct but slow; a Triton unpack is preferable). Memory win: ~4.73 GiB for FP8, ~6.80 GiB for NVFP4. No tensor-core speedup; small dequant overhead.

### Option 2: port the reference's `QuantWeight` + Triton weight-only kernels

**Idea:** vendor the proven `kohakumoss_kernels.quant` weight-only int8/FP8 GEMM (and optionally `fp8_compute`) into our tree, and make `LoRALinear.base` a `QuantWeight`-like module.

**Affected:** new `moss_lora/kernels/quant_linear.py` (Triton kernel + custom op), `lora.py`, `model/moss_image.py` dtype reads, `trainer.py` load, `config.py`, plus `triton` as a dependency. Export/merge unchanged.

**Pros:** dequantization fused into the GEMM, only `q`/`scale` saved for backward, tested on sm_120 by the reference. **Cons:** it is int8/FP8, not NVFP4; porting/tuning is more work than Option 1; adds a kernel maintenance surface. Use only if Option 1's per-layer dequant is too slow.

### Option 3: NVFP4 tensor-core compute (torchao/TE prototype)

**Idea:** a custom `torch.autograd.Function` for the frozen base: quantize `x` and `W` to NVFP4, `torch._scaled_mm` forward; quantize `dy` and `W^T` for dgrad; skip wgrad; add the existing bf16 LoRA branch. Pad the joint sequence to a multiple of 128.

**Affected:** new `moss_lora/nvfp4_linear.py`, `lora.py`, `model/moss_image.py:166-183`, `trainer.py` load/step, `config.py`, environment (`torchao` or TE), plus a capability gate and fallback. Requires `M,K,N % 128`, TN layout, swizzled scales, and both operands FP4.

**Pros:** the only option that can reduce compute time (theoretical ~1.7× end-to-end upper bound if base GEMMs are 2×; realistically 1.1–1.4× given attention and the LoRA branch). **Cons:** no supported sm_120 training path; dgrad unproven; SR/RHT must be disabled; quantization kernels add work; padding/masking complexity; quality risk. **Only after a microbenchmark proves forward+dgrad at our shapes.**

### Option 4: adopt the reference trainer or run it as a separate trainer

**Idea:** use `kohakumoss` end-to-end and convert our cache to its shard format.

**Assessment:** not a fork; it has its own dataset/cache format, sampler, checkpoint manager, export, and config engine. Replacing our trainer would discard our cache/variant sampler, resume semantics, ComfyUI adapter conversion, and validation pipeline. It would also not provide NVFP4. **Not recommended** for this feature; porting the quant modules (Option 2) is the only sensible reuse.

### Ranking

| Rank | Option | VRAM win | Speed | Effort | Risk | Verdict |
|---|---|---|---|---|---|---|
| 1 | O1 FP8 weight-only base | −4.73 GiB | ~neutral | 1–3 d | medium | **prototype** |
| 2 | O1 NVFP4 weight-only base | −6.80 GiB | ~neutral/slower | +1–2 d | medium-high | only if FP8 insufficient |
| 3 | O2 port reference quant kernels | −4.7 to −6.8 GiB | small gain | 1–2 wk | medium | if O1 dequant too slow |
| 4 | O3 NVFP4 tensor cores | −6.8 GiB + activation | 1.1–1.4× (if it works) | 2–4 wk | very high | microbenchmark only |
| 5 | O4 adopt reference trainer | −6.8 GiB (streaming) | proven | weeks | high | no |

---

## 4. Memory and speed analysis

### 4.1 Where the 18.13 GiB goes at rank 96, batch 4

The model has **5,082,027,008** parameters (42 layers × 113.25 M + text/embed/head), i.e. **9.466 GiB** in bf16. This matches `docs/RESEARCH.md:108` (“DiT bf16 10.16GB” decimal).

| Component | Formula / code | GiB at r96, B4 |
|---|---|---|
| Base weights, bf16 | `load_transformer` (`trainer.py:98-107`) loads all 42 blocks and `.to(device)`; frozen at `train()` | **9.466** |
| LoRA A/B, fp32 | `lora.py:57-58`: `A [r,in]`, `B [out,r]`; 2,322,432·r params | **0.831** |
| LoRA gradients, fp32 | autograd writes fp32 grads (same shape as params) | **0.831** |
| AdamW `exp_avg`, `exp_avg_sq`, fp32 | `torch.optim.AdamW(..., fused=True)` created in `train()`; two fp32 states per param | **1.661** |
| Checkpointed block inputs | `moss_image.py:295-296`: 42 × `checkpoint(layer, x, ...)` saves `x [1,S,3072]` bf16 per sample; S ≈ 4096 image + caption | **4.184** (S=4352) |
| Cached batch / noise / prediction | `trainer.py:456-467`: fp32 latents/noise (`x0`, `noise`, `noisy`); latent 4×128×64×64 | ~0.03 |
| VAE (validation only) | `load_validation_vae` (`trainer.py:110-129`), called when `sample_every>0`; fp32 VAE resident | 0.3–1.0 + decode transient |
| CUDA context / allocator / transient recompute | `main()` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; one block's recompute peak | ~1.0 |
| **Total (no VAE peak)** | | **~18.0** |

The measured 18.13 is reproduced almost exactly by 9.466 + 3.322 + 4.184 + ~1.06 GiB overhead, which validates S ≈ 4352–4608 tokens and confirms that the rank-96/batch-4 run was **not** simultaneously holding the validation VAE peak (or the sampling peak was outside the reported training peak).

At rank 64 the same model is ~15.9 GiB + overhead (~16.9–17.0 GiB), i.e. **1.1 GiB less** than rank 96. The rank-64 qkvo crash is therefore very unlikely to be an OOM **[INFERENCE]**; it is a shape-triggered kernel/driver failure. The padded rank-96/zero-rows workaround changing the result confirms shape sensitivity. **[INFERENCE]** The exact `K=64`/`N=64` LoRA matmul hits a bad cuBLASLt/CUTLASS kernel on sm_120; `bs8` likely hits a second bad shape (larger M), not capacity. Batch 8 at rank 96 is predicted at **22.2 GiB**, below 31.3 GiB usable; if it fails with `unknown error`, memory reduction via NVFP4 is irrelevant to that failure.

### 4.2 What NVFP4 can and cannot reduce

| Cost | NVFP4 effect | Why |
|---|---|---|
| Frozen base 9.47 GiB | **can** reduce to ~2.66 GiB | 4-bit codes (0.5 B/param) + one E4M3 scale per 16 (0.0625 B/param) + one FP32 global scale |
| LoRA params/grad/Adam 3.32 GiB | **cannot** | adapters stay fp32 for export/optimizer |
| Checkpointed activations 4.18 GiB | **cannot** automatically | `torch.utils.checkpoint` stores bf16 inputs; FP4 activation storage would need custom save/restore and would still need bf16 for the LoRA branch |
| VAE / sampling | **cannot** | separate fp32 VAE and inference pipeline |
| Cached batch | negligible | fp32 latents are only ~8 MB per batch |
| Fragmentation | **may worsen** | per-layer dequant allocates/frees a full bf16 weight; `expandable_segments` helps but reserve headroom |

FP8/INT8 storage gives the same structural win at ~4.73 GiB and is supported by native casts; NVFP4 adds 2.07 GiB more saving at the cost of a custom unpack/dequant path.

### 4.3 Capacity scenarios (GiB, S=4352, overhead 1.06)

| Base | Rank | Batch | Base | LoRA+grad+Adam | Ckpt inputs | Total | Verdict |
|---|---|---|---|---|---|---|---|
| bf16 | 64 | 4 | 9.47 | 2.21 | 4.18 | **16.92** | should fit; currently crashes (kernel) |
| bf16 | 96 | 4 | 9.47 | 3.32 | 4.18 | **18.03** | measured 18.13 |
| bf16 | 96 | 8 | 9.47 | 3.32 | 8.37 | **22.22** | predicted fit; reported failure needs diagnosis |
| bf16 | 128 | 8 | 9.47 | 4.43 | 8.37 | **23.32** | predicted fit |
| bf16 | 192 | 8 | 9.47 | 6.64 | 8.37 | **25.54** | tight with validation |
| bf16 | 256 | 8 | 9.47 | 8.86 | 8.37 | **27.76** | very tight; not recommended |
| FP8 | 96 | 8 | 4.73 | 3.32 | 8.37 | **17.48** | comfortable |
| NVFP4 | 96 | 8 | 2.66 | 3.32 | 8.37 | **15.41** | comfortable |
| NVFP4 | 192 | 8 | 2.66 | 6.64 | 8.37 | **18.73** | comfortable |
| NVFP4 | 256 | 8 | 2.66 | 8.86 | 8.37 | **20.95** | fits |
| NVFP4 | 256 | 16 | 2.66 | 8.86 | 16.73 | **29.31** | borderline; validation likely OOM |

**What becomes feasible:** with NVFP4 storage, batch 8 at rank 192 (~18.7 GiB) or rank 256 (~21 GiB) is capacity-feasible; FP8 storage makes batch 8 at rank 96 comfortable and rank 128 plausible. **But batch does not improve throughput much** because `MOSSImageTransformer.forward` loops per sample (`moss_image.py:292-303`); it mainly reduces optimizer steps and gradient noise. The real speed lever is fixing the per-sample loop / batching the 42-layer computation, which is outside this report.

### 4.4 Speed estimates

- **Weight-only storage (O1/O2):** no tensor-core speedup. The extra work is reading 2.7 GiB of codes and writing 9.5 GiB of bf16 per full forward, i.e. ~6.8 ms at 1.79 TB/s, repeated ~2–3× per step with checkpointing: **~15–25 ms/step**, ~0.3–0.5% of 5.286 s. A pure-PyTorch nibble unpack is much slower; a Triton unpack or FP8 native cast is required. **[INFERENCE]**
- **NVFP4 tensor cores (O3):** the upper bound from Amdahl, with base GEMMs ~81% of forward FLOPs (`RESEARCH.md`: attention ~19%) and a 2× FP4 GEMM: `1 / (0.19 + 0.81/2) = 1.68×`. Realistically 1.1–1.4× after quantization kernels, padding, and the bf16 LoRA branch. TE's own sm_120 workaround measures **1.77× bf16 for one GEMM** (M=1024, K=N=4096), not an end-to-end step. **[INFERENCE]**
- **Reference data point:** 6545 tok/s at 30.0 GiB (r32, 4608-token micro-batches, 64× grad accumulation) vs our ~3,300 tok/s at 5.286 s/step for 4×~4350 tokens. The reference is faster per token, but with r32, different checkpointing, block streaming, and token packing; do not attribute the gap to quantization.

### 4.5 Train/inference mismatch

Training against `W_q = dequant(quant(W))` teaches the adapter to produce `ΔW` that minimizes loss on `W_q`. Deployment on bf16 `W` computes `(W + ΔW)x`, not `(W_q + ΔW)x`; the difference is `(W - W_q)x`. For FP8 per-row, the residual is small (E4M3 has 3 mantissa bits, row absmax scaling). For NVFP4, E2M1 has 1 mantissa bit with 16-element blocks; the residual is much larger, especially for outlier-free blocks. At rank 96 the adapter has 223 M parameters, enough to absorb a large part of that residual, which improves training loss but **worsens** the deployment mismatch. **[INFERENCE]** This is not fixable by export: our adapter-only export and `merge.py` both target a bf16 base, and ComfyUI has no NVFP4 MOSS-Image base path in the reference (it refuses `nvfp4`).

Mitigations, in order:
1. Use FP8 storage, not NVFP4, for the first prototype.
2. Measure adapter transfer explicitly: train 200–500 steps on quantized base, then evaluate the same adapter on bf16 base vs quantized base (fixed prompts, PSNR/SSIM, val loss).
3. If NVFP4 is required, train with a “dequantized bf16 shadow” and periodically evaluate on bf16; or distill the adapter's behavior on bf16 after training. Neither is standard; both add work.

---

## 5. Validation plan

### Phase 0 — stack microbenchmarks (1–2 h, no repo changes)

On the 5090, torch 2.14.1+cu130:

1. `torch._scaled_mm` NVFP4 forward with `M=4352, K=3072, N=3072` and `M=4352, K=3072, N=16384`, both operands `float4_e2m1fn_x2`, E4M3 block scales, swizzled; compare to bf16. Record success/exception and TFLOPS.
2. Same for dgrad shapes: `dy [4352,3072] @ W^T [3072,3072]` and the FFN shapes.
3. If installable without touching the training venv, `te.Linear` with `NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)`, forward and `loss.backward()`; record the exact GEMM error.
4. `torchao` `nvfp4_matmul` (if installed in a throwaway env) at `M=4352` (pad to 4352? 4352/128=34, already divisible), `K=3072`, `N=3072`; verify `M%128==0`.

**Gate:** if forward+dgrad do not run at our shapes with cosine ≥0.99 vs bf16 and ≥1.2× GEMM speed, stop Option 3. Do not build the trainer around it.

### Phase 1 — numerical parity (unit tests, CPU/GPU)

- For each of the six projections and each layer, compute `W_q`; report max absolute error, RMS relative error, cosine, SQNR. Thresholds are proposals: FP8 cosine ≥0.9999, NVFP4 ≥0.995; FP8 RMS relative error ≤0.5%, NVFP4 ≤5% **[INFERENCE — set from measured distributions, not assumed]**.
- Fixed batch, fixed seed: compare `prediction` and loss for bf16 vs quantized base. Gate: FP8 loss delta <1e-3; NVFP4 <1e-2 and no NaN/Inf.
- Backward: compare `dL/dx` and `dL/dA`, `dL/dB` on the same batch. Gate: cosine ≥0.999 (FP8), ≥0.99 (NVFP4); no scale underflow (zero block amax → clamp).
- Verify `B=0` at initialization still gives the exact base output for the *same* base format (quantized or not); this checks the wrapper, not quantization.

### Phase 2 — short A/B training (200–500 steps)

- Same seed, same batch order, bf16 baseline vs quantized base; rank 96, batch 4; log loss, grad norm, sec/step, peak VRAM.
- Acceptance: no divergence; loss at step 200 within +2% of baseline; grad-norm distribution comparable; peak VRAM reduction matches the prediction (~4.7 GiB FP8, ~6.8 GiB NVFP4); step time regression <2% for weight-only.
- Save an adapter from each; run validation samples at steps 0/200 with identical prompts/seeds.

### Phase 3 — quality, export and ComfyUI

- Sample 8 fixed validation ids (the YAML already lists them) with the trainer sampler on (a) quantized base and (b) bf16 base. Compute PSNR/SSIM between the two, and compare to the bf16-trained adapter on bf16 base.
- Export with `moss-lora-export-comfy` (or `logging.comfy_export: true`), verify keys `diffusion_model.layers.N.proj.<target>.lora_A/B` + `alpha`, shapes, dtype, and `adapter_config.json` fields (`base_precision`, `base_quant_group`, code hash).
- Load the exported adapter in ComfyUI on a bf16 MOSS-Image base; sample the same prompts; compare to the trainer's bf16-base samples. Gate: no missing keys, no dtype/shape errors, visual/PSNR parity within the tolerance set in Phase 2.
- Run `moss-lora-merge --dtype bf16` and verify the merged transformer samples identically to adapter-on-bf16-base (merge path is the deployment fallback).

### Failure modes and diagnostics

| Symptom | Likely cause | Action |
|---|---|---|
| `CUDA driver error: unknown error` at rank 64 / batch 8 | illegal access in a shape-specific kernel; async error surfaces later | `CUDA_LAUNCH_BLOCKING=1`, `TORCH_USE_CUDA_DSA=1`, `compute-sanitizer --tool memcheck`; isolate the exact matmul; test padded rank 96 vs native 64; test M=8×S separately |
| OOM at batch 8 | capacity, validation VAE, fragmentation | check `max_memory_allocated` vs `max_memory_reserved`; run with `sample_every: 0`; `expandable_segments:True`; reduce rank |
| NaN/Inf after quantization | zero/underflow block scale, FP4 overflow, missing clamp | clamp block amax to `1e-12`; global scale; unit-test extreme rows |
| Loss diverges immediately | quantization error too large or dgrad wrong | compare step-1 grads; lower LR; switch FP8 |
| Adapter works on quantized base, fails on bf16 | train/inference mismatch | Phase 3 PSNR gate; reject NVFP4, keep FP8 |
| Resume changes loss | quantization not deterministic / different checkpoint | hash codes in adapter config; quantize from the same stored weights; compare hashes on resume |
| Throughput regression >2% | per-layer dequant / allocator churn | fuse dequant (Triton), cache one layer, avoid per-block `.weight` access |

---

## 6. Risks and recommendation

### Risk register

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| NVFP4 training unsupported on sm_120 (TE gate, SR/RHT, wgrad) | high | certain today | do not build on it; microbenchmark first |
| NVFP4 dgrad unavailable/slow at our shapes | high | high | Phase 0 gate; fallback FP8 |
| Train/deploy mismatch on bf16 base | high | high for NVFP4, low for FP8 | Phase 3 adapter-transfer test; prefer FP8 |
| Rank-64/batch-8 crashes are unrelated to memory | high | high **[INFERENCE]** | diagnose with sanitizer before buying VRAM |
| Pure-PyTorch FP4 unpack too slow | medium | medium | Triton unpack or FP8 path |
| Export/ComfyUI regression | medium | low | existing export tests + ComfyUI smoke |
| Maintenance burden of a custom quant layer | medium | medium | keep it behind `model.base_precision`, default bf16 |
| WSL2/driver/TE build issues | medium | medium | avoid TE; FP8 path uses torch only |

### Recommendation

**No-go for NVFP4 training on this stack today.** The reference does not implement it; TE documents NVFP4 training only for SM 10.0/10.3; the sm_120 workaround disables SR/RHT and still has no supported wgrad, with dgrad unproven. Building a custom FP4 autograd path around `_scaled_mm`/torchao would be a multi-week, high-risk project whose main payoff—base-weight VRAM—can be obtained safely with FP8 storage.

**Prototype first, in this order:**

1. **Diagnose the existing crashes (highest value, hours).** Reproduce rank-64 qkvo and batch-8 with `CUDA_LAUNCH_BLOCKING=1` and compute-sanitizer. The capacity math says batch 8 at rank 96 is ~22 GiB, so an `unknown error` is very unlikely to be an OOM **[INFERENCE]**; fixing it may unlock batch 8 without any quantization.
2. **FP8 weight-only base prototype (1–3 days).** Option 1 with `base_precision: fp8`: quantize from the loaded bf16 weights, dequantize per layer, keep LoRA fp32, export unchanged. Measure peak VRAM, step time, 200-step loss, and adapter-on-bf16 quality. This is the best effort/benefit ratio and uses only supported sm_120 ops.
3. **NVFP4 storage only if FP8 is insufficient (2–4 days).** Same wrapper, packed E2M1 codes + E4M3 block scales + FP32 global. Start with RTN; do not attempt SR on sm_120. Re-run the adapter-transfer gate; reject if bf16-base quality drops.
4. **NVFP4 tensor cores only after Phase 0 succeeds (2–4 weeks).** If and only if `_scaled_mm`/TE/torchao forward+dgrad pass the microbenchmark at 4352×3072×3072 and 4352×3072×16384 with a real speedup, prototype a custom autograd function. Otherwise drop it.

**What would change the recommendation:** NVIDIA ships an sm_120 NVFP4 training path in TE/cuBLASLt with SR/RHT support and working dgrad; ComfyUI gains an NVFP4 MOSS-Image base loader; or our quality tests show NVFP4 adapters transfer to bf16 within the FP8 tolerance. None of these is true in the inspected code/docs as of 2026-11-04.

---

## References (files actually read)

**This repo**
- `moss_lora/trainer.py` — `load_transformer` 98–107, `load_validation_vae` 110–129, `ParamEMA` 144–160, `train` 404–563, batch/optimizer 448–480, `main()` env.
- `moss_lora/lora.py` — `LoRALinear` 40–78; fp32 A/B, zero B, per-call dtype cast.
- `moss_lora/model/moss_image.py` — `_Block.forward` 166–183; `MOSSImageTransformer.forward` 286–304; checkpoint call 295–296; 42 layers, dim 3072, ffn 8192, latent 128.
- `moss_lora/config.py` — `ModelConfig`, `LoraConfig`, strict YAML.
- `configs/train_5090_v4_5ep_r64.yaml` — rank 64, batch 3, 34,900 steps, sample every 3000.
- `moss_lora/export_comfy.py`, `moss_lora/merge.py`, `moss_lora/sample.py` — adapter-only export; bf16 merge; sampling path.
- `docs/RESEARCH.md:108-114, 216, 442` — 10.16 GB base, fp32 latents, 12.84 GB r32 smoke.
- `docs/5090.md:103-135` — memory ladder and checkpointing estimates.

**Reference (`/tmp/KohakuMossTrainer`, commit `b06ac60`)**
- `src/kohakumoss/quant/comfy_format.py:23-24` — supported Comfy formats (no nvfp4).
- `tests/quant/test_comfy_refusal.py:22, 47-51, 92` — `nvfp4` is the unknown/refused format.
- `src/kohakumoss/quant/weights.py:29-44` — int8/FP8 per-row quantization.
- `src/kohakumoss/quant/compute.py:18-23, 87-140, 163-206` — FP8/MXFP8 compute and caches.
- `src/kohakumoss/quant/w4a8.py:1-8` — W4A8 int4/int6 codes decode to int8.
- `docs/internals/quant-kernels.md:1-27` — precision table; `README.md:154-161` — sm_120 fp8/MXFP8 tested.
- `src/kohakumoss/memory/profiles.py:40-80` — profiles, `partial:` checkpointing, streaming.
- `configs/moss/recipes/recipes_lora_5090.py:18-24` — `partial:0.2`, 4608 tokens, grad acc 64.
- `docs/performance/training-5090.md` — 40.26 s/step, 6545 tok/s, 30.0 GiB, r32.
- `pyproject.toml:13-30` — no TE/torchao/CUTLASS dependency; `lycoris-lora`, Triton/TileLang.

**External**
- NVIDIA TransformerEngine, `docs/features/low_precision_training/nvfp4/nvfp4.rst` — NVFP4 format, SR, RHT, “Supported devices: Training SM 10.0, SM 10.3; Inference SM 10.0+”.
- `NVIDIA/TransformerEngine#2255` — sm_120 default recipe failure, `disable_rht`/`disable_stochastic_rounding` workaround, wgrad `CUBLAS_STATUS_NOT_SUPPORTED`, 1.77× single-GEMM measurement.
- PyTorch 2.14.1 `aten/src/ATen/native/cuda/ScaledBlas.cpp:1187-1230`, `ScaledBlasDeviceUtils.cpp`, `torch/_meta_registrations.py:7203-7215` — NVFP4 `_scaled_mm` plumbing and block layout.
- NVIDIA cuBLAS docs §3.1.4.4 — FP4 block scaling, TN layout, VEC16_UE4M3, CC 10.x/11.0.
- NVIDIA PTX ISA — `mma.sync ... kind::mxf4nvf4` block-scaled FP4; `cvt.rn.satfinite.e2m1x2.f32` on sm_120a; `cvt.rs` absent.
- `pytorch/ao` `torchao/prototype/moe_training/nvfp4_training/nvfp4_linear.py` — prototype NVFP4 training linear, `M/K/N % 128`, `F.scaled_mm`.
