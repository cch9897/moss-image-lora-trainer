# NVFP4 Training Feasibility — Remote GPU Box Audit

Read-only audit of `user@<other-host>` (WSL2 Debian 13 trixie) against
`/home/user/MOSS-Lora-Trainer`, interpreter `/home/user/MOSS-Lora-Trainer/venv/bin/python`.
Nothing was installed, no repo file was modified, no training was started.
Audit date: 2026-11-04.

**Verdict in one line:** the hardware and the *installed* PyTorch can execute a native NVFP4
GEMM on sm_120 at ~1.2 PFLOP/s, but **no NVFP4 training path exists in this environment**:
Transformer Engine, torchao, bitsandbytes, apex and flash-attn are all absent, `cutlass-dsl`
is absent, and `torch._scaled_mm` has no autograd and rejects mixed fp4/bf16 and fp4/fp8 operands.

---

## 1. Host and GPU

| check | command | result | implication / blocker |
| --- | --- | --- | --- |
| Connectivity / OS | `ssh … 'uname -a; cat /etc/os-release'` | `Linux GPU-host 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 5 18:30:46 UTC 2025 x86_64`; `Debian GNU/Linux 13 (trixie)` | WSL2 guest, not bare metal. |
| GPU identity | `/usr/lib/wsl/lib/nvidia-smi` | `NVIDIA GeForce RTX 5090`, `32607 MiB`, `00000000:01:00.0`, `170 SM` (`torch` props), `42 C`, `19 W` idle | `nvidia-smi` is **not on `PATH`**; only `/usr/lib/wsl/lib/nvidia-smi` works (WSL passthrough driver). |
| Driver / CUDA UMD | `nvidia-smi` header + `-q` | `NVIDIA-SMI 620.67.01`, `KMD Version: 621.01`, `CUDA UMD Version: 13.5`; query API reports `driver_version 621.01` | Driver is new enough for every CUDA 12.8+/13.x wheel; not a blocker. |
| Compute capability | `nvidia-smi --query-gpu=compute_cap` / `torch.cuda.get_device_capability()` | `12.0` (sm_120) in both | Target arch for NVFP4 block-scaled MMA; confirmed by both driver and torch. |
| Power / clocks | `nvidia-smi -q \| grep -i power` | `Current 540.00 W`, `Requested 540.00 W`, `Default 600.00 W`, `Max 600.00 W` | Card is power-capped 60 W below default, so sustained FLOP/s will sit below the 1676 TFLOPS spec sheet. |
| Memory | `torch.cuda.get_device_properties(0).total_memory` | `34162343936` bytes = `31.8 GiB` (32607 MiB reported by driver) | Fine for LoRA on a 32 GB card; not relevant to FP4 GEMM support. |
| Disk | `df -h /` | `/dev/sdd 1007G, 583G used, 374G avail (61%)` | Enough for a CUDA wheel + kernels; not a blocker. |
| Host RAM / swap | `free -g` | `Mem 47 GiB total, 45 available; Swap 63 GiB` | Not a blocker. |
| WSL version | `wsl.exe --version` | `WSL 2.5.10.0`, kernel `6.6.87.2-1`, `WSLg 1.0.66`, Windows `10.0.26200.8246` | Modern WSL2; GPU passthrough working. |

## 2. Python / PyTorch

| check | command | result | implication / blocker |
| --- | --- | --- | --- |
| Interpreter | `venv/bin/python -VV` | `Python 3.13.5 (main, Jun 25 2025) [GCC 14.2.0]` | cp313 wheels required for any compiled extension. |
| PyTorch build | `torch.__version__`, `torch.version.cuda` | `2.14.1+cu130`, CUDA `13.0`, `cudnn 92400` (9.24.0), `git_version 5c4886908584029761b579af026dcfb627c84070` | cu130 wheel on a CUDA 13.5 UMD driver — forward compatible. |
| Device visible to torch | `torch.cuda.is_available()`, `get_device_capability()` | `True`, `(12, 0)`, `NVIDIA GeForce RTX 5090` | Torch sees the 5090 natively. |
| Compiled arch list | `torch.cuda.get_arch_list()` | `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` | **`sm_120` kernels are compiled in** — no PTX-JIT-only fallback. |
| FP4 dtype | `hasattr(torch,'float4_e2m1fn_x2')` | `True` — `torch.float4_e2m1fn_x2` | Packed E2M1 type exists (2 values/byte). |
| Scale dtypes | `hasattr(...)` | `torch.float8_e8m0fnu` `True`, `torch.float8_e4m3fn` `True`, `torch.float8_e5m2` `True`, `torch.int4`/`torch.uint4` `True` | Both NVFP4 (e4m3 per-16) and MXFP4 (e8m0 per-32) scale types are first-class. |
| BLAS backend | `torch.backends.cuda.preferred_blas_library()` | `_BlasBackend.Cublas` | cuBLASLt path, i.e. the vendor FP4 GEMM is reachable. |
| cuBLAS runtime | `pip list \| grep nvidia-cublas` | `nvidia-cublas 13.1.1.3` | cuBLAS 13.x has block-scaled (NVFP4) GEMM support; confirmed empirically below. |
| `_scaled_mm` schema | `torch._C._get_schema("aten::_scaled_mm","")` | `aten::_scaled_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor` | Classic 1-scale-per-side API; accepts `float4_e2m1fn_x2` (see `_check_scaled_mm_sizes`). |
| `_scaled_mm_v2` schema | `torch._C._get_schema("aten::_scaled_mm_v2","")` | `(Tensor self, Tensor mat2, Tensor[] scale_a, int[] recipe_a, int[] swizzle_a, Tensor[] scale_b, int[] recipe_b, int[] swizzle_b, Tensor? bias, ScalarType? out_dtype, int[] contraction_dim=[], bool use_fast_accum=False)` | Newer block-scale API carrying explicit recipes/swizzles. |
| Public wrapper | `inspect.signature(torch.nn.functional.scaled_mm)` | `(mat_a, mat_b, scale_a, scale_recipe_a, scale_b, scale_recipe_b, swizzle_a=None, swizzle_b=None, bias=None, output_dtype=torch.bfloat16, contraction_dim=(), use_fast_accum=False)` | Public, non-underscore entry point exists in 2.14. |
| Scale recipes | `torch.nn.functional.ScalingType.__members__` | `TensorWise:0, RowWise:1, BlockWise1x16:2, BlockWise1x32:3, BlockWise1x128:4, BlockWise128x128:5`; `SwizzleType: NO_SWIZZLE:0, SWIZZLE_32_4_4:1` | `BlockWise1x16` + e4m3 scales is exactly the NVFP4 recipe; `BlockWise1x32` + e8m0 is MXFP4. |

## 3. Installed packages, CUDA toolkit

| check | command | result | implication / blocker |
| --- | --- | --- | --- |
| Package inventory | `venv/bin/pip list \| grep -iE "transformer\|torch\|flash\|bitsandbytes\|apex\|triton\|ninja\|nvidia\|cuda\|cublas\|cutlass\|accelerate"` | `torch 2.14.1`, `transformers 5.20.1`, `accelerate 1.15.0`, `triton 3.8.0`, `ninja 1.13.2`, `flash-linear-attention 0.5.2`, `nvidia-cublas 13.1.1.3`, `nvidia-cudnn-cu13 9.24.0.43`, `nvidia-nccl-cu13 2.30.7`, `nvidia-cusparselt-cu13 0.8.1`, `cuda-toolkit 13.0.3.0`, `cuda-bindings 13.4.4` | 79 packages total. **No** `transformer-engine`, `torchao`, `flash-attn`, `bitsandbytes`, `apex`, `cutlass`, `nvidia-modelopt`. |
| CUDA toolkit | `which nvcc` | `nvcc: command not found` (not on `PATH`) | Toolkit presence must be tested by path. |
| nvcc by path | `/usr/local/cuda/bin/nvcc --version` | `Cuda compilation tools, release 13.3, V13.3.73`; `/usr/local/cuda -> /usr/local/cuda-13.3`; `/usr/local/cuda-13.0` also installed | Full nvcc 13.3 available for JIT/extension builds; only `PATH` is missing (a shell-export, not an install). |

## 4. Import tests (venv, no installs)

| check | command | result | implication / blocker |
| --- | --- | --- | --- |
| Transformer Engine | `python -c "import transformer_engine"` | `FAIL transformer_engine  ModuleNotFoundError: No module named 'transformer_engine'` | **Blocker** for any TE-based NVFP4 recipe (`DelayedScaling`/`NVFP4BlockScaling`). |
| TE pytorch backend | `import transformer_engine.pytorch` | `ModuleNotFoundError: No module named 'transformer_engine'` (same missing top-level) | No fused TE Linear/GEMM. |
| torchao | `import torchao`, `import torchao.quantization` | `ModuleNotFoundError: No module named 'torchao'` (both) | **Blocker** for `torchao` NVFP4 quantization + QAT flows; also absent from system `python3`. |
| flash-attn | `import flash_attn`, `flash_attn_2_cuda` | `ModuleNotFoundError: No module named 'flash_attn'` / `'flash_attn_2_cuda'` | Not required by this repo's `requirements.txt`; no blocker for training, only for FA-specific kernels. |
| bitsandbytes | `import bitsandbytes` | `ModuleNotFoundError: No module named 'bitsandbytes'` | Repo does not use it; 4-bit bnb path unavailable. |
| apex | `import apex` | `ModuleNotFoundError: No module named 'apex'` | No apex fused optimizers / FP4 experiments. |
| triton | `import triton` | `OK triton version=3.8.0 file=…/site-packages/triton/__init__.py` | Triton is present and imports cleanly. |
| triton FP4 language support | `dir(triton.language)` filtered for `float4`/`nvfp4` | `[]` (empty) | No first-class `tl.float4` dtype symbol; FP4 in Triton would need bit-manipulation, not a ready path. |
| cutlass-dsl (for torch's CutlassDSL/quack NVFP4 templates) | `import cutlass` | `FAIL ModuleNotFoundError: No module named 'cutlass'` | torch's vendored `quack`/`cutedsl` block-scaled kernels (`torch/_vendor/quack/*`) cannot run; only the cuBLASLt-backed eager path works. |

## 5. Does torch/CUDA actually run FP4 GEMM on sm_120? — **Yes**

Throwaway scripts in `/tmp` (no repo edits). FP4 tensors were built by encoding nibbles
directly into `torch.float4_e2m1fn_x2`, with a verified lossless pack/unpack roundtrip.

| check | command | result | implication / blocker |
| --- | --- | --- | --- |
| Pack roundtrip | `/tmp/nvfp4_smoke2.py` TEST 3 | `PASS pack/unpack roundtrip equal: True` | Test inputs are exactly representable E2M1 values, so the reference matmul is exact. |
| NVFP4 `_scaled_mm` correctness | `torch._scaled_mm(a4, b4, sa, sb, out_dtype=torch.bfloat16)`, `M=256,K=512,N=384`, e4m3 1×16 block scales | `PASS shape (256, 384) torch.bfloat16 max_abs_err 2.0 relerr 0.0020100502512562816 allfinite True` | Native FP4 GEMM produces correct, finite results on sm_120. |
| `_scaled_mm_v2` NVFP4 recipe | `ScalingType.BlockWise1x16` + `SwizzleType.SWIZZLE_32_4_4`, scale shapes `(M,K//16)`, `(K//16,N)` | `PASS shape (256, 384) torch.bfloat16 max_abs_err 2.0` | Both the legacy and the recipe-based API work. |
| Kernel actually used | `torch.profiler` over `_scaled_mm` at 4096³ | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn`, `109.4 us/call` | **Definitive**: a CUTLASS3 sm120 block-scaled `ue4m3 × e2m1` tensor-op kernel — real FP4 tensor cores, not emulation. |
| FP4 throughput | 20 iters, 4096³ | `0.11 ms/gemm ~1215.78 TFLOP/s` | ~72% of the 1676 TFLOPS dense FP4 peak listed in this repo's `docs/5090.md` (card is capped at 540 W, not 600 W). |
| FP8 contrast | `_scaled_mm` fp8×fp8, 4096³ | `0.26 ms ~528.62 TFLOP/s` | FP4 is ~2.3× the measured FP8 throughput. |
| BF16 contrast | `x @ y` bf16, 4096³ | `0.62 ms ~220.11 TFLOP/s` | FP4 is ~5.5× the measured BF16 throughput. |
| **Autograd through FP4 GEMM** | FP4 operands + `out.float().sum().backward()` | `out.requires_grad: False grad_fn: None` → `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | **Blocker**: FP4 GEMM is forward-only. No backward kernel is exposed; there is no autograd graph to train through. |
| FP4 → bf16 dequant | `a4.to(torch.bfloat16)` | `DEQUANT PASS (256, 256) torch.bfloat16` | Dequant is possible (so a manual STE path is constructible), but it is not a fused FP4 training kernel. |
| Mixed fp4 × fp8 (W4A8) | fp8 `[M,K]` × fp4 `[K/2,N]` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x512 and 256x384)` | Mixed-precision FP4/FP8 GEMM is not accepted by the eager aten path. |
| Mixed bf16 × fp4 | bf16 `[M,K]` × fp4 `[K/2,N]` | `CUDA error: device-side assert triggered` (`c10/core/DynamicCast.h:82: fetch_and_cast: … Assertion 'false' failed`) | BF16 activations against an FP4 weight are **not** supported; it hard-fails in the kernel (process-local, no persistence, but a poisoned CUDA context for that process). |

## 6. Reference repo: `KohakuBlueleaf/KohakuMossTrainer`

Shallow clone at `/tmp/KohakuMossTrainer` (inspected only; nothing installed).

| check | command | result | implication / blocker |
| --- | --- | --- | --- |
| Install metadata | `cat pyproject.toml` | `requires-python >=3.10`; deps: `torch>=2.13,<2.15`, `numpy`, `packaging`, `pillow>=9.4`, `requests`, `safetensors`, `tqdm`, `kohaku-engine`, `anyschedule`, `kohakuvault>=1.1`, `lycoris-lora>=4.1.3`, `diffusers>=0.38`, `transformers>=5.3`, `accelerate` | Matches the installed torch 2.14.1 exactly; the remote venv is already inside the pin. |
| Optional extras | same file | `kernels-cuda = [tilelang>=0.1.15, triton-windows>=3.7,<3.9; win32]`, `quant = [bitsandbytes]`, `baselines = [flash-linear-attention, causal-conv1d]` | No `quant` extra beyond bitsandbytes; **no transformer-engine, no torchao, no NVFP4 extra**. |
| NVFP4 in the codebase | `grep -rn -i "nvfp4\|float4\|torchao\|transformer.engine"` | Only 5 hits, all in `tests/quant/test_comfy_refusal.py` and `tests/quant/test_comfy_formats.py`, asserting that a ComfyUI `format: "nvfp4"` checkpoint is **rejected** with `ValueError` | Upstream treats `nvfp4` as an unsupported *import* format, not a training precision. |
| Supported quant formats | `src/kohakumoss/quant/weights.py` | `FORMATS = ("int8", "fp8")` | No FP4 weight/activation format in the trainer. |
| Arch gating | `src/kohakumoss_kernels/hardware/arch.py` | `has_fp8: bool`, set as `entry >= 89`; **no `has_fp4` field exists** | The capability model has no FP4 concept at all. |
| Documented precision matrix | `README.md` "Supported GPUs" | sm120 row: `Triton, then TileLang, compile, eager` / `bf16, fp16` / `fp8, MXFP8 (block-scaled MMA)` / `tested` / `RTX 5090` | Upstream's fastest documented sm120 compute is fp8/MXFP8 — **not** NVFP4. |
| Install guide | `docs/guides/install.md` | CUDA index selection: driver CUDA ≥13.0 → cu130, ≥12.8 → cu128, ≥12.6 → cu126; venv Python 3.12; `torch>=2.13,<2.15` | Would produce a cu130 env on this box (driver reports 13.5), i.e. same torch family as installed. |

## 7. What the environment does and does not support

Verified capabilities:

- sm_120 FP4 tensor cores work through cuBLASLt/CUTLASS3: `torch._scaled_mm` and
  `F.scaled_mm` with `BlockWise1x16`/e4m3 (NVFP4) and `BlockWise1x32`/e8m0 (MXFP4) scales.
- Measured 1215.78 TFLOP/s at 4096³ (fp4) vs 528.62 (fp8) and 220.11 (bf16).
- `torch.float4_e2m1fn_x2`, `torch.float8_e8m0fnu`, `torch.float8_e4m3fn` are all present;
  `sm_120` is in the compiled arch list.

Verified blockers for *training* with NVFP4:

1. `torch._scaled_mm` is forward-only (no `grad_fn`, `requires_grad=False`); a backward error
   test raises `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`.
2. No quantization/recipe library is installed: `transformer_engine`, `torchao` both
   `ModuleNotFoundError` (repo venv *and* system `python3`).
3. `cutlass-dsl` is absent, so torch's vendored CutlassDSL/quack block-scaled kernels
   (`torch/_vendor/quack/blockscaled_gemm_utils.py`) cannot be used; the working path is
   cuBLASLt-backed eager only.
4. Eager FP4 GEMM accepts only fp4×fp4; fp4×bf16 hits `c10/core/DynamicCast.h:82`
   `Assertion 'false' failed`, and fp4×fp8 is rejected with a shape error.
5. The target repo has no FP4 code path at all (`FORMATS = ("int8","fp8")`, no `has_fp4`),
   and the reference project documents only fp8/MXFP8 on sm120.

## 8. Unverified items

- Whether a `transformer-engine` wheel with sm_120 NVFP4 kernels exists for cp313 +
  torch 2.14/cu130, and whether it would import without ABI changes — [UNVERIFIED]
  (no install permitted; nothing to inspect locally).
- Whether `torchao`'s NVFP4 QAT path supports sm_120 on torch 2.14.1 — [UNVERIFIED]
  (package absent).
- Whether TileLang (upstream's sm120 kernel backend, `tilelang>=0.1.15`) ships FP4 MMA
  support — [UNVERIFIED] (not installed on the box, not audited).
- End-to-end training numerics with an FP4 base + LoRA adapters — [UNVERIFIED]
  (cannot be measured without a backward path; audit was read-only).
