# `CUDA driver error: unknown error` on RTX 5090 / sm_120 / WSL2 — external research

Research date: 2026-11-04. CPU/web only; no GPU was touched. Sources: GitHub issue/PR API,
NVIDIA Developer Forums API, NVIDIA CUDA/cuBLAS documentation, PyPI metadata, and the
locally installed `torch 2.14.1+cpu` source tree (read-only, same version as the GPU box).

Target environment: RTX 5090 32 GB (sm_120), WSL2, Windows driver 621.01,
`torch 2.14.1+cu130`, `transformers 5.20.1`, `diffusers 0.41.0`, `nvidia-cublas 13.1.1.3`.
Symptom: native LoRA rank 64 with q/k/v/o targets raises `RuntimeError: CUDA driver error:
unknown error` at batch 1/4/8; rank-64 FFN-only works; rank 96/128 all-targets work at bs1;
rank 96 bs4 works, bs8 fails; padded rank-64 (rank-96 kernel) works at bs1/bs4, fails at bs8.

---

## 0. Bottom line

1. **`CUDA driver error: ...` is not a cuBLAS error.** In PyTorch the string comes from
   `C10_CUDA_DRIVER_CHECK` / `AT_CUDA_DRIVER_CHECK`, which wrap **CUDA Driver API**
   (`cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuLaunchKernel`, …). A cuBLAS or
   cuBLASLt failure is reported by `TORCH_CUDABLAS_CHECK` as
   `CUDA error: CUBLAS_STATUS_... when calling ...`. This is verifiable in the local
   `torch 2.14.1` headers: `torch/include/ATen/cuda/Exceptions.h:183-204` vs `:63-70`.
2. **The best-documented cause of this exact message in training is
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on WSL2/Windows**, where the
   allocator's CUDA-VMM calls (`cuMemSetAccess`) return `CUDA_ERROR_UNKNOWN (999)`.
   There are at least six independent reports, one with **the same `torch 2.14.1+cu130`**.
   This repo force-enables that setting: `moss_lora/trainer.py:539` and
   `moss_lora/benchmark.py:32` (both `os.environ.setdefault(...)`), and the local
   `docs/5090.md` §3.1/§7 already records one such crash on this machine (traced to WSL
   `RLIMIT_NOFILE`/`dxg` EMFILE), fixed by `ulimit -n 65536`.
3. Because the allocator failure is **intermittent and allocation-history dependent**
   (documented explicitly in the AerialLidarClassifier thread), a rank/batch correlation
   is *not* proof of a shape-specific GEMM bug. The rank-64 story is a plausible but
   currently unproven inference. The message text itself is strong evidence **against** a
   cuBLAS kernel as the direct source.
4. Real sm_120 cuBLAS bugs do exist (NVIDIA is investigating one shape-specific cuBLASLt
   split-K kernel; CUDA 13.2 release notes list sm_120 GEMM issues introduced in 13.1.0),
   but none matches a plain 2-D fp32 `x @ B^T` with K=64, and none produces a driver-API
   error string. Keep them as plan B.
5. **No public report was found** of a LoRA rank-64 (or diffusion batch-8) shape-specific
   `CUDA driver error: unknown error` on sm_120. See §4 for the explicit no-match list.

Highest-value next action (GPU agent): rerun the exact failing config with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (plus `ulimit -n 65536`) and with
`TORCH_SHOW_CPP_STACKTRACES=1`. That single A/B distinguishes the allocator/VMM theory from
the cuBLAS theory; see §2.

---

## 1. Ranked evidence table

Rank = how likely the item explains *our* failure and how actionable it is.

| # | Source (URL, date, state) | Environment | Exact symptom | Match to rank-64 / bs-8 | Workaround | Testable without install/sudo | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | [pytorch/pytorch#192330](https://github.com/pytorch/pytorch/issues/192330) — 2026-08-06, closed 2026-09-12 (fixed) | WSL2, RTX A4000/A5000, drivers 581.80/596.86/610.88, torch 2.8.0+cu128 / 2.13.0+cu126 / 2.14.0.dev+cu132 | First CUDA allocation with `PYTORCH_ALLOC_CONF=expandable_segments:True` fails: `RuntimeError: CUDA driver error: unknown error`, raised from `setAccess at c10/cuda/CUDACachingAllocator.cpp`; `cuMemSetAccess` → `CUDA_ERROR_UNKNOWN (999)` when the driver advertises `gpuDirectRDMACapable=1` | **Exact message + exact platform**; not shape-specific (fails at first alloc). NVIDIA confirmed a **WSL driver bug** and shipped a fix in R615 (616.92): attribute 110 now reports 0 | Disable `expandable_segments`; or use driver ≥ 616.92. Our 621.01 should contain the fix **[INFERENCE]** | Yes (env var only) | **High** for the message class; medium for our box because 621.01 post-dates the fix |
| 2 | [iconnorthomas/GenFill PR #2](https://github.com/iconnorthomas/GenFill/pull/2) — 2026-10-14, closed | Windows 11 WDDM, **torch 2.14.1+cu130**, driver 610.88, RTX PRO 6000 Blackwell (sm_120) | Every local generation fails at `pipe.to(device)`: `CUDA driver error: unknown error`; driver log: `Failed to create GPU mapping`, `Returning 999 (CUDA_ERROR_UNKNOWN) from cuMemSetAccess`; `expandable_segments:True` fails, unset works | **Same torch build, same cu13.0, same sm_120, exact message**; not shape-specific | Do not set `expandable_segments` on Windows/WSL | Yes | **High** |
| 3 | [akharroubi/AerialLidarClassifier#8](https://github.com/akharroubi/AerialLidarClassifier/issues/8) — 2026-10-19, closed 2026-10-29 | Windows 11, **torch 2.14.1+cu126**, driver 596.72, 2× RTX A6000 | `CUDA driver error: unknown error` on model load; a plain `torch.empty(1<<25, device="cuda")` reproduces it **intermittently** — sometimes 4 MiB is enough, sometimes a larger allocation is needed; standard allocator works | **Same torch 2.14.x + exact message**; proves the failure is intermittent and allocation-size/history dependent — a natural explanation for apparent rank/batch specificity | Standard allocator (a probe allocation was *not* reliable because the failure is intermittent) | Yes | **High** |
| 4 | [unslothai/unsloth#2408](https://github.com/unslothai/unsloth/issues/2408) (2025-04-25, closed) + [#3511](https://github.com/unslothai/unsloth/issues/3511) (2025-10-26, closed) + [unsloth-zoo PR #1648](https://github.com/unslothai/unsloth-zoo/pull/1648) (2026-10-09, merged) | WSL2 (incl. Docker on WSL2), RTX 4080 SUPER / RTX 6000 Ada, driver 581.57, CUDA 13.0 | `RuntimeError: CUDA driver error: unknown error` during training / at first GPU buffer allocation. Maintainer root-cause: "`CUDA driver error:` … comes from `C10_CUDA_DRIVER_CHECK`, which the caching allocator only uses on the expandable segment path"; `cuMemCreate`/`cuMemAddressReserve`/`cuMemMap` fail on WSL2; newer Unsloth strips `expandable_segments` on WSL | **Exact message, training workload, LoRA fine-tuning**; not shape-specific | Strip `expandable_segments` on WSL2/Windows (kernel-string detection for Docker) | Yes | **High** (maintainer-confirmed mechanism) |
| 5 | [pytorch/pytorch#182286](https://github.com/pytorch/pytorch/issues/182286) — 2026-05-04, **open** | WSL2, RTX PRO 6000 Blackwell 96 GB, torch 2.10+cu128 and 2.11+cu130, drivers 596.36/581.80 | Allocator cannot satisfy 3–18 GiB contiguous allocations; explicitly: "`PYTORCH_ALLOC_CONF=expandable_segments:True` breaks CUDA driver entirely on Blackwell with `RuntimeError: CUDA driver error: unknown error`" | Exact message + Blackwell + WSL2; not shape-specific | Disable `expandable_segments` | Yes | **High** for message class |
| 6 | [NVIDIA/TensorRT-LLM#8243](https://github.com/NVIDIA/TensorRT-LLM/issues/8243) — 2025-10-09, open | B200, driver 580.82, CUDA 13.0 | `expandable_segments:True` crashes at startup (CUDA graph/workspace address invalidation suspected) | Different GPU/workload; confirms `expandable_segments` instability on new architectures | Disable it | Yes | Medium/low |
| 7 | [NVIDIA Forums topic 372787](https://forums.developer.nvidia.com/t/cublaslt-sm-120-blackwell-tf32-split-k-nvjet-kernel-raises-warp-barrier-arrival-mismatch-intermittent-illegal-ac/372787) — 2026-06-10; NVIDIA replies 2026-07-17 | RTX 5090 Laptop, driver 595.71.05, CUDA 13.3, **cuBLAS 13.5.1.27**, llama.cpp fp32 GEMMs via cuBLASLt | cuBLASLt JIT kernel `nvjet_sm120_sss_tf32_mma_32x32x64_..._splitK_TNNN` raises **"Warp Barrier Arrival Mismatch"** → intermittent `cudaErrorIllegalAddress` or GPU hang; *shape-specific* (certain FP32 GEMMs on sm_120), disappears under serialization | **Shape-specific sm_120 GEMM bug — the closest "library bug" analogue**, but: TF32 math, cuBLAS**Lt**, laptop 5090, and the error is **illegal access/hang, not a driver-error string** | `NVIDIA_TF32_OVERRIDE=0` (proven by reporter: 960+ clean generations). NVIDIA is reproducing; **no fix announced as of the last post (2026-07-17)** | Yes (env var) | **Medium** as a class; low as the direct cause of *this* message |
| 8 | [ultralytics/ultralytics#26300](https://github.com/ultralytics/ultralytics/issues/26300) — 2026-09-23, closed | RTX PRO 6000 Blackwell sm120, driver 595.84, torch 2.14.0+cu132, nvidia-cublas **13.4.0.1** | `CUDA error: an illegal memory access was encountered` inside cuBLAS kernel `nvjet_sm120_hsh_mma_16x32x32_..._TNNN` (fp16 TMA GEMM); context corrupted afterwards | Illegal access, **not** `CUDA driver error`; fp16 kernel, not fp32 | **nvidia-cublas 13.8.0.4** fixed it (1,500 clean calls); later closed as not standalone-reproducible | Requires pip install (no sudo) | Medium (shows 13.x cuBLAS sm_120 kernel churn; our 13.1.1.3 is older) |
| 9 | [CUDA Toolkit 13.2 release notes → cuBLAS known issues](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) (cuBLAS 13.2 / 13.2 Update 2 sections) | sm_120/12.1, cuBLAS 13.1.0+ | (a) Strided **batched** GEMM with broadcast operands may read out of bounds on cc 12.0/12.1 — introduced in cuBLAS **13.1.0** [6040940][5996751]; (b) GEMM kernels on cc 10.x/12.x may read device alpha/beta before `cudaGridDependencySynchronize()` (PDL WAR hazard) — introduced in **13.1.0** [CUB-10409] | **Documented sm_120 GEMM bugs in exactly our cuBLAS minor (13.1.x)** — but (a) is batched/broadcast only, (b) is PDL-only (PyTorch does not use programmatic dependent launch for GEMM). No plain 2-D GEMM entry | Upgrade cuBLAS; avoid affected shapes/features | Env-only? No — needs cuBLAS upgrade | Medium as context; **low** as direct match |
| 10 | [ggml-org/llama.cpp#24398](https://github.com/ggml-org/llama.cpp/issues/24398), [#24399](https://github.com/ggml-org/llama.cpp/issues/24399) — 2026-06-10, closed | RTX 5090 Laptop, driver 595.71.05, CUDA 13.3, cuBLAS 13.5.1.27 | Two *different* sm_120 device exceptions ("Warp Misaligned Register" in flash-attn kernel; out-of-range shared-memory store in `mul_mat_q<Q8_0,128>`) | Device-kernel bugs in a different codebase; not our GEMM; shown for the "same box, three distinct sm_120 bugs" context | Remove offending kernel/weights | n/a | Low (context only) |
| 11 | [NVIDIA/TransformerEngine#2255](https://github.com/NVIDIA/TransformerEngine/issues/2255) — 2025-10-09, open | RTX 50 series, TE 2.8 built for sm_120 | `FP4 cvt PTX instructions are architecture-specific. Try recompiling with sm_XXXa` | **No match** — compile-arch issue, unrelated to our runtime error | Build with `sm_120a` | n/a | None (context) |
| 12 | [verl-project/verl#3664](https://github.com/verl-project/verl/issues/3664) — 2025-10-02, closed by cleanup bot | WSL2, RTX PRO 6000 Blackwell sm_120, torch 2.8.0+cu128, driver 581.42 | `RuntimeError: CUDA driver error: unknown error` during FSDP `state_dict()` in rollout mode | Same message + sm_120 + WSL2; different call site (tensor copy/state dict), **no diagnosis or fix** in the issue | None recorded | n/a | Low (unresolved data point) |
| 13 | [vllm-project/vllm#41619](https://github.com/vllm-project/vllm/issues/41619) — 2026-05-04, closed | WSL2, RTX PRO 6000 Blackwell, torch 2.10+cu128 | WSL2 exposes ~16 GiB hidden CUDA overhead; large contiguous allocations fail despite free VRAM; companion to #182286 | Background on WSL2 memory accounting, not our message | Non-WSL Linux | n/a | Low |

### Notes on the two strongest rows

* **PyTorch #192330** is the only report with an NVIDIA confirmation and a named root cause
  (`gpuDirectRDMACapable` VMM flag plus `cuMemSetAccess` → 999). It was **fixed in R615
  (616.92)**; our driver 621.01 is newer, so this *specific* defect should be gone. The
  mechanism (allocator VMM failing on WSL2) still recurs through other paths — see rows 2–5.
* **GenFill PR #2** is the closest environment match: `torch 2.14.1+cu130`, sm_120,
  `expandable_segments:True`, and the same `cuMemSetAccess` → 999. It is native Windows
  WDDM, not WSL2, but WSL2's GPU-PV path is also a paravirtualized WDDM path.

---

## 2. Tests to run, in priority order

All are environment-variable/API tests; no package installs, no sudo. Run each on the
*exact* failing config (rank 64, q/k/v/o, bs1 first) and record the full stderr.

### Step 0 — capture the failure verbatim (do this first, it decides everything)

```bash
ulimit -n                      # expect 1024 if the WSL default was never raised
dmesg | tail -80               # look for: dxg ... get_unused_fd_flags failed ... EMFILE,
                               #           nvrm, Xid, "GPU has fallen off the bus"
TORCH_SHOW_CPP_STACKTRACES=1 \
CUDA_LAUNCH_BLOCKING=1 \
TORCH_USE_CUDA_DSA=1 \
python -m moss_lora.trainer --config <failing rank64 bs1 config>
```

What to look for:

* If the Python traceback ends in `torch/optim/optimizer.py` / `AdamW._init_group` /
  `torch.zeros_like` / any allocation, or the C++ stack mentions
  `c10/cuda/CUDACachingAllocator.cpp`, `ExpandableSegment`, `cuMemCreate`, `cuMemMap`,
  `cuMemSetAccess` → **allocator/VMM path** (rows 1–6). Rank/batch correlation is then an
  allocation-pattern artifact.
* If it ends inside a cuBLAS call and the message is
  `CUDA error: CUBLAS_STATUS_... when calling ...` → different failure than reported;
  re-capture the exact text.
* If the message carries extra driver log lines (`Failed to create GPU mapping`,
  `Returning 999 ... from cuMemSetAccess`) → definitive allocator/VMM.
* `dmesg` Xid / "fallen off the bus" → hardware/driver reset, not a GEMM bug.

Send back: exact first 15 lines of the traceback, the C++ stack, `ulimit -n`, dmesg tail.

### Step 1 — raise the fd limit (cheapest, matches this repo's prior incident)

```bash
ulimit -n 65536
ulimit -n                      # verify
python -m moss_lora.trainer --config <rank64 bs1 config>
```

* If this alone fixes it → WSL `RLIMIT_NOFILE`/`dxg` exhaustion with `expandable_segments`
  (local `docs/5090.md` §7 already diagnosed this exact failure once). Keep the ulimit in
  every launcher; no code change strictly required, but see Step 2.
* If it does not → continue.

### Step 2 — disable expandable segments (the decisive A/B)

Both names are set because `trainer.py` writes the legacy `PYTORCH_CUDA_ALLOC_CONF`, while
torch 2.14 documents `PYTORCH_ALLOC_CONF`; set both to the same value so `setdefault`
cannot re-enable it.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_ALLOC_CONF=expandable_segments:False
python -m moss_lora.trainer --config <rank64 bs1 config>
```

Then the matrix: rank64 bs1, bs4, bs8; rank96 bs8; padded rank64 bs8.

* Fixed everywhere → root cause is the expandable-segment allocator on this WSL2 stack.
  Keep it disabled in the launcher (and consider changing the `setdefault` defaults to
  platform-aware, like Unsloth/GenFill did — but that is a repo change, not part of this
  report).
* Fixed at bs1 but not bs8 → allocator is involved, but there is a second, memory-pressure
  or shape-dependent problem; continue to Step 3 with the failing bs8 case.
* No change → the message is *not* the expandable allocator; go to Step 3.

Optional confirmation: re-enable `expandable_segments:True` and repeat once; if the failure
returns only in that mode, the A/B is conclusive.

### Step 3 — determine whether cuBLAS/cuBLASLt is even in the failing call

```bash
CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE=/tmp/cublaslt.log \
python -m moss_lora.trainer --config <failing config>
```

* Empty/absent `cublaslt.log` around the failure → cuBLASLt is not on the failing path.
* Non-empty → note the last `algo`/`config id`, compute type, and `M,N,K` before the crash;
  that is the shape to target in Step 5.

Then force each BLAS backend in turn (API, no env var needed):

```python
import torch
print(torch.backends.cuda.preferred_blas_library())   # default: cublas
torch.backends.cuda.preferred_blas_library("cublaslt")  # or "cublas"
```

Run the failing config once per backend. If only one backend fails, that pins the library.
(Env equivalent: `TORCH_BLAS_PREFER_CUBLASLT=1`.)

### Step 4 — TF32 / split-K knobs (only meaningful if Step 3 implicates cuBLAS)

```bash
python - <<'PY'
import torch
print("allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
print("fp32_precision:", torch.backends.cuda.matmul.fp32_precision)  # "none" = IEEE on 2.14
print("float32_matmul_precision:", torch.get_float32_matmul_precision())
PY
```

* If TF32 is somehow enabled (a config/plugin may set it), rerun with
  `NVIDIA_TF32_OVERRIDE=0` — this is the *proven* workaround for the NVIDIA forum sm_120
  split-K bug (row 7).
* If TF32 is off (expected: `allow_tf32=False`, `fp32_precision="none"`), the row-7 bug is
  unlikely to be hit, but the test is free; also try the inverse
  (`torch.backends.cuda.matmul.allow_tf32=True`) purely as a diagnostic — if the failure
  *disappears* when TF32 is enabled, that is strong evidence of a bad FP32 kernel.
* Disabling split-K is documented but has no documented env syntax:
  `CUBLAS_BATCH_INVARIANCE_FLAGS` "can also be controlled process-wide", and enabling
  batch invariance "disables split-K" ([cuBLAS docs, Experimental: Batch Invariance](https://docs.nvidia.com/cuda/cublas/index.html)).
  The accepted string values are **not documented** [UNVERIFIED]; treat as a last resort,
  not a first-line test.

### Step 5 — isolate the exact GEMM shape (only if Steps 1–4 leave cuBLAS implicated)

Standalone, no trainer, fp32, same layout as the LoRA branch (`F.linear(x, B)` with
`B: [out, rank]`, i.e. TN GEMM), swept over the real shapes:

```python
import torch, itertools
torch.backends.cuda.matmul.allow_tf32 = False
S = 4352                      # tokens per sample from docs/nvfp4_training_plan.md
for rank in (64, 96):
    for out_f in (3072, 8192):
        for bs in (1, 4, 8):
            M, K, N = bs * S, rank, out_f
            x = torch.randn(M, K, device="cuda", dtype=torch.float32)
            w = torch.randn(N, K, device="cuda", dtype=torch.float32)
            for i in range(50):
                y = torch.nn.functional.linear(x, w)   # x @ w.T
                torch.cuda.synchronize()
            print("ok", rank, out_f, bs)
```

Run it twice: once with `expandable_segments:False`, once with `True`. If it fails only in
the `True` run, it is the allocator regardless of shape. If a specific `(rank, out, bs)`
tuple fails in both, you have a minimal cuBLAS reproducer to attach to a PyTorch/NVIDIA
report (and to test the 13.8 upgrade against).

### Step 6 — optional cuBLAS upgrade (requires install, no sudo)

Only if Step 5 gives a clean shape reproducer. `nvidia-cublas` (CUDA 13 line) has
13.8.2.12 (2026-10-27) and 13.8.0.4 (2026-09-16); 13.1.1.3 was released 2026-04-08
([PyPI history](https://pypi.org/project/nvidia-cublas/#history)). Ultralytics reported
13.4.0.1 → 13.8.0.4 fixing an sm_120 `nvjet` illegal access (row 8), but NVIDIA's 13.5.1.27
still had the TF32 split-K bug (row 7), so an upgrade is **not guaranteed** to fix the
latter. Do it in a throwaway venv copy, not the training env:

```bash
python -m venv /tmp/cublas-test && /tmp/cublas-test/bin/pip install --no-deps nvidia-cublas==13.8.0.4
# point LD_LIBRARY_PATH at /tmp/cublas-test/lib/python3.*/site-packages/nvidia/cublas/lib
```

### Priority summary

| Priority | Test | Cost | Decides |
|---|---|---|---|
| 1 | `ulimit -n 65536` + dmesg + C++ stack traces | minutes | allocator/fd vs everything else |
| 2 | `expandable_segments:False` on both env vars | minutes | allocator/VMM vs cuBLAS |
| 3 | `CUBLASLT_LOG_LEVEL=5` + `preferred_blas_library` A/B | minutes | whether cuBLASLt is even involved |
| 4 | `NVIDIA_TF32_OVERRIDE=0` / TF32 toggle | minutes | row-7 class bug |
| 5 | standalone shape sweep | ~1 h | minimal reproducer |
| 6 | cuBLAS 13.8 in a scratch venv | ~1 h | library-version fix |

---

## 3. Is this a known cuBLASLt/cuBLAS 13.1 sm_120 bug, or trainer/data?

### Evidence *for* a library/driver bug

* **NVIDIA is actively investigating a shape-specific sm_120 cuBLASLt kernel bug** on an
  RTX 5090 (forum 372787): the TF32 split-K `nvjet_sm120_...` kernel raises a device-side
  "Warp Barrier Arrival Mismatch", which surfaces as an illegal access or hang and is
  sensitive to GEMM shape and to serialization. This is the single best public analogue for
  "rank-64 works, rank-96 doesn't, batch 8 differs".
* **cuBLAS 13.1.0 introduced documented sm_120 GEMM defects** (CUDA 13.2 release notes:
  batched-broadcast OOB [6040940], PDL alpha/beta WAR [CUB-10409]). Our wheel is
  13.1.1.3, i.e. the same minor line.
* **cuBLAS 13.x sm_120 kernels have a history of illegal-access fixes**: nvjet fp16 kernel
  fixed between 13.4.0.1 and 13.8.0.4 (ultralytics #26300); llama.cpp found two more
  distinct sm_120 device-kernel exceptions on the same reporter's 5090.
* **Shape sensitivity is real in this library family**: split-K selection is heuristic and
  changes with M/N/K, which is exactly the pattern (K=64 vs K=96; bs1/4 vs bs8).

### Evidence *against* a cuBLAS bug being the direct cause here

* **The message cannot come from cuBLAS.** `CUDA driver error: unknown error` is produced
  only by the Driver-API check macro; a cuBLAS/cuBLASLt status renders as
  `CUDA error: CUBLAS_STATUS_... when calling ...`. A compute-kernel fault would surface as
  `CUDA error: an illegal memory access was encountered` (runtime API) at the next sync.
  For a GEMM to produce a *driver* error, the failing call would have to be a VMM
  allocation issued from inside the GEMM path — possible for workspace allocation, but
  PyTorch's cuBLAS workspace is allocated once, before the LoRA matmuls.
* **TF32 is off by default** in PyTorch (`allow_tf32=False`, `fp32_precision="none"`), and
  the confirmed sm_120 forum bug is in a TF32 kernel. BF16x9 FP32 emulation is not a
  candidate either: cuBLAS documents it only for compute capability 10.0/10.3, not 12.0
  ([cuBLAS docs, Floating Point Emulation](https://docs.nvidia.com/cuda/cublas/index.html)).
* **PyTorch defaults to cuBLAS, not cuBLASLt**, for BLAS ops
  (`torch.backends.cuda.preferred_blas_library()` → `cublas`; the docstring says "defaults
  to cuBLAS even if both are available"), so the cuBLASLt-specific forum kernel is not
  automatically on our path.
* **The same message has a proven, benign explanation on this exact stack**: WSL2
  `expandable_segments` VMM failure (rows 1–5), including one report with the identical
  torch build and one local incident on this machine. It is intermittent and
  allocation-history dependent, which better fits "rank64 bs1 fails but rank96 bs1 works"
  than a deterministic shape bug would.

### Trainer/data angle

* Nothing in the repo points to bad data: the failures reproduce with synthetic benchmark
  inputs (`moss-lora-bench`, `benchmark.py`), the loss path is fp32 MSE, and the LoRA
  adapters start at zero. A NaN/Inf would produce a loss anomaly, not a driver error.
* The trainer **does** contribute a relevant configuration: it force-enables
  `expandable_segments:True` (`trainer.py:539`, `benchmark.py:32`) on a platform where that
  setting is repeatedly documented as broken. That is a trainer-level cause, not a
  data-level one.
* The rank/batch pattern has *not* been shown to be causal. The padded-rank experiment
  changes allocation sizes and layout as well as the GEMM K, so it does not isolate the
  kernel from the allocator.

### Verdict

**Most likely: allocator/VMM failure on WSL2, triggered or exposed by
`expandable_segments:True` (and possibly fd exhaustion).** Confidence: high for the message
class, medium for this specific run until Step 0/2 are done.

**Second: a shape-dependent cuBLAS/cuBLASLt sm_120 kernel defect.** Confidence: medium as a
class (NVIDIA has an open, confirmed-shape bug on the same GPU), low as the direct cause of
a driver-error string. It becomes the leading theory only if disabling expandable segments
does not help **and** the same shapes fail in a standalone matmul loop.

**Not supported: a data/trainer-math bug.** No evidence found.

---

## 4. Explicit "no match found" statements

* **No public issue** was found for a LoRA **rank-64** (or any adapter rank) q/k/v/o GEMM
  crashing on RTX 5090/sm_120 with `CUDA driver error: unknown error`. GitHub issue search
  over `unslothai/unsloth`, `huggingface/peft`, `huggingface/diffusers`, `bmaltais/kohya_ss`
  returned no such report; the LoRA-on-5090 hits are `no kernel image is available`
  (kohya_ss #3276) or unrelated.
* **No public report** was found of a diffusion-LoRA **batch-size-8** shape-specific
  sm_120 failure. The batch correlation remains local and unexplained externally.
* **No NVIDIA statement** that forum topic 372787 (TF32 split-K `nvjet`) is fixed; the last
  post (2026-07-17) is NVIDIA asking for a reproducer and cuBLASLt logs.
* **No public hit at all** for the WSL kernel message `dxg get_unused_fd_flags failed`
  (GitHub search: 0 results), so the fd-exhaustion explanation currently rests on this
  repo's own `docs/5090.md` observation [UNVERIFIED externally].
* **No documented cuBLAS known issue** matches a plain 2-D fp32 TN GEMM with K=64 on
  sm_120. The 13.1-era entries are batched/broadcast and PDL-only; the forum entry is
  TF32/cuBLASLt. Do not claim a documented exact match.
* **No report ties driver 621.01 to CUDA compute failures.** NVIDIA-forum searches for
  `621.01`/`driver 621` returned only D3D12, OptiX, Vulkan and display issues (e.g. topics
  387036, 387414, 387416, 386503), not CUDA training/GEMM failures.
* **No WSL2 report on a desktop RTX 5090** was found. The WSL2 reports are A4000/A5000
  (#192330), RTX PRO 6000 Blackwell (#182286, #41619, verl #3664) and RTX 6000 Ada
  (Unsloth #3511); treat "same sm_120 + WSL2" as the common factor, not the exact card.
* **`_scaled_mm` / FP8 sm120 reports** (vLLM, Ultralytics) concern FP8/NVFP4 paths this
  trainer does not use; no fp32 LoRA analogue was found.
* **GitHub code search could not be used**: unauthenticated `/search/code` returns
  `401 Requires authentication`. All GitHub evidence above comes from issue/PR search, so
  code comments or READMEs that mention the string are not covered.

---

## 5. Caveats

* Forum 372787 is an RTX 5090 **Laptop** on native Linux with driver 595.71.05 and cuBLAS
  13.5.1.27; our desktop 5090 on WSL2/621.01/13.1.1.3 differs in every version. Same
  architecture, different stack.
* PyTorch #192330's specific root cause is fixed in R615 (616.92); our 621.01 is newer
  [INFERENCE: version ordering]. Its value here is the *mechanism* and the exact error
  string, not proof of an unfixed bug on our driver.
* GenFill/AerialLidarClassifier are native Windows WDDM reports. WSL2 uses GPU-PV/DXG, a
  different paravirtualization path, so treat them as strong analogies, not identical.
* The local `torch 2.14.1+cpu` source inspection proves message provenance
  (`Exceptions.h`) and default BLAS/TF32 behaviour, but not GPU-runtime behaviour.
* Dates/statuses are as observed via the GitHub/NVIDIA APIs on 2026-11-04; issues can
  change after that.

---

## Appendix A — message provenance (local torch 2.14.1)

| Call site | Macro | Message on failure |
|---|---|---|
| cuBLAS / cuBLASLt | `TORCH_CUDABLAS_CHECK` (`ATen/cuda/Exceptions.h:63`) | `CUDA error: <status> when calling ...` |
| CUDA Driver API | `C10_CUDA_DRIVER_CHECK` (`c10/cuda/driver_api.h:11`), `AT_CUDA_DRIVER_CHECK` (`ATen/cuda/Exceptions.h:183`) | `CUDA driver error: <cuGetErrorString>` — or `unknown error` if `cuGetErrorString` itself fails |
| CUDA Runtime API | `C10_CUDA_CHECK` | `CUDA error: <cudaGetErrorString>` |

The caching allocator's expandable-segment path is the main training-time user of the
Driver-API macro (`cuMemCreate`/`cuMemMap`/`cuMemSetAccess`), consistent with Unsloth's
root-cause comment and with PyTorch #192330's `setAccess at c10/cuda/CUDACachingAllocator.cpp`.

## Appendix B — why rank 64 looks like a shape bug (and why that may be a trap)

LoRA forward is `base(x) + ((x @ A^T) @ B^T) * scaling` with `A: [rank, in]`,
`B: [out, rank]` (`moss_lora/lora.py`). The second GEMM is `[M, rank] @ [rank, out]`, i.e.
**K = rank** — 64 vs 96 — and `B` is a transposed view (`F.linear(x, B)`), the TN layout
where split-K/nvjet kernels are selected. That is a genuinely plausible place for a
shape-specific library bug.

But the same K=64 shape exists in the FFN-only run that works, and the padded-rank run
changes both K and every allocation size. Until the allocator is excluded, "rank 64" is
correlated with, not proven to cause, the failure.
