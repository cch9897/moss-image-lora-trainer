# `CUDA driver error: unknown error` on RTX 5090 / WSL2 — diagnosis and fix

Date: 2026-11-04. Diagnosed on the live box (`user@<other-host>`), GPU work bounded to
smoke runs. Production tree untouched during diagnosis; all experiments ran in
`/home/user/diag_repo` (byte-identical copy of `/home/user/MOSS-Lora-Trainer`, verified with
`diff -r`).

**Status (final): the fix is applied and verified in BOTH trees** — remote production
`/home/user/MOSS-Lora-Trainer` (§7.2) and the local git checkout
`/home/user/projects/MOSS/MOSS-Lora-Trainer` (§7.3). The two `moss_lora/{trainer,benchmark}.py`
files are **byte-identical** across both trees (`cmp` clean).

---

## 1. TL;DR

**The failure is not a CUDA kernel, cuBLAS shape, or OOM bug. It is file-descriptor
exhaustion in the WSL2 GPU paravirtualization (`dxg`) path, caused by the trainer
force-enabling `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.**

* `moss_lora/trainer.py:539` (and `benchmark.py:32`) do
  `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")`.
* On WSL2 every expandable-segment VMM mapping (`cuMemCreate`/`cuMemMap`/`cuMemSetAccess`)
  consumes a `dxg` file descriptor. Measured peaks: **42 fds** with the standard allocator,
  **1305 / 1538 / 1780 fds** for rank64 bs1 / bs4 / bs8 with expandable segments.
* The login shell's soft `RLIMIT_NOFILE` is **1024**. When the count crosses it, the VMM
  call fails and PyTorch raises `RuntimeError: CUDA driver error: unknown error` from
  whichever allocation happens next — which is why the traceback landed in three different
  places (forward cast, checkpoint recompute, Adam `_init_group`) on three batch sizes.
* **Fix (verified):** raise `RLIMIT_NOFILE` to 65536 before CUDA init — a ~20-line helper
  called at the top of `main()`, no change to rank, math, or numerics.
  `ulimit -n 65536` in the launcher works equally well. Disabling `expandable_segments` is an
  alternative (it drops fd use to 42).

Verified after the fix: rank64 bs1 ×1, bs4 ×10, bs8 ×10, rank96 bs8 ×10, all clean, with
finite losses, non-zero adapter gradients, and no NaN/Inf in the saved adapters.

---

## 2. Environment and GPU health

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 31.82 GiB, sm_120 (12.0), L2 96 MiB |
| KMD / UMD | driver 621.01, CUDA UMD 13.5 (`nvidia-smi` 620.67.01) |
| Host | WSL2, Debian, kernel 7.0.0-38-generic, boot 2026-11-04 ≈09:53 |
| torch | 2.14.1+cu130, allocator backend `native` |
| IDLE state | 0 % util, 1.2 GiB used (WSL overhead), 39 °C, 11 W |
| ECC / Xid | ECC N/A (consumer); **no Xid, no "fallen off the bus", no NVRM errors** |
| `dmesg` | only benign `misc dxg: dxgk: dxgkio_escape: Ioctl failed: -22` noise |
| Toolkit | `compute-sanitizer`, `cuda-gdb`, `nvcc` present in `/usr/local/cuda-13.3/bin` (not on `PATH`) |

There is no hardware fault, no ECC event, and no OOM: the largest observed reservation was
22.9 GiB of 31.8 GiB.

### 2.1 Error-string provenance (important)

`CUDA driver error: …` is raised **only** by PyTorch's driver-API check macro
(`C10_CUDA_DRIVER_CHECK`, `torch/include/c10/cuda/driver_api.h:24-36`). A cuBLAS/cuBLASLt
failure would instead read `CUDA error: CUBLAS_STATUS_… when calling …`.

The literal text `unknown error` (rather than e.g. `out of memory`) is the branch taken
when `cuGetErrorString()` *itself* fails to stringify the status — i.e. the CUDA context was
already in a degraded state when the check ran. That is the signature of a failed VMM
driver call on WSL2, not of a bad GEMM.

---

## 3. The decisive experiment

Same config (`configs` copied from `runs/smoke_rank64_bs1.yaml`), same data, same seed,
one variable: the soft fd limit.

| # | Command (abridged) | fd limit | expandable | Result |
|---|---|---|---|---|
| A1 | `python diagrun2.py diag_ab_r64bs1_on.yaml` | 1024 | **ON** | **FAIL** — `RuntimeError: CUDA driver error: unknown error` at `trainer.py:475 optimizer.step()` → `torch/optim/adam.py:182 _init_group` |
| A2 | `ulimit -n 65536; python diagrun2.py …` | 65536 | ON | **PASS** — step 1, 12.47 GB, 0.285 s/step |

A1 reproduces the *production* rank64-bs1 traceback byte for byte (same file and line:
`trainer.py:475` → `adam.py:238 step` → `adam.py:182 _init_group`).

A control proved the mode was really active (`torch.cuda.memory_snapshot()` →
`is_expandable: True` in A1/A2, `False` when the env var is set after CUDA init).

> **Methodological trap found:** touching CUDA before `main()` runs (e.g. calling
> `torch.cuda.get_device_properties(0)` at import time) freezes the allocator config, so the
> `setdefault` in `main()` is silently ignored and the process runs with the *standard*
> allocator. Any future A/B on allocator settings must set the env var before CUDA init.
> Verified: with early init the 256 MiB segment reports `is_expandable: False`; with the env
> var set first, `is_expandable: True`.

---

## 4. Mechanism: fd accounting

Peak open fds sampled from `/proc/<pid>/fd` at 100 ms during real training steps
(full model, logs under `/home/user/runs/diag_*`). Peaks were measured with the soft limit
raised to 65536 so each run could reach its true peak; the "outcome" column states what the
same config did at the stock 1024 limit.

| Run | expandable | peak fds | soft limit | outcome |
|---|---|---|---|---|
| rank64 bs1 | ON | **1305** | 1024 | FAIL at Adam state init |
| rank64 bs1 | ON | 1305 | 65536 | PASS |
| rank64 bs4 ×10 | ON | **1538** | 1024 | FAIL (production) / PASS at 65536 |
| rank64 bs8 ×10 | ON | **1780** | 1024 | FAIL (production) / PASS at 65536 |
| rank96 bs1 | ON | 714 | 1024 | PASS (below the limit) |
| rank96 bs8 ×10 | ON | 1240 | 1024 | FAIL (production) / PASS at 65536 |
| rank64 bs8 | **OFF** | **42** | 1024 | PASS |

The correspondence is exact: **every config whose peak exceeds 1024 fds failed; every config
below it passed.**

### 4.1 Why it looked shape- and rank-specific

All previously reported observations fall out of the fd budget without any kernel bug:

* **rank64 vs rank96 at bs1** — rank64 allocates more, smaller VMM segments (1305 fds) than
  rank96 (714 fds, fewer/larger segments). Only rank64 crossed 1024. Nothing about K=64.
* **ffn-only passes, qkvo fails** — fewer adapter tensors → fewer segments → under the limit.
* **bs1 < bs4 < bs8** — 1305 → 1538 → 1780 fds; monotone in batch size.
* **padded rank64 (= rank96 kernel) passed bs1/bs4 but failed bs8** — identical to rank96:
  714/`<1024` at small batch, 1240 at bs8.
* **failure site varies** (forward cast at `lora.py:65/70`, checkpoint-recompute `lora.py:70`,
  Adam `_init_group`) — the error surfaces at the *next allocation* after the fd budget is
  exhausted, and which allocation that is depends on the allocation schedule.
* **"worked in late October"** — the fd budget depends on the launching shell's `ulimit -n`
  and on the allocator's segment layout; a soft limit of 1024 makes the outcome marginal
  rather than deterministic.
* **`CUDA_LAUNCH_BLOCKING=1` appears to help in ad-hoc runs** — do not trust it. Every
  early "passing" run in this investigation had touched CUDA before `main()` (device
  properties probe), which silently disabled expandable segments; that, not launch blocking,
  is what changed. See the trap in §3.
* **`expandable_segments:False` fixes every batch size** — it removes the VMM path entirely
  (42 fds), so the fd budget stops mattering.

---

## 5. compute-sanitizer

`compute-sanitizer` **is** available (`/usr/local/cuda-13.3/bin/compute-sanitizer`, not on
`PATH`). Two runs:

1. **Full model, rank64 bs1, expandable ON, fd limit 65536** (`--tool memcheck
   --launch-timeout 120`): started, loaded the checkpoint, then **did not finish the first
   step within a 600 s cap → killed by `timeout` (rc=124)**. No verdict; memcheck's ~20-50×
   slowdown makes a full diffusion step impractical here. Log:
   `runs/diag_sanitizer_r64bs1.out` (314 bytes, ends inside step 1).
2. **Isolated LoRA branch** (`standalone_lora.py`, real dims, rank64, M=4352,
   q + ffn_up_gate + ffn_down, backward, expandable segments ON, fd limit 65536):

   ```
   ========= COMPUTE-SANITIZER
   PASS q            M= 4352 r= 64 bwd max_abs_err=7.82e-03
   PASS ffn_up_gate  M= 4352 r= 64 bwd max_abs_err=7.82e-03
   PASS ffn_down     M= 4352 r= 64 bwd max_abs_err=7.83e-03
   ========= ERROR SUMMARY: 0 errors
   ```

**0 memory errors.** `racecheck` was not run: the LoRA branch has no shared-memory
programming of its own (all kernels are cuBLAS/elementwise). This confirms the fault is a
driver-API resource failure (fd exhaustion), not an out-of-bounds access or race; memcheck
is not the right instrument for this bug — the `/proc/<pid>/fd` measurement is.

### 5.1 Shape isolation (the blamed shapes are numerically clean)

`standalone_lora.py` reproduces `LoRALinear` exactly (bf16 base `F.linear` + fp32 A/B cast to
the activation dtype, identical to `lora.py:62-70`) using the real model dimensions
(dim 3072, ffn 8192, q/k/v/o `3072×3072`, `ffn_up_gate` `16384×3072`, `ffn_down`
`3072×8192`) and compares against an fp32 reference.

* 36/36 combinations pass: {q,k,v,o,ffn_up_gate,ffn_down} × M=4352 × rank{64,96} ×
  {fwd, bwd, Adam step}.
* Max abs error vs the fp32 reference: **7.8e-3** (fwd/bwd) and **1.5e-2** (after one Adam
  step) — exactly bf16 rounding, and **identical for rank 64 and rank 96**.
* All 36 also pass with `expandable_segments:True` active.

So no `(rank, M, projection)` tuple is miscompiled; the difference between "rank64 fails" and
"rank96 passes" is purely the fd budget of the surrounding allocation pattern (§4.1).

---

## 6. Workarounds tested

| Workaround | bs1 | bs4 | bs8 | Verdict |
|---|---|---|---|---|
| `ulimit -n 65536` (expandable ON) | PASS | PASS ×10 | PASS ×10 | **works**, keeps expandable segments |
| `_raise_fd_limit()` in `main()` (patch, §7) | PASS (1305 fds) | PASS ×10 (1538 fds) | PASS ×10 (1780 fds) | **works from a stock 1024-fd shell** |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` + `PYTORCH_ALLOC_CONF=…:False` | PASS | — | PASS ×10 (42 fds) | works; alternative if fds cannot be raised |
| `CUBLAS_WORKSPACE_CONFIG`, `NVIDIA_TF32_OVERRIDE=0`, `torch.set_float32_matmul_precision` | n/a | n/a | n/a | **not needed** — root cause is not BLAS; not pursued further |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (current default) | FAIL | FAIL | FAIL | **the trigger** |

`expandable_segments:False` peaks at 22.04 GB at bs8 over 10 steps — statistically identical
to the 22.03 GB measured with expandable segments — so there is no VRAM penalty at these
shapes. It does, however, give up the fragmentation protection the setting was added for,
which is why raising the fd limit is the preferred fix.

---

## 7. Recommended minimal fix

Raise the soft fd limit in-process, before any CUDA allocation. Applied to
`moss_lora/trainer.py` and `moss_lora/benchmark.py` in the scratch copy
(`/home/user/diag_repo`, diff in `docs/cuda_crash_diagnosis.patch`):

```python
def _raise_fd_limit() -> None:
    """Raise RLIMIT_NOFILE before CUDA init.

    On WSL2 each ``expandable_segments`` VMM mapping (cuMemCreate/cuMemMap/
    cuMemSetAccess) consumes one dxg file descriptor: a bs8 step peaks at
    ~1700 fds. With the default soft limit (1024) the mapping call fails with
    CUDA_ERROR_UNKNOWN and surfaces as ``CUDA driver error: unknown error``
    from whichever allocation happens next.
    """
    try:
        import resource
    except ImportError:  # non-POSIX platform
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
        if soft != resource.RLIM_INFINITY and soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass


def main(argv: list[str] | None = None) -> None:
    _raise_fd_limit()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ...
```

Belt-and-braces for the shell launchers (`runs/run_train_v4_5ep_r64.sh`,
`run_train_v4_5ep_r128.sh`), which also covers any code path that never reaches `main()`:

```bash
ulimit -n 65536   # WSL2: expandable_segments maps one dxg fd per VMM segment
```

**Capacity preserved:** the patch touches neither rank, alpha, target list, dtype, nor the
LoRA math. rank64 all-targets keeps 148.6 M trainable parameters (42 layers × 3,538,944 =
252 A/B tensors), and the adapter shapes are unchanged: `lora_A [64, in]`, `lora_B [out, 64]`.
No effective-rank padding or kernel substitution is required.

### 7.1 Verification of the fix (patched code, shell limit left at 1024)

| Run | rc | peak fds | loss (step 10) | peak VRAM | sec/step | adapter |
|---|---|---|---|---|---|---|
| rank64 bs1 ×1 | 0 | 1305 | 0.296510 | 12.47 GB | 0.287 | OK |
| rank64 bs4 ×10 | 0 | 1538 | 0.360705 | 17.29 GB | 5.30 | OK — 504 tensors, 0 NaN/Inf, B non-zero |
| rank64 bs8 ×10 | 0 | 1780 | 0.384049 | 22.03 GB | 10.43 | OK — 504 tensors, 0 NaN/Inf, B non-zero |
| rank96 bs8 ×10 | 0 | 1240 | 0.384053 | 22.86 GB | 10.55 | OK |

`peak fds 1780 > 1024` proves the in-process raise took effect — a process still capped at
1024 could not hold 1780 descriptors (the `/proc/<pid>/limits` line sampled at fork time in
those runs still showed 1024 because it was read before `main()` ran; the production runs in
§7.2 sample it later and show 65536). The adapter check confirms 252 `lora_A` + 252
`lora_B` tensors, `nan_or_inf=0`, `B all-zero: 0/252`, `A all-zero: 0/252`, i.e. gradients
flowed and nothing degenerated.

Footnote: the **rank96 bs8** row above is a rank-96 sanity check measured with the external
`ulimit -n 65536` arm, not an in-code-patch run; the in-code patch itself was verified on the
rank64 bs1/bs4/bs8 rows (and again on production in §7.2).

### 7.2 Production cutover (applied 2026-11-04)

* `docs/cuda_crash_diagnosis.patch` applied to `/home/user/MOSS-Lora-Trainer` with
  `patch -p1`: both files applied **with no offset or fuzz**. Originals backed up in
  `/home/user/runs/prod_fd_backup/`. Production `trainer.py`/`benchmark.py` are now
  byte-identical to the verified scratch copies (`diff` clean); no other file changed.
* Static/unit checks: `ruff` and `pytest` are **not installed** (venv or system) and
  installing was out of scope, so the CPU-only suites were executed through a minimal
  pytest-compatible shim (`/tmp/minipytest.py`, supports `raises`/`approx`/`parametrize`/
  `tmp_path`): `tests.test_config`, `tests.test_lora`, `tests.test_latent` → **29 passed,
  0 failed**; `tests.test_sample_merge` → 4 passed, 4 skipped (need `monkeypatch`/model
  fixtures). `test_model`/`test_trainer_smoke`/`test_vlm` were not run (model fixtures, GPU).
  `py_compile` OK; `python -m moss_lora.trainer --help` OK.
* Direct check of the new helper: `(1024, 1048576) → (65536, 1048576)`, hard limit
  unchanged, idempotent on repeat calls.
* Production smokes, **stock 1024-fd shell, no allocator env overrides**, run through
  `venv/bin/python -m moss_lora.trainer` (the real entrypoint):

| Run | rc | peak fds | soft NOFILE in `/proc/<pid>/limits` | loss | peak VRAM | wall |
|---|---|---|---|---|---|---|
| rank64 bs1, 1 step | 0 | 1305 | **65536** | 0.296510 | 12.47 GB | 12 s |
| rank64 bs4, 1 step | 0 | 1473 | **65536** | 0.456034 | 15.73 GB | 15 s |
| rank64 bs8, 1 step | 0 | 1716 | **65536** | 0.379640 | 20.51 GB | 21 s |

* Adapters (all three runs): 504 tensors = 252 `lora_A` + 252 `lora_B`, `nan_or_inf=0`,
  `B all-zero: 0/252`, `r=64`, `alpha=64`, targets
  `q,k,v,o,ffn_up_gate,ffn_down`.
* The losses reproduce the pre-patch deterministic values exactly (0.296510 / 0.456034 /
  0.379640), so the fix changes no numerics.
* **Residual caveat:** the raise is bounded by the hard limit (1048576 here; `min(65536,
  hard)` handles lower caps). Any launcher that bypasses `main()` — e.g. a script that
  imports the model itself — should still carry `ulimit -n 65536`.

### 7.3 Local git checkout sync (applied 2026-11-04)

The local git checkout `/home/user/projects/MOSS/MOSS-Lora-Trainer` was still unpatched
after the production cutover; `docs/cuda_crash_diagnosis.patch` has now been applied there
as well (working tree only, **not committed**):

* `patch -p1 --dry-run` clean, then `patch -p1` → both files applied with **no offset, no
  fuzz, no `.rej`/`.orig`**. The patch delta is exactly **+23/−0 per file**: the 20-line
  `_raise_fd_limit()` helper, two separating blank lines, and the single
  `_raise_fd_limit()` call as the first statement of `main()`. No other local file was
  touched; rank, alpha, target list, dtype and LoRA math are unchanged.
* Insertion points are identical to production: `trainer.py:538` (def) / `trainer.py:561`
  (call), `benchmark.py:31` (def) / `benchmark.py:54` (call).
* **md5 — local vs remote, identical to the values reported for production:**

| File | Local (patched) | Remote production (§7.2) | Match |
|---|---|---|---|
| `moss_lora/trainer.py` | `a68bd5b1ed69fcb5c69cd84895917485` | `a68bd5b1ed69fcb5c69cd84895917485` | yes |
| `moss_lora/benchmark.py` | `d9b8ac8d24ebca24c39ebf782508531a` | `d9b8ac8d24ebca24c39ebf782508531a` | yes |

  `cmp` against the remote files (read-only `ssh` transfer) is byte-clean for both, so the
  local checkout is a byte-exact copy of the verified production fix.
* Local static/unit checks (CPU-only; no GPU used, NVFP4 benchmark running):
  `python -m py_compile moss_lora/trainer.py moss_lora/benchmark.py` OK;
  `ruff check` → *All checks passed!*;
  `.venv` pytest on `tests/test_trainer_smoke.py tests/test_lora.py tests/test_config.py
  tests/test_export_comfy.py tests/test_checkpoint_retention.py` → **46 passed, 1 skipped,
  0 failed**.
* Direct helper check on this host: with the process soft limit lowered to 1024,
  `resource.getrlimit(RLIMIT_NOFILE)` went `(1024, 524288)` → `(65536, 524288)` — soft limit
  raised, **hard limit unchanged** — and a second call (via `benchmark._raise_fd_limit()`)
  was a no-op, i.e. idempotent. Both modules import and expose the helper.
* Note on `git diff --stat`: `moss_lora/trainer.py` shows `94 insertions(+), 3 deletions(-)`
  because that file already carried 71 uncommitted insertions / 3 deletions unrelated to this
  task **before** the patch was applied (71 + 23 = 94). The fd-limit patch itself is
  `+23/−0` in each file; `benchmark.py` was clean vs `HEAD` and its entire local diff is the
  fix (`23 insertions(+)`). Nothing was committed.

---

## 8. bs8 classification

**Not OOM, and not a separate shape bug — same fd/VMM failure.**

* rank64 bs8 peak: **22.03 GB allocated / 22.9 GB reserved** of 31.82 GB. 9 GB headroom.
* rank96 bs8 peak: 22.86 GB. Also fine.
* Three concurrent rank64 processes (bs1+bs4+bs8 ≈ 48 GB aggregate, i.e. deliberate
  oversubscription) **all completed** — WSL2 paged to host memory without a driver error, so
  VRAM pressure alone does not produce this error.
* An OOM in PyTorch raises `torch.OutOfMemoryError` with an allocator report; the observed
  error is a bare driver-API failure with no allocator report, and the process was 9 GB
  short of the limit.
* The validation VAE is **not** loaded: `trainer.py:429` gates it on
  `logging.sample_every > 0`, and every diagnostic config uses `sample_every: 0`, so the
  22.03 GB peak is training-only.
* bs8 needs 1780 fds; the limit is 1024. That is the whole story.

---

## 9. Exact reproduction commands

```bash
# scratch copy (production tree untouched)
rsync -a --exclude venv --exclude __pycache__ /home/user/MOSS-Lora-Trainer/ /home/user/diag_repo/

# --- reproduce the failure (expandable segments ON, stock 1024-fd shell) ---
cd /home/user/diag_repo
/home/user/MOSS-Lora-Trainer/venv/bin/python diagrun2.py /home/user/runs/diag_ab_r64bs1_on.yaml
#   -> RuntimeError: CUDA driver error: unknown error
#      trainer.py:475 optimizer.step -> torch/optim/adam.py:182 _init_group

# --- the one-variable proof: same run, higher fd limit ---
( ulimit -n 65536; /home/user/MOSS-Lora-Trainer/venv/bin/python diagrun2.py /home/user/runs/diag_ab_r64bs1_on.yaml )
#   -> [step 1] loss=0.296510 ... peak_vram_gb=12.47

# --- measure the fd count that causes it ---
#   (sampler: ls /proc/<pid>/fd | wc -l every 100 ms while the trainer runs)
#   rank64 bs1/4/8 -> 1305 / 1538 / 1780   |   expandable OFF -> 42   |   rank96 bs1 -> 714

# --- prove the allocator mode is actually active (env var + a real allocation) ---
cd /home/user/MOSS-Lora-Trainer
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True venv/bin/python -c \
  'import torch; x = torch.empty(4 << 20, device="cuda"); print(any(s.get("is_expandable") for s in torch.cuda.memory_snapshot()))'
#   -> True with the env var set first, False if it is set after any torch.cuda.* call

# --- verify the patch (uses the scratch tree, shell limit stays 1024) ---
cd /home/user/diag_repo && python apply_fd_patch.py
/home/user/MOSS-Lora-Trainer/venv/bin/python diagrun2.py /home/user/runs/diag_v_r64_bs8.yaml   # 10 steps, passes

# --- alternative workaround (no code change) ---
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False PYTORCH_ALLOC_CONF=expandable_segments:False \
  /home/user/MOSS-Lora-Trainer/venv/bin/python diagrun2.py /home/user/runs/diag_v_r64_bs8.yaml

# --- adapter sanity ---
python adapter_check.py /home/user/runs/diag_v_r64_bs8/adapter_model.safetensors
```

Note: `ulimit -n 65536` and the env-var overrides must be in effect *before* the Python
process initializes CUDA; setting `PYTORCH_CUDA_ALLOC_CONF` after any `torch.cuda.*` call
has no effect (see §3).

---

## 10. Confidence and residual unknowns

| Claim | Status |
|---|---|
| Message originates in `C10_CUDA_DRIVER_CHECK` (driver API), not cuBLAS | **Proven** (torch headers + exact reproduction) |
| expandable segments + WSL2 VMM is the failing path | **Proven** (`is_expandable` flag; ON fails / OFF passes on identical config) |
| fd exhaustion is the mechanism | **Proven by correlation** — peak fds 1305/1538/1780 vs limit 1024, and raising the limit fixes it with no other change. `[INFERENCE]` the exact syscall returning 999 is `cuMemMap`/`cuMemSetAccess`; the failing call could not be symbolized because `addr2line` failed (`pipe() failed`) and the driver could not stringify the error. |
| No kernel/cuBLAS defect involved | **Strongly supported** — `compute-sanitizer --tool memcheck` on the LoRA branch: `ERROR SUMMARY: 0 errors`; standalone sweep: 36/36 shapes numerically correct at rank 64 and 96; disabling expandable segments (which changes no kernel) fixes it; no shape reproduces it under the standard allocator |
| bs8 is not OOM | **Proven** — 22.03 GB of 31.82 GB, three-way oversubscription survives |
| Why it worked in late October | **Unresolved** `[INFERENCE]`: the launcher's `ulimit -n` and the segment schedule decide the margin; at 1024 fds the outcome is marginal, not deterministic. Worth checking the historical launcher shell for a `ulimit -n` line. |

Sanitizer note: memcheck on the **full model** could not complete within a 600 s cap
(rc=124, no verdict) — a full diffusion step under memcheck's ~20-50× slowdown is
impractical here. The isolated LoRA branch is clean, which is the relevant signal since the
failure is a driver-resource exhaustion, not a memory-safety fault. `racecheck` is not
applicable (no custom shared-memory kernels in the LoRA path).

---

## 11. Artifacts

All under `/home/user/runs/` (existing smoke dirs preserved untouched):

* `diag_ab_r64bs1_on.yaml`, `diag_ab_r64bs1_off.yaml` — the A/B configs
* `diag_ab_r64bs1_on/`, `diag_ab_r64bs1_off/` — A/B outputs (ON run has no checkpoint: it failed)
* `diag_v_r64_bs4/`, `diag_v_r64_bs8/`, `diag_v_r96_bs8/` — 10-step verification runs + adapters
* `diag_v_r96_bs1/` — rank96 bs1 (fd baseline)
* `diag_p_r64_bs1/`, `diag_p_r64_bs4/`, `diag_p_r64_bs8/` — patched-code runs (stock 1024-fd shell)
* `diag_x_r64_bs8_OFF.out`, `diag_x_r96_bs1.out`, `diag_v_*.out` — run logs
* `prod_fd_bs1/`, `prod_fd_bs4/`, `prod_fd_bs8/` + `prod_fd_bs*.yaml` / `.runlog` — the
  production-entrypoint smokes (§7.2), with saved adapters and checkpoints
* `prod_fd_backup/` — pre-patch copies of `moss_lora/trainer.py` and `moss_lora/benchmark.py`
  (md5 `b580fea5…` / `a86e96f4…`), kept for rollback
* `diag_sanitizer_r64bs1.out` — full-model memcheck attempt that hit the 600 s cap (no verdict)
* Patched scratch tree `/home/user/diag_repo` with `apply_fd_patch.py` (the patch),
  `diagrun2.py` (CUDA-touch-free launcher), `standalone_lora.py` (shape sweep),
  `adapter_check.py` (adapter sanity)
* `docs/cuda_crash_diagnosis.patch` — the change against production `moss_lora/{trainer,benchmark}.py`
