# clean30 v4 r64 bs8 5-epoch — live status

Snapshot: **2026-11-04 17:06:51 +08** (remote clock, `user@<other-host>`).
Run: `clean30_v4_5ep_r64_bs8`, config `configs/train_5090_v4_5ep_r64_bs8.yaml`.
Read-only inspection; nothing was modified, restarted, stopped or resumed.

**Verdict: training is alive and mechanically healthy** — no CUDA/OOM/NaN errors, steady
10.45 s/step, stable VRAM, stable fd count, loss finite but **flat** (no measurable
improvement in 1,720 steps). No checkpoint exists yet, so the first restartable point is
step 2,614 (~19:43 today).

## Current status

| item | value | evidence |
|---|---|---|
| trainer | **PID 52606**, alive, PPID 1, `SNsl`, 100 % CPU, RSS 3,424 MiB | `ps -o pid,ppid,etime,pcpu,rss,stat,cmd -p 52606` |
| dataloader workers | 53062–53065 (children of 52606, `num_workers: 4`), ~1.5 GiB each | `ps --ppid 52606` |
| dashboard | **PID 52501**, `python -m moss_lora.dashboard --root /home/user/runs --host 0.0.0.0 --port 7860`, LISTEN 0.0.0.0:7860 | `ss -ltnp`, `ps` |
| step | **1,720 / 13,070** (13.2 % overall), epoch **1/5** at 65.8 % (1,720/2,614) | `tail -1 metrics.csv`, `/api/run` |
| latest loss | **0.4024** (step 1,720, σ=0.569, grad_norm 0.0344) | metrics row |
| loss, whole run | mean 0.4076, median 0.4006, sd 0.0528, range 0.3046–0.5420 | 86 rows, steps 20–1,720 |
| speed | **10.45 s/step** (recent), 10.39 s/step mean excluding the contaminated first row, 10.51 s/step wall-clock including startup | metrics `sec_per_step`, elapsed |
| VRAM | torch peak allocated **22.51 GiB**; `nvidia-smi` device **25,431 / 32,607 MiB (78 %)**; 7.2 GiB headroom | metrics + `nvidia-smi` |
| GPU | 99 % util, 542 W, 67.4 °C, SM 2,812 MHz, mem 13,801 MHz; only PID 52606 | `nvidia-smi --query-gpu` |
| ETA | **≈ 33.1 h** → 2026-11-06 ≈ 02:10 +08; ≈ **02:25** including 4 validations + 9 checkpoint writes + final export | 11,350 steps × 10.49 s |
| checkpoints | **none yet** — `checkpoints/` does not exist; first save is the epoch-end at 2,614 (~19:43) | `ls` (ENOENT) |
| samples | `samples/step_000000/` — 8 PNGs (8.8 MiB), written 12:06–12:08; next at 3,000 | `ls -la samples/step_000000` |
| dashboard HTTP | **200**, 58,607 bytes at `http://<other-host>:7860/` | `curl -w '%{http_code}'` |
| disk | `/home/user` 1,007 G, 633 G used, **324 G avail (67 %)**; projected checkpoints ≈ 15.5 GiB | `df -h /home/user` |

Run age at snapshot: 5 h 01 m (started 12:05:41, `.pid` mtime; `ps` ELAPSED 05:01:09).
Wall-clock rate including the step-0 validation: 1,720 / 18,070 s = 10.51 s/step.

## Loss trend

`logging.log_every: 20`, so each row is a 20-step mean. Rows: **86**, steps **20 → 1,720**;
steps 2,000/2,614/3,000/5,000 are not reached yet.

| step | loss | grad_norm | lr | sigma | s/step |
|---|---|---|---|---|---|
| 20 | 0.3998 | 0.0255 | 1.575e-05 | 0.6229 | 19.516\* |
| 40 | 0.4248 | 0.0152 | 3.075e-05 | 0.4931 | 10.603 |
| 60 | 0.5209 | 0.0214 | 4.575e-05 | 0.5756 | 10.573 |
| 100 | 0.3586 | 0.0127 | 7.575e-05 | 0.5799 | 10.401 |
| 500 | 0.3928 | 0.0359 | 1.498e-04 | 0.5561 | 10.463 |
| 1,000 | 0.4257 | 0.1347 | 1.486e-04 | 0.6979 | 10.194 |
| 1,500 | 0.4303 | 0.0435 | 1.463e-04 | 0.5036 | 10.357 |
| 1,720 (current) | 0.4024 | 0.0344 | 1.449e-04 | 0.5692 | 10.452 |
| 2,000 / 2,614 / 3,000 / 5,000 | — | — | — | — | not reached |

\* step 20 averages the step-0 baseline validation (8 images × 30 steps) plus first-step
CUDA warmup into 20 steps; steady state starts at step 40.

**Recent-window stats** (only 86 rows exist, so last-100 = last-500 = full history):

| window | n | mean | median | sd | min | max |
|---|---|---|---|---|---|---|
| last 100 rows | 86 (all) | 0.40758 | 0.40064 | 0.05278 | 0.30461 (step 220) | 0.54196 (step 940) |
| last 500 rows | 86 (all) | 0.40758 | 0.40064 | 0.05278 | 0.30461 | 0.54196 |
| first 10 rows | 10 | 0.40750 | — | 0.0572 | — | — |
| last 10 rows | 10 | 0.42167 | — | 0.0548 | — | — |

200-step bucket means (loss): 0.4075, 0.4153, 0.3877, 0.4153, 0.4424, 0.3847, 0.4065,
0.3907, 0.4252 — noise around a 0.40–0.44 band, no trend.

Trend test (OLS `loss ~ 1 + sigma + step/1000`): step coefficient **−0.0017 per 1,000
steps**, t = −0.14, R² = 0.004; sigma coefficient −0.043, t = −0.6. **Conclusion: loss is
statistically flat**, not decreasing and not diverging. Loss variance is dominated by
random σ (logit-normal timesteps) and caption-variant mix, not by a drift term.

Representative spikes (all within 4 sd of the mean, immediately followed by recovery):
0.5209 (step 60), 0.5400 (340), 0.5225 (820), 0.5420 (940), 0.5368 (1540). No step is a
divergence: no NaN/Inf, no sustained rise, grad_norm never approached the clip limit.

## Stability checks

- **Errors:** `grep -nEi "CUDA driver error|CUDA_ERROR|OOM|out of memory|Traceback|NaN|inf\b|Error"` over the 83-line trainer log returns **nothing**. No warnings either.
- **Grad norm:** median 0.0316, p90 0.0763, p95 0.1347, p99 0.3445, max **0.3701** (step 600). All values are ≤ 0.37 vs `max_grad_norm: 1.0`, so clipping never engaged; spikes at 600 (0.370), 700 (0.264), 1,000 (0.135), 1,260 (0.206), 1,300 (0.116) are 3–10× the median but an order of magnitude below the clip — normal for LoRA.
- **LR:** warmup 200 steps 1.5e-5 → 1.5e-4 (linear), then cosine; observed 1.449e-4 at step 1,720 matches `1.5e-4·0.5·(1+cos(π·(1720−200)/(13070−200)))` = 1.4489e-4 to 4 digits. Schedule is on spec and reaches ~0 at 13,070.
- **Speed:** 10.076–10.603 s/step over rows 2–86 (mean 10.385, sd 0.096); 200-step buckets 10.34 → 10.45 with no drift; no degradation. The 19.5 s/step at step 20 is the one-time validation/warmup artefact described above.
- **VRAM:** torch `max_memory_allocated` 22.26 GiB at step 20 → 22.47 by step 500 → **22.51 GiB from step 1,260 onward (plateau)**; `nvidia-smi` 25,431 MiB constant (torch peak + ~2.3 GiB reserved/context). 7.2 GiB below the 32,607 MiB device limit. No growth trend.
- **fd / NOFILE:** `/proc/52606/limits` → `Max open files 65536`; current **1,816 fds (2.8 % of limit)**, of which **1,776 are `anon_inode:dxgresource`** (WSL VMM mappings; the code docstring documents a bs8 peak of ~1,700). Sampled every ~55 s for **10 minutes (≈57 steps): 1,816 → 1,816, RSS 3,424 → 3,424 MiB — zero growth**. Workers: 896–908 fds each (866 dxg), inherited handles. No leak in steady state.
- **GPU power/thermal:** `SW Power Cap: Active` (542 W vs 575 W TDP) is expected at 99 % util; `HW Thermal Slowdown: Not Active`, 67.4 °C, P1. No throttling risk.
- **Variant mix** (13,120 samples, rows 1–86): short 23.1 %, medium 23.2 %, long 22.7 %, original 17.7 %, **layout 13.2 %** (layout is the least frequent variant; stable across the run, not drifting).

## Checkpoints — inventory and retention

**Actual inventory at snapshot: there is no `checkpoints/` directory.**

```
$ ls /home/user/runs/clean30_v4_5ep_r64_bs8/
metrics.csv  samples
$ ls /home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints
ls: cannot access '.../checkpoints': No such file or directory
```

Nothing has been written yet because the first save is the epoch-end at step 2,614
(`save_every: 3000` → first regular save at 3,000). The launch audit's expected layout is
therefore **not yet observable on disk**; what follows is the verified code path and a
measured size reference, not an observation of this run's directories.

Policy implementation (remote `moss_lora/trainer.py` md5 `da955c64ac7eb266b1044de1b8a7a94e`,
identical to local; `prune_checkpoints`, lines 232–253):

```python
candidates = [path for path in root.glob("step_*")
              if path.is_dir() and (path / TRAINER_STATE_NAME).is_file()]
candidates.sort(key=_checkpoint_step)
epoch_end = {path for path in candidates if _is_epoch_end_checkpoint(path)}
regular = [path for path in candidates if path not in epoch_end]
keep = set(regular[-keep_last:]) | epoch_end
```

- **Full state at every save, including epoch ends:** `save_checkpoint` always writes
  `adapter_model.safetensors`, `optimizer.pt`, `lr_scheduler.pt`, `rng.pt`,
  `trainer_state.json` (step/epoch/batch_index/epoch_end/version/full config);
  `adapter_ema.safetensors` only when EMA is on (this run: `ema_decay: 0`, off).
  The epoch-end call at step 2,614 uses the same function with `epoch_end=True`, so it is
  a full-state checkpoint by construction.
- **Epoch-end dirs are outside the 5 regular slots:** they are partitioned out before
  `regular[-keep_last:]`, then unioned back — an epoch-end checkpoint can never displace a
  regular one, and is never pruned regardless of step.
- **Policy is unit-tested:** `tests/test_checkpoint_retention.py` covers newest-N pruning,
  epoch-end survival, and `test_epoch_end_checkpoint_does_not_consume_a_regular_slot`
  (launch audit lists 9 passing tests).
- **Pruning will not fire in this run:** only 4 regular saves exist (3,000 / 6,000 / 9,000 /
  12,000) for `keep_last_checkpoints: 5`, so "pruned to newest 5" is not exercised here;
  all 4 survive. If the run were longer, the oldest regular dir would be deleted while
  every epoch-end dir stays.
- **Coexistence at 2,614/3,000:** at the 3,000 save, candidates are
  `{002614 (epoch_end), 003000 (regular)}` → `keep = {003000} ∪ {002614}` → both retained.
  This cannot be listed from disk until ~20:51 today.

Expected final inventory (9 dirs, all full state):

```
checkpoints/step_002614  (epoch_end)      checkpoints/step_007842  (epoch_end)
checkpoints/step_003000  (regular)        checkpoints/step_009000  (regular)
checkpoints/step_005228  (epoch_end)      checkpoints/step_010456  (epoch_end)
checkpoints/step_006000  (regular)        checkpoints/step_012000  (regular)
checkpoints/step_013070  (epoch_end, final)
```

Projected size: measured reference `prod_fd_bs8/checkpoints/step_000001` (same r64/alpha64,
same six targets, same base model) = `adapter_model.safetensors` 594,592,688 B +
`optimizer.pt` 1,189,522,965 B + scheduler/RNG/state/config ≈ 18.5 KB = **1.662 GiB per
checkpoint**. 9 × 1.662 GiB = **14.95 GiB**, plus the final `comfyui_lora.safetensors`
(~0.55 GiB) written into the run root → **≈ 15.5 GiB total (4.8 % of the 324 G free)**.

Verification commands to run once 2,614 lands (exact, copy-paste):

```bash
ssh user@<other-host> '
  ls -la /home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints/
  for d in /home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints/step_*; do
    echo "== $d $(du -sh $d | cut -f1)"
    ls -l $d
    python3 -c "import json,sys;d=json.load(open(\"$d/trainer_state.json\"));print({k:d[k] for k in (\"step\",\"epoch\",\"batch_index\",\"epoch_end\")})"
  done'
```

At 2,614 expect `epoch_end: true, epoch: 1, batch_index: 0, step: 2614`; after the 3,000
save expect both `step_002614` and `step_003000` present, the latter with
`epoch_end: false, epoch: 1, batch_index: 386`.

## Dashboard

- Reachable: `http://<other-host>:7860/` → **HTTP 200**, 58,607 bytes; bound
  `0.0.0.0:7860` by PID 52501.
- `GET /api/runs` lists this run as `status: running, step: 1720, latest_loss: 0.402433,
  sample_count: 8, checkpoint_count: 0`.
- `GET /api/run?path=clean30_v4_5ep_r64_bs8` →

```json
{"status":"running","step":1720,"max_steps":null,"epoch":null,
 "checkpoint_count":0,"eta_seconds":null,"stale_seconds":10.4}
```

- **Missing data is expected, not corruption:** `max_steps`, `epoch`, `batch_index`, the
  config block and ETA are all null until the first `trainer_state.json` exists
  (`dashboard.py` `_state()` reads it from the run root or newest checkpoint). They will
  populate after 2,614. ETA is therefore computed externally in this report.
- `stale_seconds` oscillates 0–210 s because metrics are appended every 20 steps
  (≈ 210 s); that is the normal refresh granularity, not a stalled run. Requests without
  `?path=` return HTTP 400 by design.
- Dashboard process is healthy: 5 h uptime, 51 MiB RSS, no errors in
  `/home/user/runs/dashboard_v4.log` (only access-log lines).

## Risks / observations

1. **No recovery point yet.** First checkpoint is 2,614 (~19:43, 2 h 36 m away). A crash
   before then loses the full 5 h of progress. Nothing to fix now; consider `save_every`
   < 3,000 for the next run.
2. **Loss is flat, not improving.** 1,720 steps (0.66 epoch, 13,760 samples) with
   0.4076 mean and no significant slope (t = −0.14). No divergence and no instability, but
   no evidence of learning yet either. The first meaningful checkpoints are the epoch-end
   at 2,614 (can eval the adapter) and the sample comparison at 3,000 vs the step-0
   baseline; judge quality there, not from train loss.
3. **VRAM headroom is adequate but not huge** (78 % of device). Allocated peak has been
   flat since step 1,260; the 3,000-step validation reuses the already-loaded VAE, so a
   jump is unlikely — still check `peak_vram_gb` in the step-3,000 row and watch for
   validation-time OOM.
4. **WSL dxg fds are near their documented operating level** (1,776 vs ~1,700 expected)
   but the limit is 65,536 (36× headroom) and the count was bit-identical for 10 minutes.
   The launcher's `ulimit -n 65536` is doing its job; without it the default 1,024 would
   fail with `CUDA driver error: unknown error`.
5. **Swap-in of validation/checkpoint I/O will perturb the next few rows:** the 20-step
   row after each save/validation will show an inflated `sec_per_step` (spread over 20
   steps). Do not read those rows as a slowdown.
6. **GPU is power-capped, not thermally limited** (542 W, 67 °C, 0 µs HW thermal
   slowdown). Expected for a 5090 at 99 % util; no action.
7. **Dashboard misleading nulls** until 2,614 — use `/api/runs` step or the CSV for
   progress; ETA must be derived externally for now.

### ETA breakdown (rate 10.49 s/step from step 1,720 at 17:06:51)

| event | step | ETA (+08) |
|---|---|---|
| epoch 1 end checkpoint | 2,614 | 2026-11-04 19:43 |
| regular checkpoint + samples | 3,000 | 2026-11-04 20:51 |
| epoch 2 end | 5,228 | 2026-11-05 03:21 |
| regular | 6,000 | 2026-11-05 05:36 |
| epoch 3 end | 7,842 | 2026-11-05 10:58 |
| regular | 9,000 | 2026-11-05 14:21 |
| epoch 4 end | 10,456 | 2026-11-05 18:36 |
| regular | 12,000 | 2026-11-05 23:06 |
| final / epoch 5 end | 13,070 | **2026-11-06 02:13** (≈02:25 with save+sample overhead) |

## Monitoring commands

```bash
# progress / latest log line
ssh user@<other-host> 'tail -3 /home/user/runs/clean30_v4_5ep_r64_bs8.log'
ssh user@<other-host> 'tail -1 /home/user/runs/clean30_v4_5ep_r64_bs8/metrics.csv'
ssh user@<other-host> 'wc -l < /home/user/runs/clean30_v4_5ep_r64_bs8/metrics.csv'

# process + fd headroom
ssh user@<other-host> 'pid=$(cat /home/user/runs/clean30_v4_5ep_r64_bs8.pid); ps -o pid,etime,pcpu,rss,stat,cmd -p $pid; grep "Max open files" /proc/$pid/limits; ls /proc/$pid/fd | wc -l'

# errors (empty = good)
ssh user@<other-host> 'grep -nEi "CUDA driver error|CUDA_ERROR|OOM|out of memory|Traceback|NaN|inf\b|Error" /home/user/runs/clean30_v4_5ep_r64_bs8.log'

# GPU
ssh user@<other-host> '/usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm --format=csv,noheader; /usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'

# checkpoints / retention
ssh user@<other-host> 'ls -la /home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints/ 2>&1; du -sh /home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints/* 2>/dev/null'

# samples / disk
ssh user@<other-host> 'ls -la /home/user/runs/clean30_v4_5ep_r64_bs8/samples/; df -h /home/user'

# dashboard
curl -s "http://<other-host>:7860/api/run?path=clean30_v4_5ep_r64_bs8"
curl -s http://<other-host>:7860/api/runs
```

## Key raw outputs

```
$ pgrep -af "moss_lora"
52501 venv/bin/python -m moss_lora.dashboard --root /home/user/runs --host 0.0.0.0 --port 7860
52606 venv/bin/python -m moss_lora.trainer --config configs/train_5090_v4_5ep_r64_bs8.yaml
53062..53065 venv/bin/python -m moss_lora.trainer --config configs/train_5090_v4_5ep_r64_bs8.yaml

$ ps -o pid,ppid,etime,pcpu,rss,stat -p 52606,52501
  52606       1    05:01:09  100 3423608 SNsl
  52501       1    04:52:00  0.1   51772 SNsl

$ grep "Max open files" /proc/52606/limits
Max open files            65536                65536                files

$ ls -l /proc/52606/fd | awk '{print $11}' | sed -E 's/[0-9]+/N/g' | sort | uniq -c | sort -rn
   1776 anon_inode:dxgresource
     27 pipe:[N]
      6 anon_inode:[eventfd]
      ...  (total 1816)

fd samples (10 min, ~55 s apart): 1816, 1816, 1816, 1816, 1816, 1816, 1816, 1816, 1816, 1816, 1816
dxgresource each time: 1776; RSS 3,424,232 kB → 3,424,232 kB

$ /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv
NVIDIA GeForce RTX 5090, 99 %, 25431 MiB, 32607 MiB, 542.08 W, 67.40, 2812 MHz, 13801 MHz

$ /usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
52606, [Not Found], [N/A]          # WSL does not expose per-process VRAM

$ nvidia-smi -q -d PERFORMANCE | grep -A6 "Clocks Event Reasons"
        Idle                                           : Not Active
        Applications Clocks Setting                    : Not Active
        SW Power Cap                                   : Active
        HW Slowdown                                    : Not Active
            HW Thermal Slowdown                        : Not Active

$ tail -2 /home/user/runs/clean30_v4_5ep_r64_bs8.log
[step 1700] loss=0.499192 lr=1.450e-04 sigma=0.4920 grad_norm=0.0299 sec_per_step=10.368 peak_vram_gb=22.51 variant_histogram={"layout":19,"long":45,"medium":32,"original":28,"short":36}
[step 1720] loss=0.402433 lr=1.449e-04 sigma=0.5692 grad_norm=0.0344 sec_per_step=10.452 peak_vram_gb=22.51 variant_histogram={"layout":21,"long":39,"medium":36,"original":36,"short":28}

$ ls -la /home/user/runs/clean30_v4_5ep_r64_bs8/samples/step_000000
-rw-rw-r-- 1 user user 1607562 Nov  4 12:06 item_00000.png
-rw-rw-r-- 1 user user 1337960 Nov  4 12:06 item_03000.png
-rw-rw-r-- 1 user user  850731 Nov  4 12:07 item_06000.png
-rw-rw-r-- 1 user user 1064030 Nov  4 12:07 item_09000.png
-rw-rw-r-- 1 user user 1482092 Nov  4 12:07 item_12000.png
-rw-rw-r-- 1 user user  740318 Nov  4 12:08 item_15000.png
-rw-rw-r-- 1 user user 1327931 Nov  4 12:08 item_18000.png
-rw-rw-r-- 1 user user  825150 Nov  4 12:08 item_20000.png

$ df -h /home/user
/dev/sdd       1007G  633G  324G  67% /

$ curl -s "http://127.0.0.1:7860/api/run?path=clean30_v4_5ep_r64_bs8"
{"id":"clean30_v4_5ep_r64_bs8","name":"clean30_v4_5ep_r64_bs8","status":"running",
 "step":1720,"max_steps":null,"epoch":null,"batch_index":null,
 "updated_at":"2026-11-04T09:06:40Z","stale_seconds":200.8,"refresh_seconds":5.0,
 "eta_seconds":null,"eta_at":null,
 "stats":{"loss":0.402433,"lr":0.0001449,"sigma":0.5692,"grad_norm":0.0344,
 "sec_per_step":10.452,"peak_vram_gb":22.51,"loss_ema":0.41043186863371606},
 "config":{"base_model":null,...,"max_steps":null,...},"checkpoint_count":0}

$ OLS loss ~ 1 + sigma + step/1000 (86 rows)
beta = [0.43525, -0.04308, -0.00165]   t = [9.38, -0.6, -0.14]   R2 = 0.00438
```

Note on `peak_vram_gb`: it is `torch.cuda.max_memory_allocated()/2**30` (`moss_lora/utils.py`
lines 194–198), i.e. allocated bytes only; the `nvidia-smi` figure additionally includes
reserved-but-unallocated cache and the CUDA context, hence 22.51 GiB vs 24.83 GiB.
