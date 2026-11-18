# clean30_v4_5ep_r64_bs8 — paused (SIGSTOP), 2026-11-05

Reversible in-memory pause. Process, CUDA context and VRAM are preserved; nothing was
killed, restarted, or modified (no config/code/checkpoint changes).

Host: `user@<other-host>` (WSL2 Debian), repo `/home/user/MOSS-Lora-Trainer`.

## Process tree (all in one isolated session/process group)

| PID | PPID | PGID | SID | STAT (after pause) | Role |
|---|---|---|---|---|---|
| **52606** | 1 | 52606 | 52606 | `TNsl` | trainer main (session leader) |
| 279539 | 52606 | 52606 | 52606 | `TNl` | dataloader worker 0 |
| 279540 | 52606 | 52606 | 52606 | `TNl` | dataloader worker 1 |
| 279541 | 52606 | 52606 | 52606 | `TNl` | dataloader worker 2 |
| 279542 | 52606 | 52606 | 52606 | `TNl` | dataloader worker 3 |

Dashboard **52501** is a *separate* session (`PGID/SID 52501`) and was **not** touched;
it still answers on `:7860` (HTTP 200).

Command: `venv/bin/python -m moss_lora.trainer --config configs/train_5090_v4_5ep_r64_bs8.yaml`
Launched originally with `nohup setsid bash /home/user/runs/run_train_v4_5ep_r64_bs8.sh`.
Log (both stdout/stderr): `/home/user/runs/clean30_v4_5ep_r64_bs8.log`.
Reparented to PID 1, so it is not owned by any shell/tmux that could kill it.

## Pause command used

```bash
kill -STOP -52606        # whole isolated process group: main + 4 workers
```

Safe because pgid 52606 contains only trainer + its workers; the dashboard (52501) and
any SSH shell live in different sessions/pgids.

Fallback (if the group is ever restructured):

```bash
kill -STOP 52606 279539 279540 279541 279542
```

## State at pause (2026-11-05 19:01:16 +08:00 baseline, frozen since)

- Logged step: **10480** (last metrics row), `loss=0.529890`, `lr=1.450e-05`,
  `sigma=0.5673`, `grad_norm=0.0432`, `sec_per_step=10.423`, `peak_vram_gb=22.58`.
  In-memory progress was ~step 10490–10500 when frozen (nothing was flushed after 10480).
- Total schedule: `max_steps=13070` (5 epochs × 2614) → **~80.2% done**, in epoch 5 of 5.
- Latest log line: `[step 10480] loss=0.529890 ... variant_histogram={"layout":14,"long":28,"medium":49,"original":25,"short":44}`
- `metrics.csv`: 525 lines, last modified `2026-11-05 18:58:41` (mtime unchanged >30 s after stop).
- `.log`: 98374 bytes, mtime also frozen at `2026-11-05 18:58:41`.
- Dashboard `/api/runs`: `step: 10480, max_steps: 13070, status: "running"` (it derives status
  from metric recency; it will look stale/frozen, which is expected).

## Latest checkpoint — cold resume is possible

`/home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints/step_010456/` (written 18:54, epoch 4 end):

```
adapter_config.json       296 B
adapter_model.safetensors 594592688 B
lr_scheduler.pt           1501 B
optimizer.pt              1189522965 B
rng.pt                    14614 B
trainer_state.json        2138 B   -> step 10456, epoch 4, batch_index 0, epoch_end true
```

All checkpoints: `step_002614, 003000, 005228, 006000, 007842, 009000, 010456`.

If the paused process were killed instead, the run would cold-resume from **step 10456**
(epoch 4 end) and lose **~35–45 steps (~6–8 min)** of optimizer/RNG progress; the epoch-5
start is otherwise intact.

## Verification after `kill -STOP` (+8 s and +38 s)

```
    PID  PPID   PGID    SID STAT     %CPU CMD
  52606     1  52606  52606 TNsl      100 venv/bin/python -m moss_lora.trainer --config configs/...
 279539 52606  52606  52606 TNl       0.1 ...
 279540 52606  52606  52606 TNl       0.1 ...
 279541 52606  52606  52606 TNl       0.1 ...
 279542 52606  52606  52606 TNl       0.1 ...
  52501 52501  52501  52501 SNsl      0.1 venv/bin/python -m moss_lora.dashboard ...   # untouched
```

- Every trainer/worker PID has `T` in STAT (stopped by signal). No process exited; PIDs unchanged.
- Log + metrics size/mtime identical at T+0 and T+30 s → no progress.
- GPU (RTX 5090): utilization **99% → 1–9%** (idle noise), power **~540 W → 34–45 W**;
  memory **26 489 MiB still allocated** (expected — the CUDA context is frozen, not freed).
  `nvidia-smi --query-compute-apps` still lists PID 52606.

## Resume (reversible — VRAM is NOT freed by SIGSTOP)

```bash
ssh user@<other-host>
kill -CONT -52606        # resumes main + all 4 workers in one call
```

Or individually, if the group was split:

```bash
kill -CONT 52606 279539 279540 279541 279542
```

Then confirm with `ps -o pid,stat -p 52606,279539,279540,279541,279542` (`T` → `S/R`) and that
`/home/user/runs/clean30_v4_5ep_r64_bs8.log` advances again. A short 1–2 step dip in
`sec_per_step` on the first step after resume is normal.

**VRAM warning:** SIGSTOP keeps the ~26 GB CUDA allocation reserved. To actually free the GPU,
the process must be killed and relaunched from `checkpoints/step_010456` with
`--resume /home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints/step_010456`, losing the
~35–45 steps since that checkpoint. Do not do this unless explicitly requested.

## Do not

- Do **not** send SIGTERM/SIGINT/SIGKILL to 52606 or its workers.
- Do **not** `kill -CONT -52501` (dashboard group) or restart the dashboard.
- Do **not** launch a second trainer: the GPU still holds 26 GB reserved, so a new run would OOM.
