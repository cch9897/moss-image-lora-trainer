# clean30 v4 rank64 bs8 5-epoch launch

Date: 2026-11-04 (UTC). Host: user@<other-host> (WSL2 Debian, RTX 5090).
Run: `clean30_v4_5ep_r64_bs8`, config `configs/train_5090_v4_5ep_r64_bs8.yaml`.

## Exact step math

Source: `/home/user/cache/clean30_v4_1m/index.json` (`items` array) and
`CachedDataset.__len__`; sampler `BucketBatchSampler(..., batch_size=8, drop_last=True)`.

| quantity | value |
|---|---|
| cached items | **20,953** |
| steps / epoch | **2,614** |
| max_steps (5 epochs) | **13,070** |

`drop_last` drops the last partial batch **per bucket**, so `steps_per_epoch` is
`sum(items_per_bucket // 8)`, not `20,953 // 8`. The global floor is 2,619, but only 2,614
complete same-shape batches exist; asking for 2,619 would either crash collation or run
into a sixth epoch. The existing bs3 recipe's 6,980 steps/epoch (6,980 x 5 = 34,900) is
the same per-bucket math, which confirms this interpretation.

Per-bucket counts at bs8 (sum = 2,614):

```
(704,1408) 838//8=104   (768,1344) 1235//8=154  (832,1280) 974//8=121
(896,1152) 692//8=86    (896,1216) 1941//8=242  (1024,1024) 1273//8=159
(1152,896) 1086//8=135  (1216,896) 5721//8=715  (1280,832) 4208//8=526
(1344,768) 2119//8=264  (1408,704) 866//8=108
```

Live check with the real cache (remote venv):

```
dataset_len 20953
bs 8 batches/epoch 2614
bs 3 batches/epoch 6980
global_floor_8 2619
```

## Checkpoint policy audit

Files: `moss_lora/trainer.py`, `tests/test_checkpoint_retention.py`,
`tests/test_trainer_smoke.py`.

1. `save_epoch: true` saves at every epoch boundary. With 2,614 optimizer steps/epoch
   (gradient_accumulation_steps=1) the epoch-end saves land at 2,614 / 5,228 / 7,842 /
   10,456, and the final save at 13,070 is also `epoch_end=True`. Verified by code path
   and by the new resume test (2 epochs -> epoch-end dirs at step 3 and 6 on the tiny
   fixture, `epoch_end: true`).
2. Epoch-end checkpoints are full-state: `save_checkpoint` writes
   `adapter_model.safetensors` (LoRA weights), `optimizer.pt`, `lr_scheduler.pt`,
   `rng.pt`, `trainer_state.json` (step/epoch/batch_index/epoch_end/full config);
   `adapter_ema.safetensors` is added when EMA is enabled (this run has `ema_decay: 0`).
   **Defect found and fixed:** an epoch-end checkpoint stored `batch_index` equal to the
   number of batches consumed in the finished epoch. On resume, `train()` slices
   `batches[batch_index:]`, gets an empty list and raises
   `RuntimeError: No batches in epoch N`. Fixed by making `batch_index` the resume
   cursor: in-loop epoch-end saves pass `(epoch + 1, 0)`, and the final save normalizes
   to `(epoch + 1, 0)` when it lands exactly on an epoch boundary
   (`micro_in_epoch >= len(sampler)`). Resume from any epoch-end checkpoint now starts at
   the first batch of the next epoch.
3. `keep_last_checkpoints: 5` counts only non-epoch checkpoints. **Defect found and
   fixed:** `prune_checkpoints` previously took the newest 5 of *all* checkpoints and
   then added epoch-end ones, so an epoch-end checkpoint inside the newest window
   displaced a regular checkpoint. Pruning now partitions candidates into epoch-end and
   regular, keeps `regular[-5:]`, and unions all epoch-end dirs. All epoch-end dirs are
   never pruned.
4. Regular saves happen every 3,000 steps (`completed_step % save_every == 0`) and are
   pruned to the newest 5 regular dirs. In this run only 3,000 / 6,000 / 9,000 / 12,000
   exist (13,070 is not a multiple of 3,000), so all four survive; the policy caps at 5
   if the run were longer.

Code changes (local == remote, md5 `da955c64ac7eb266b1044de1b8a7a94e`):

```diff
-    keep = set(candidates[-keep_last:])
-    keep.update(path for path in candidates if _is_epoch_end_checkpoint(path))
+    epoch_end = {path for path in candidates if _is_epoch_end_checkpoint(path)}
+    regular = [path for path in candidates if path not in epoch_end]
+    keep = set(regular[-keep_last:]) | epoch_end
...
-            save_checkpoint(..., completed_step, epoch, micro_in_epoch, epoch_end=True)
+            save_checkpoint(..., completed_step, epoch + 1, 0, epoch_end=True)
...
-    final_dir = save_checkpoint(..., completed_step, epoch, micro_in_epoch,
-                                epoch_end=config.train.save_epoch)
+    finished_epoch = len(sampler) > 0 and micro_in_epoch >= len(sampler)
+    final_dir = save_checkpoint(..., completed_step,
+                                epoch + 1 if finished_epoch else epoch,
+                                0 if finished_epoch else micro_in_epoch,
+                                epoch_end=config.train.save_epoch)
```

Tests added:
- `tests/test_checkpoint_retention.py::test_epoch_end_checkpoint_does_not_consume_a_regular_slot`
  (epoch-end at step 5, keep_last 3, steps 1..6 -> `[3,4,5,6]` remain, `[1,2]` pruned;
  the old code would have pruned 3 and kept only 2 regular dirs).
- `tests/test_trainer_smoke.py::test_epoch_end_checkpoint_resumes_at_next_epoch`
  (full-state files present, `epoch_end: true`, cursor `(1,0)`, resume to step 6 works,
  final cursor `(2,0)`, metrics rows 1..6).

Test output (local `.venv`, Python 3.14.4, pytest 9.1.1):

```
$ .venv/bin/python -m pytest tests/test_checkpoint_retention.py \
      tests/test_trainer_smoke.py::test_epoch_end_checkpoint_resumes_at_next_epoch -q
.........                                                                [100%]
9 passed in 4.88s

$ .venv/bin/python -m pytest tests/test_trainer_smoke.py -q
...........                                                              [100%]
11 passed in 5.46s
```

## Config

`configs/train_5090_v4_5ep_r64_bs8.yaml`, identical local and remote:

```
local  b5a80c68562bbcd1c2c8277470695c59  configs/train_5090_v4_5ep_r64_bs8.yaml
remote b5a80c68562bbcd1c2c8277470695c59  /home/user/MOSS-Lora-Trainer/configs/train_5090_v4_5ep_r64_bs8.yaml
```

Diff versus `configs/train_5090_v4_5ep_r64.yaml` (header expanded; only the four
requested values changed):

```diff
-  batch_size: 3
+  batch_size: 8
-  output_dir: /home/user/runs/clean30_v4_5ep_r64
+  output_dir: /home/user/runs/clean30_v4_5ep_r64_bs8
-  max_steps: 34900
+  max_steps: 13070
-  learning_rate: 1.0e-4
+  learning_rate: 1.5e-4
```

`/home/user/runs/clean30_v4_5ep_r64_bs8` did not exist before launch, so no `_v2`
directory was needed. Kept unchanged: rank 64 / alpha 64, LoRA targets,
timestep sampling, cache path, `save_every: 3000`, `sample_every: 3000` (same validation
IDs/sampler settings), `keep_last_checkpoints: 5`, `save_epoch: true`,
`comfy_export: true`, `comfy_export_checkpoints: false`.
LR rationale (also in the config header): bs8 scaling from the bs3 1e-4 recipe;
1e-4 * 8/3 = 2.67e-4 linear, 1.5e-4 is the conservative step for the larger batch.

Effective values loaded by the running environment:

```
bs 8 max_steps 13070 lr 0.00015 keep 5 save_epoch True save_every 3000 sample_every 3000
out /home/user/runs/clean30_v4_5ep_r64_bs8
```

## Launcher and dashboard

Launcher: `/home/user/runs/run_train_v4_5ep_r64_bs8.sh` (chmod +x)

```bash
#!/usr/bin/env bash
set -euo pipefail
ulimit -n 65536
cd /home/user/MOSS-Lora-Trainer
export PYTHONUNBUFFERED=1
echo $$ > /home/user/runs/clean30_v4_5ep_r64_bs8.pid
exec venv/bin/python -m moss_lora.trainer \
  --config configs/train_5090_v4_5ep_r64_bs8.yaml \
  >> /home/user/runs/clean30_v4_5ep_r64_bs8.log 2>&1
```

Started with `nohup setsid bash /home/user/runs/run_train_v4_5ep_r64_bs8.sh`.

| process | PID | evidence |
|---|---|---|
| trainer | **52606** | `venv/bin/python -m moss_lora.trainer --config configs/train_5090_v4_5ep_r64_bs8.yaml`, still alive |
| dashboard | **52501** | `venv/bin/python -m moss_lora.dashboard --root /home/user/runs --host 0.0.0.0 --port 7860` |

Logs:
- trainer: `/home/user/runs/clean30_v4_5ep_r64_bs8.log`
- metrics: `/home/user/runs/clean30_v4_5ep_r64_bs8/metrics.csv`
- dashboard: `/home/user/runs/dashboard_v4.log`

Dashboard bind/HTTP proof:

```
LISTEN 0 2048 0.0.0.0:7860 0.0.0.0:*
HTTP 200 http://127.0.0.1:7860/
GET /api/run?path=clean30_v4_5ep_r64_bs8 -> {"status":"running","step":40,...}
```

(The dashboard's `max_steps` field stays `null` until the first checkpoint writes
`trainer_state.json`, which contains the full config; first checkpoint is at step 2,614.)

## First-step verification

fd limit from `/proc/52606/limits`:

```
Max open files            65536                65536                files
```

First logged metrics rows (`log_every: 20`, so step 20 is the earliest row):

```
step,loss,lr,sigma,grad_norm,sec_per_step,peak_vram_gb,variant_histogram
20,0.399837,1.575e-05,0.6229,0.0255,19.516,22.26,"{""layout"":19,""long"":34,""medium"":49,""original"":28,""short"":30}"
40,0.424803,3.075e-05,0.4931,0.0152,10.603,22.40,"{""layout"":24,""long"":37,""medium"":30,""original"":29,""short"":40}"
60,0.520920,4.575e-05,0.5756,0.0214,10.573,22.40,"{""layout"":27,""long"":30,""medium"":33,""original"":36,""short"":34}"
```

- Finite losses (0.3998 / 0.4248 / 0.5209); no NaN.
- Training keeps advancing after warmup (step 40 -> 60, 10.573 s/step).
- No `CUDA driver error`, OOM or traceback in the log (`grep -Ein` returns nothing).
- `peak_vram_gb = 22.40` (torch peak; `nvidia-smi` process total ~25.4 GiB includes the
  resident validation VAE).
- Steady-state `sec_per_step = 10.603` at step 40 and `10.573` at step 60. Step 20's
  19.5 is contaminated: `t_last` is set before the step-0 baseline validation, so that
  line averages the 8-sample x 30-step validation plus first-step CUDA warmup over 20
  steps (hence the few minutes of log silence before step 20).
- Effective batch size 8, rank 64, LR at step 20 is the warmup value 1.575e-5.

## Monitoring

```bash
# trainer log / metrics
ssh user@<other-host> 'tail -f /home/user/runs/clean30_v4_5ep_r64_bs8.log'
ssh user@<other-host> 'tail -f /home/user/runs/clean30_v4_5ep_r64_bs8/metrics.csv'
# process + fd
ssh user@<other-host> 'cat /home/user/runs/clean30_v4_5ep_r64_bs8.pid; grep "Max open files" /proc/$(cat /home/user/runs/clean30_v4_5ep_r64_bs8.pid)/limits'
# checkpoints
ssh user@<other-host> 'ls /home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints'
# dashboard
curl http://<other-host>:7860/
```

Expected checkpoint dirs (`.../checkpoints/step_NNNNNN`):

- epoch-end (full state, retained forever): **002614, 005228, 007842, 010456, 013070**
- regular (newest 5 kept, only 4 exist this run): **003000, 006000, 009000, 012000**
- samples: `samples/step_000000` (baseline) then 003000 / 006000 / 009000 / 012000.

At completion the run keeps 9 checkpoint dirs: 4 regular + 5 epoch-end. Resume example:

```bash
ssh user@<other-host>
cd /home/user/MOSS-Lora-Trainer
venv/bin/python -m moss_lora.trainer \
  --config configs/train_5090_v4_5ep_r64_bs8.yaml \
  --resume /home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints/step_005228
```
