# cleanV2 2-epoch extension (cleanV2_ext2ep_r64_bs8)

Date: 2026-11-11. Host: `user@<gpu-host>` (WSL2, RTX 5090).

The cleanV2 4-epoch run (`cleanV2_4ep_r64_bs8`, trainer PID 69669) is extended by
two epochs. Nothing in the running run or its cache is modified: the extension
resumes from the completed 4-epoch boundary checkpoint into a **new** output dir.

## Exact step math

| quantity | value |
|---|---|
| cached items | 24,598 |
| steps / epoch | **3,071** (`sum(items_per_bucket // 8)`) |
| old max_steps | 12,284 (4 epochs) |
| extension steps | 6,142 (2 epochs) |
| new max_steps | **18,426** (6 epochs) |

Expected saves in the extension run (`save_every: 3000`, `save_epoch: true`,
`keep_last_checkpoints: 5`; epoch-end checkpoints are never pruned):

| step | kind | `trainer_state` |
|---|---|---|
| 15,000 | regular | step 15000, epoch 4, batch 2716, epoch_end false |
| **15,355** | epoch 5 end | step 15355, epoch 5, batch 0, epoch_end true |
| 18,000 | regular | step 18000, epoch 5, batch 2645, epoch_end false |
| **18,426** | final | step 18426, epoch 6, batch 0, epoch_end true |

`save_every` triggers on `completed_step % 3000 == 0`; resuming at 12,284
means the next regular save is 15,000 (there is no 12,300 checkpoint).
Retained extension checkpoints are 15000 / 15355 / 18000 / 18426
(~1.66 GiB each, ~6.6 GiB total).

## Resume and LR semantics (verified against the code)

`moss_lora/trainer.py`: `build_lr_scheduler` is built from the **new** config
(`max_steps=18426`, `warmup_steps=0`) before `resume_checkpoint` calls
`lr_scheduler.load_state_dict`, which restores `last_epoch=12284` and the old
`_last_lr = [0.0]` (the completed 4-epoch cosine ends at zero). Because
`get_last_lr()[0] <= 0`, `train()` sets every optimizer group to
`max(get_lr()) = 1.5e-4 * 0.5 * (1 + cos(pi * 12284 / 18426))`.

Read-only simulation on the remote venv (torch 2.14.1+cu130), using
`checkpoints/step_009000/lr_scheduler.pt` stepped to 12284 to reproduce the
boundary state:

```
old schedule at 12284 : last_epoch 12284, get_last_lr [0.0], optimizer lr 0.0
after new load        : last_epoch 12284, get_last_lr [0.0], get_lr [3.75e-05]
resume fix applied    : optimizer lr 3.75e-05
step 12284 -> lr 3.750000e-05   (lambda 0.25)
step 12285 -> lr 3.748893e-05
step 15355 -> lr 1.004809e-05   (lambda 0.0669873)
step 18426 -> lr 0.0            (lambda 0.0)
```

The first resumed optimizer step uses 3.75e-5; the following step uses
3.748893e-5 after the first `lr_scheduler.step()`.

## Files

* Config: `configs/train_5090_cleanV2_ext2ep_r64_bs8.yaml`
* Watcher: `scripts/queue_cleanv2_ext2ep.sh`
* Extension output: `/home/user/runs/cleanV2_ext2ep_r64_bs8`
* Extension log: `/home/user/runs/cleanV2_ext2ep_r64_bs8.log`
* Watcher log: `/home/user/runs/cleanV2_ext2ep_watch.log`
* Watcher PID: `/home/user/runs/cleanV2_ext2ep_watch.pid`

## Watcher behavior

1. Single instance via `flock` on `/home/user/runs/cleanV2_ext2ep_watch.lock`;
   exits if the extension is already running or `step_018426` is complete; aborts
   if partial extension state exists without a live trainer.
2. Waits for PID 69669 to exit (checking the cmdline so a reused PID is not
   mistaken for the trainer), then waits up to 120 s for all
   `venv/bin/python -m moss_lora.trainer` processes to disappear.
3. Requires `metrics.csv` last complete row == 12284 and
   `checkpoints/step_012284/` with `trainer_state.json` =
   `{step: 12284, epoch: 4, batch_index: 0, epoch_end: true}` plus
   `adapter_model.safetensors`, `optimizer.pt`, `lr_scheduler.pt`, `rng.pt`.
   Any mismatch aborts without launching.
4. Requires >= 20 GiB free on `/home/user` and `/mnt/c` (checked with `df -Pk`,
   logged with `df -h`).
5. Writes `$EXT_RUN/launch_ext.sh` with `ulimit -n 65536`,
   `PYTHONUNBUFFERED=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, then
   starts it with `nohup setsid`; the launcher records its `$$` in
   `trainer.pid` and execs:

```
venv/bin/python -m moss_lora.trainer \
    --config configs/train_5090_cleanV2_ext2ep_r64_bs8.yaml \
    --resume /home/user/runs/cleanV2_4ep_r64_bs8/checkpoints/step_012284
```

6. Verifies the extension process is alive 10 s after launch, then exits; the
   trainer runs in its own session.

Start the watcher now (already done for this queue):

```bash
nohup setsid bash /home/user/MOSS-Lora-Trainer/scripts/queue_cleanv2_ext2ep.sh \
    > /home/user/runs/cleanV2_ext2ep_watch.log 2>&1 < /dev/null &
cat /home/user/runs/cleanV2_ext2ep_watch.pid
```

Cancel before the base run finishes (kills only the watcher; the base run and
any already-launched extension are untouched):

```bash
kill "$(cat /home/user/runs/cleanV2_ext2ep_watch.pid)"
```

Expected extension epoch ends: **15,355** and **18,426**. The run is complete
when `checkpoints/step_018426/trainer_state.json` reports step 18426.
