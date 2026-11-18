# clean30_v4_5ep_r64_bs8 — ComfyUI export and release (2026-11-05)

Run: `/home/user/runs/clean30_v4_5ep_r64_bs8` (host `user@<other-host>`, WSL2 Debian).
Exported from checkpoint **step_010456 (end of epoch 4 of 5)** — the last complete
full-state checkpoint. The paused in-memory process was ~10,490–10,500 (≈35–45 steps
ahead) but had no saved state, so that progress is **not** in the export. The 5-epoch
schedule (`max_steps=13070`) was **not** reached (~80.2% complete).

## Export

```bash
cd /home/user/MOSS-Lora-Trainer
venv/bin/python -m moss_lora.export_comfy \
  --adapter /home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints/step_010456 \
  --output  /home/user/runs/clean30_v4_5ep_r64_bs8/comfyui_lora_step_010456.safetensors
```

Output:

```
Exported 504 LoRA weights + 252 alpha scalars (756 keys, 594632360 bytes)
  from .../checkpoints/step_010456/adapter_model.safetensors
rank=64 alpha=64.0 layers=42 targets=q,k,v,o,ffn_up_gate,ffn_down
```

| Field | Value |
|---|---|
| Path | `/home/user/runs/clean30_v4_5ep_r64_bs8/comfyui_lora_step_010456.safetensors` |
| Size | 594,632,360 bytes (0.595 GB) |
| SHA256 | `d1244ef6718ad86235bda8f4d32fb56e180263f79d14ad5cfb09115bbdd818d5` |
| Manifest | `/home/user/runs/clean30_v4_5ep_r64_bs8/comfyui_lora_step_010456.manifest.json` |
| Key count | **756** = 252 `lora_A.weight` + 252 `lora_B.weight` + 252 `alpha` (source adapter: 504 tensors) |
| Layers | 42 (indices 0..41) × 6 targets (`q,k,v,o,ffn_up_gate,ffn_down`) |
| Rank / alpha | 64 / 64.0 (config `r=64`, `alpha=64.0`); A rank dim = 64, B rank dim = 64 |
| Dtype | `torch.float32` for all tensors; alpha is a 1-element float32 scalar |
| Naming | every key matches `^diffusion_model\.layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)\.(lora_A\.weight|lora_B\.weight|alpha)$` |
| Non-finite | 0 NaN/Inf tensors |
| Equivalence | all 504 A/B tensors are bit-identical (`torch.equal`) to `adapter_model.safetensors` |
| Metadata | `moss_lora_comfy=1`, `format=pt`, `base_model=/home/user/models/MOSS-Image-Base` |

Validation used `safetensors.safe_open` (framework `pt`, CPU) on the written file.

### Test caveat

`tests/test_export_comfy.py` **does not exist** on the host, and the venv has **no
pytest** installed (`venv/bin/python -m pytest` → "No module named pytest"; no `.venv`).
The only existing comfy test is `tests/test_trainer_smoke.py::test_comfy_export_failure_is_non_fatal`;
its assertion was reproduced manually and passed:

```
export_comfy_adapter("/tmp/does_not_exist_adapter") -> None
stderr: WARNING: ComfyUI LoRA export failed ...: Adapter weights not found ...
```

Nothing was installed to work around the missing pytest.

## Release

Trainer was still SIGSTOP-frozen (all PIDs in state `T`) when the export started; it was
killed only after validation succeeded.

```bash
kill -KILL -52606    # trainer main + 4 dataloader workers (isolated pgid)
kill -TERM  52501    # dashboard
```

Evidence:

- Before: `52606 TNsl`, `279539/279540/279541/279542 TNl` (all `moss_lora.trainer`).
- After: `ps -p 52606,279539,279540,279541,279542` → empty; `pgrep -af moss_lora` → none.
- Dashboard 52501 exited **2 s** after SIGTERM (no KILL needed); `ss -ltnp` → port 7860
  closed; `curl http://127.0.0.1:7860/` → `000` (connection refused).
- No `run_train*.sh` / tmux / nohup wrappers remain. No `.pid` file existed in the run dir
  (nothing to remove).

### Final GPU state (`/usr/lib/wsl/lib/nvidia-smi`)

```
No running processes found
~2518–2805 MiB / 32607 MiB, 7–16% util, ~52–65 W (idle P5)
```

Down from **26,489 MiB, 99% util, ~540 W** while the trainer was running. All of the
trainer's VRAM is freed. The residual ~2.5 GiB is the WSL2 dxgkrnl/WSLg baseline with
**zero** compute apps (nvidia-smi lists "No running processes found"; there are no
`/dev/nvidia*` nodes in WSL — only `/dev/dxg`), so it is not held by our processes and
will not block a new run. (It is higher than the ~1.2 GiB idle figure sometimes seen;
memory had already decayed 2.8 → 2.5 GiB and is driver-held.)

## What was kept (nothing deleted)

```
/home/user/runs/clean30_v4_5ep_r64_bs8/            13 GB total
├── checkpoints/  step_002614 003000 005228 006000 007842 009000 010456
├── samples/      step_000000, step_006000, ... (PNGs)
├── metrics.csv   525 rows, last step 10480
├── comfyui_lora_step_010456.safetensors   (new)
└── comfyui_lora_step_010456.manifest.json (new)
/home/user/runs/clean30_v4_5ep_r64_bs8.log  98374 B, last line step 10480
```

The log/metrics were already frozen at `2026-11-05 18:58:41` by the pause and were not
modified by the export or shutdown.

## Cold resume (not performed)

```bash
cd /home/user/MOSS-Lora-Trainer
nohup setsid venv/bin/python -m moss_lora.trainer \
  --config configs/train_5090_v4_5ep_r64_bs8.yaml \
  --resume /home/user/runs/clean30_v4_5ep_r64_bs8/checkpoints/step_010456 \
  >> /home/user/runs/clean30_v4_5ep_r64_bs8.log 2>&1 &
```

- `step_010456/trainer_state.json` = `step 10456, epoch 4, batch_index 0, epoch_end true`
  → resumes at the start of epoch 5, with optimizer, LR scheduler and RNG restored.
- `train.ema_decay: 0.0` in the config, so EMA is `None` and the absent
  `adapter_ema.safetensors` does **not** block `resume_checkpoint`.
- `--resume` accepts the checkpoint dir directly (it contains `trainer_state.json`).
- Progress lost by killing instead of continuing: only the ~35–45 steps (~6–8 min)
  between step 10456 and the frozen in-memory step.

## Caveats

- The export is **step 10,456 (epoch 4/5)**, not the paused ~10,490–10,500 state and not
  a completed 5-epoch model. Epoch 5 (steps 10,457–13,070) was never trained.
- `tests/test_export_comfy.py` is missing and pytest is not installed remotely; validation
  was done with a direct `safe_open` script plus the existing smoke-test assertion.
- ComfyUI loads this file via the LoRA loader; do not use `adapter_model.safetensors`
  (raw trainer naming) directly.
