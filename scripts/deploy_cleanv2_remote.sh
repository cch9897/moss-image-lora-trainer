#!/usr/bin/env bash
# cleanV2 production sequence for the WSL host (GPU-host / RTX 5090).
#
# Run on the remote host:
#   ssh -i ~/.ssh/id_ed25519_gpu_host -o BatchMode=yes \
#       -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
#       user@<gpu-host>
#   cd $HOME/MOSS-Lora-Trainer
#   bash scripts/deploy_cleanv2_remote.sh plan
#   bash scripts/deploy_cleanv2_remote.sh generate
#   bash scripts/deploy_cleanv2_remote.sh cache      # returns PID immediately
#   bash scripts/deploy_cleanv2_remote.sh finalize   # after cache exits 0
#   bash scripts/deploy_cleanv2_remote.sh train      # after finalize
#
# The old $HOME/cache/clean30_v4_1m (201G) is never touched.  The new
# cleanV2 cache is built at $HOME/cache/cleanV2_1m (~236G).
set -euo pipefail

REPO=${REPO:-$HOME/MOSS-Lora-Trainer}
VENV=${VENV:-$REPO/venv/bin/python}
META=${META:-$HOME/dataset/clean30_top256/final_v4/metadata.jsonl}
IMAGES_ROOT=${IMAGES_ROOT:-$HOME/dataset/clean30_top256/final_v4}
OUT=${OUT:-$HOME/dataset/cleanV2}
CACHE=${CACHE:-$HOME/cache/cleanV2_1m}
CACHE_LOG=${CACHE_LOG:-$HOME/runs/cleanV2_cache.log}
CACHE_PID=${CACHE_PID:-$HOME/runs/cleanV2_cache.pid}
CONFIG=${CONFIG:-configs/train_5090_cleanV2_4ep_r64_bs8.yaml}
RUN_DIR=${RUN_DIR:-$HOME/runs/cleanV2_4ep_r64_bs8}
TRAIN_LOG=${TRAIN_LOG:-$HOME/runs/cleanV2_4ep_r64_bs8.log}
LAUNCHER=${LAUNCHER:-$HOME/runs/run_train_cleanV2_4ep_r64_bs8.sh}
EPOCHS=${EPOCHS:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
MODE=${1:-plan}

MODEL=$HOME/models/MOSS-Image-Base
NEGATIVE_PROMPT="nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
VARIANT_WEIGHTS="original=0.16,long=0.2133,medium=0.2133,short=0.25,layout=0.20"
EXPECTED_BASE=20953
EXPECTED_BALANCED=24598

require_paths() {
  [ -f "$META" ] || { echo "missing metadata: $META" >&2; exit 1; }
  [ -d "$IMAGES_ROOT" ] || { echo "missing images root: $IMAGES_ROOT" >&2; exit 1; }
  [ -x "$VENV" ] || { echo "missing venv python: $VENV" >&2; exit 1; }
  cd "$REPO"
}

disk_free_gb() {
  df -Pk $HOME | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

check_disk() {
  local free
  free=$(disk_free_gb)
  echo "disk free on $HOME: ${free}G"
  if [ "$free" -lt 220 ]; then
    echo "refusing: need >=220G free for the 236G cleanV2 cache (peak text temp ~187G)" >&2
    exit 1
  fi
}

plan() {
  cat <<EOF
# 1) generate the cleanV2 manifest (foreground, ~1-2 min for 20,953 rows)
cd $REPO
$VENV scripts/make_clean_v2.py \\
  --metadata $META \\
  --images-root $IMAGES_ROOT \\
  --output $OUT

# 2) verify base/balanced/tag-only counts and validation
$VENV - <<'PY'
import json
r=json.load(open("$OUT/report.json"))
assert r["counts"]["base_rows"] == $EXPECTED_BASE
assert r["counts"]["balanced_rows"] == $EXPECTED_BALANCED
assert r["counts"]["tag_only_short_base"] == $EXPECTED_BASE
assert r["counts"]["empty_captions"] == 0
assert r["validation"]["passed"], r["validation"]
print(json.dumps(r["counts"], indent=2))
PY

# 3) build the NEW cache in the background (old clean30_v4_1m is untouched)
mkdir -p $HOME/runs
nohup setsid bash -c 'ulimit -n 65536; exec $VENV -m moss_lora.cache \\
  --input $OUT/metadata.jsonl \\
  --output $CACHE \\
  --model $MODEL \\
  --target-area 1048576 \\
  --max-text-length 1024 \\
  --negative-prompt "$NEGATIVE_PROMPT" \\
  --variant-weights $VARIANT_WEIGHTS \\
  --missing-caption error' \\
  >> $CACHE_LOG 2>&1 < /dev/null &
echo \$! > $CACHE_PID

# 4) after the cache exits 0 and index.json has 24,598 items: exact step math
STEPS=\$($VENV - <<'PY'
import json
from collections import Counter
idx=json.load(open("$CACHE/index.json"))
counts=Counter((int(i["bucket_w"]), int(i["bucket_h"])) for i in idx["items"])
print(sum(n//$BATCH_SIZE for n in counts.values()))
PY
)
MAX_STEPS=\$((STEPS * $EPOCHS))
# write configs/train_5090_cleanV2_4ep_r64_bs8.yaml with max_steps=\$MAX_STEPS

# 5) launch (only after cache+config validation)
mkdir -p $RUN_DIR
nohup setsid bash $LAUNCHER >/dev/null 2>&1 < /dev/null &
echo \$! > $RUN_DIR/trainer.pid
EOF
}

generate() {
  require_paths
  "$VENV" scripts/make_clean_v2.py \
    --metadata "$META" \
    --images-root "$IMAGES_ROOT" \
    --output "$OUT"
  "$VENV" - "$OUT/report.json" <<'PY'
import json, sys
path = sys.argv[1]
report = json.load(open(path))
counts = report["counts"]
errors = []
if counts["base_rows"] != 20953:
    errors.append(f"base_rows={counts['base_rows']}")
if counts["balanced_rows"] != 24598:
    errors.append(f"balanced_rows={counts['balanced_rows']}")
if counts["expected_balanced_rows"] != 24598:
    errors.append(f"expected_balanced_rows={counts['expected_balanced_rows']}")
if counts["tag_only_short_base"] != 20953:
    errors.append(f"tag_only_short_base={counts['tag_only_short_base']}")
if counts["tag_only_short_balanced"] != 24598:
    errors.append(f"tag_only_short_balanced={counts['tag_only_short_balanced']}")
if counts["empty_captions"] != 0:
    errors.append(f"empty_captions={counts['empty_captions']}")
if report["artists"]["known"] != 255:
    errors.append(f"known artists={report['artists']['known']}")
if report["artists"]["unknown_rows"] != 771:
    errors.append(f"unknown rows={report['artists']['unknown_rows']}")
if report["artists"]["target"] != 74:
    errors.append(f"target={report['artists']['target']}")
if not report["validation"]["passed"]:
    errors.append(f"validation={report['validation']}")
if errors:
    raise SystemExit("cleanV2 validation failed: " + "; ".join(errors))
print(json.dumps(counts, indent=2))
PY
}

cache() {
  require_paths
  check_disk
  [ -f "$OUT/metadata.jsonl" ] || { echo "run generate first" >&2; exit 1; }
  if [ -e "$CACHE" ]; then
    echo "cache path already exists: $CACHE (delete it explicitly to rebuild)" >&2
    exit 2
  fi
  mkdir -p $HOME/runs
  nohup setsid bash -c "ulimit -n 65536; exec '$VENV' -m moss_lora.cache \
    --input '$OUT/metadata.jsonl' \
    --output '$CACHE' \
    --model '$MODEL' \
    --target-area 1048576 \
    --max-text-length 1024 \
    --negative-prompt '$NEGATIVE_PROMPT' \
    --variant-weights '$VARIANT_WEIGHTS' \
    --missing-caption error" \
    >> "$CACHE_LOG" 2>&1 < /dev/null &
  echo $! > "$CACHE_PID"
  sleep 5
  echo "cache PID: $(cat "$CACHE_PID")"
  echo "cache log: $CACHE_LOG"
  ps -p "$(cat "$CACHE_PID")" -o pid=,etime=,cmd= || true
  tail -n 5 "$CACHE_LOG" || true
}

finalize() {
  require_paths
  [ -f "$CACHE/index.json" ] || { echo "cache index missing: $CACHE/index.json" >&2; exit 1; }
  if [ -f "$CACHE_PID" ] && kill -0 "$(cat "$CACHE_PID")" 2>/dev/null; then
    echo "cache still running (PID $(cat "$CACHE_PID")); wait for exit 0 first" >&2
    exit 2
  fi
  local steps max_steps items
  items=$("$VENV" - "$CACHE/index.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["items"]))
PY
)
  [ "$items" -eq "$EXPECTED_BALANCED" ] || {
    echo "cache items $items != $EXPECTED_BALANCED" >&2; exit 1; }
  steps=$("$VENV" - "$CACHE/index.json" "$BATCH_SIZE" <<'PY'
import json, sys
from collections import Counter
idx=json.load(open(sys.argv[1]))
batch=int(sys.argv[2])
counts=Counter((int(i["bucket_w"]), int(i["bucket_h"])) for i in idx["items"])
print(sum(n//batch for n in counts.values()))
PY
)
  max_steps=$((steps * EPOCHS))
  mkdir -p "$RUN_DIR"
  cat > "$CONFIG" <<YAML
# cleanV2: balanced 24,598 rows, 4 epochs, bs8, rank 64, 1M pixels.
# steps/epoch = sum(items_per_bucket // 8) = $steps (exact from cleanV2 index)
# max_steps = $steps * $EPOCHS = $max_steps
model:
  path: $MODEL
  dtype: bf16
  local_files_only: true
data:
  cache_dir: $CACHE
  num_workers: 4
  batch_size: $BATCH_SIZE
  drop_last: true
  pin_memory: true
train:
  device: cuda
  output_dir: $RUN_DIR
  max_steps: $max_steps
  gradient_accumulation_steps: 1
  learning_rate: 1.5e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]
  warmup_steps: 200
  lr_scheduler: cosine
  max_grad_norm: 1.0
  seed: 42
  gradient_checkpointing: true
  compile: false
  compile_mode: default
  sdpa_backend: auto
  ema_decay: 0.0
  timestep_sampling: logit_normal
  logit_normal_mean: -0.40
  logit_normal_std: 0.95
  sigma_cutoff: 0.95
  uniform_mix: 0.01
  keep_last_checkpoints: 5
  save_epoch: true
lora:
  rank: 64
  alpha: 64
  dropout: 0.0
  targets: [q, k, v, o, ffn_up_gate, ffn_down]
  text_projection: false
  final_linear: false
logging:
  log_every: 20
  save_every: 3000
  sample_every: 3000
  sample_steps: 30
  sample_cfg: 3.0
  sample_skip_layer: 22
  sample_negative_prompt: "$NEGATIVE_PROMPT"
  sample_val_ids: [item_00000, item_03000, item_06000, item_09000,
                   item_12000, item_15000, item_18000, item_20000]
  comfy_export: true
  comfy_export_checkpoints: false
YAML
  cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ulimit -n 65536
cd $REPO
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo \$\$ > $RUN_DIR/trainer.pid
exec $VENV -m moss_lora.trainer --config $CONFIG >> $TRAIN_LOG 2>&1
EOF
  chmod +x "$LAUNCHER"
  echo "cache items: $items"
  echo "steps/epoch: $steps"
  echo "max_steps:   $max_steps"
  echo "config:      $CONFIG"
  echo "launcher:    $LAUNCHER"
}

train() {
  require_paths
  [ -f "$CONFIG" ] || { echo "run finalize first" >&2; exit 1; }
  [ -x "$LAUNCHER" ] || { echo "missing launcher: $LAUNCHER" >&2; exit 1; }
  mkdir -p "$RUN_DIR"
  nohup setsid bash "$LAUNCHER" >/dev/null 2>&1 < /dev/null &
  sleep 5
  local pid
  pid=$(cat "$RUN_DIR/trainer.pid")
  echo "trainer PID: $pid"
  echo "trainer log: $TRAIN_LOG"
  ps -p "$pid" -o pid=,etime=,cmd= || true
  tail -n 5 "$TRAIN_LOG" || true
}

case "$MODE" in
  plan) plan ;;
  generate) generate ;;
  cache) cache ;;
  finalize) finalize ;;
  train) train ;;
  *) echo "usage: $0 {plan|generate|cache|finalize|train}" >&2; exit 2 ;;
esac
