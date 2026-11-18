#!/usr/bin/env bash
# Queue the cleanV2 2-epoch extension after cleanV2_4ep_r64_bs8 finishes.
#
# Waits for the current trainer PID, verifies the completed 4-epoch boundary
# checkpoint, then launches the extension detached. Idempotent: a second
# watcher exits, and an already-running / already-complete extension is left
# alone. Any incomplete state aborts instead of resuming from a partial run.
#
# Start (detached):
#   nohup setsid bash $HOME/MOSS-Lora-Trainer/scripts/queue_cleanv2_ext2ep.sh \
#       > $HOME/runs/cleanV2_ext2ep_watch.log 2>&1 < /dev/null &
# Cancel while it is still waiting:
#   kill "$(cat $HOME/runs/cleanV2_ext2ep_watch.pid)"
set -u -o pipefail

REPO=$HOME/MOSS-Lora-Trainer
PY=$REPO/venv/bin/python

BASE_RUN=$HOME/runs/cleanV2_4ep_r64_bs8
BASE_CONFIG=configs/train_5090_cleanV2_4ep_r64_bs8.yaml
BASE_PID=69669
RESUME_STEP=12284
RESUME_CKPT=$BASE_RUN/checkpoints/step_012284

EXT_RUN=$HOME/runs/cleanV2_ext2ep_r64_bs8
EXT_CONFIG=configs/train_5090_cleanV2_ext2ep_r64_bs8.yaml
EXT_LOG=$HOME/runs/cleanV2_ext2ep_r64_bs8.log
EXT_PID_FILE=$EXT_RUN/trainer.pid
EXT_LAUNCHER=$EXT_RUN/launch_ext.sh
FINAL_STEP=18426
FINAL_CKPT=$EXT_RUN/checkpoints/step_018426

WATCH_PID_FILE=$HOME/runs/cleanV2_ext2ep_watch.pid
LOCK_FILE=$HOME/runs/cleanV2_ext2ep_watch.lock
MIN_FREE_KB=$((20 * 1024 * 1024))   # 20 GiB
POLL_SECONDS=60

log() { printf '%s %s\n' "$(date -Is)" "$*"; }
abort() { log "ABORT: $*"; exit 1; }

pid_cmdline() {
    local pid="${1:-}"
    [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null
}

pid_is_base_trainer() {
    local cmd
    cmd=$(pid_cmdline "${1:-}") || return 1
    [[ "$cmd" == *"moss_lora.trainer"* && "$cmd" == *"$BASE_CONFIG"* ]]
}

pid_is_ext_trainer() {
    local cmd
    cmd=$(pid_cmdline "${1:-}") || return 1
    [[ "$cmd" == *"moss_lora.trainer"* && "$cmd" == *"$EXT_CONFIG"* ]]
}

any_trainer_pids() {
    pgrep -f 'venv/bin/python -m moss_lora[.]trainer' 2>/dev/null || true
}

ext_state_ok() {
    local state="$FINAL_CKPT/trainer_state.json"
    [[ -f "$state" ]] || return 1
    "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1]))
raise SystemExit(0 if int(s.get("step", -1)) == int(sys.argv[2]) else 1)' \
        "$state" "$FINAL_STEP" 2>/dev/null
}

# ------------------------------------------------------------------ single instance
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "another watcher already holds $LOCK_FILE; nothing to do"
        exit 0
    fi
else
    log "WARNING: flock not found; relying on the PID file for single-instance checks"
fi

if [[ -f "$WATCH_PID_FILE" ]]; then
    other=$(tr -dc '0-9' < "$WATCH_PID_FILE" 2>/dev/null)
    if [[ -n "$other" && "$other" != "$$" ]] && pid_cmdline "$other" | grep -q 'queue_cleanv2_ext2ep[.]sh'; then
        log "another watcher is already running (PID $other); nothing to do"
        exit 0
    fi
fi
printf '%s\n' "$$" > "$WATCH_PID_FILE"

log "watcher started (PID $$); waiting for base trainer PID $BASE_PID"

# ------------------------------------------------------------------ idempotency
if [[ -f "$EXT_PID_FILE" ]]; then
    running=$(tr -dc '0-9' < "$EXT_PID_FILE" 2>/dev/null)
    if pid_is_ext_trainer "$running"; then
        log "extension is already running (PID $running); nothing to do"
        exit 0
    fi
fi
running=$(pgrep -f "venv/bin/python -m moss_lora[.]trainer --config $EXT_CONFIG" | head -n1 || true)
if [[ -n "$running" ]] && pid_is_ext_trainer "$running"; then
    log "extension is already running (PID $running, found by pgrep); nothing to do"
    exit 0
fi
if ext_state_ok; then
    log "extension is already complete (step_$FINAL_STEP); nothing to do"
    exit 0
fi
if [[ -f "$EXT_RUN/metrics.csv" || -d "$EXT_RUN/checkpoints" ]]; then
    abort "partial extension state exists under $EXT_RUN but no extension trainer is running; refusing to relaunch automatically"
fi

# ------------------------------------------------------------------ wait for base run
while pid_is_base_trainer "$BASE_PID"; do
    sleep "$POLL_SECONDS"
done

if kill -0 "$BASE_PID" 2>/dev/null; then
    log "PID $BASE_PID is alive but is not the cleanV2 4-epoch trainer; treating the run as exited"
fi
log "base trainer PID $BASE_PID is gone; waiting for all trainer processes to exit"

for _ in $(seq 1 60); do
    pids=$(any_trainer_pids)
    [[ -z "$pids" ]] && break
    sleep 2
done
pids=$(any_trainer_pids)
if [[ -n "$pids" ]]; then
    abort "trainer process(es) still running after PID $BASE_PID exited: $(echo $pids)"
fi

# ------------------------------------------------------------------ verify completion
[[ -f "$BASE_RUN/metrics.csv" ]] || abort "missing $BASE_RUN/metrics.csv"
last_step=$(awk -F, 'NF >= 7 && $1 ~ /^[0-9]+$/ {step=$1} END {print step}' "$BASE_RUN/metrics.csv")
log "base metrics.csv last valid step: ${last_step:-<none>}"
[[ "$last_step" == "$RESUME_STEP" ]] || abort "expected metrics.csv last step $RESUME_STEP, got ${last_step:-<none>}"

[[ -d "$RESUME_CKPT" ]] || abort "missing final 4-epoch checkpoint $RESUME_CKPT"
log "verifying $RESUME_CKPT"
"$PY" - "$RESUME_CKPT" "$RESUME_STEP" <<'PY' || abort "final 4-epoch checkpoint verification failed"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_step = int(sys.argv[2])
required = ("adapter_model.safetensors", "optimizer.pt", "lr_scheduler.pt", "rng.pt")
missing = [name for name in required if not (path / name).is_file()]
if missing:
    raise SystemExit(f"missing checkpoint file(s): {missing}")
state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
actual = (int(state.get("step", -1)), int(state.get("epoch", -1)),
          int(state.get("batch_index", -1)), bool(state.get("epoch_end", False)))
expected = (expected_step, 4, 0, True)
if actual != expected:
    raise SystemExit(f"trainer_state (step, epoch, batch_index, epoch_end) = {actual}, expected {expected}")
print(f"checkpoint OK: step={actual[0]} epoch={actual[1]} batch_index={actual[2]} epoch_end={actual[3]}")
PY

# ------------------------------------------------------------------ disk space
guest_kb=$(df -Pk $HOME | awk 'NR==2 {print $4}')
host_kb=$(df -Pk /mnt/c | awk 'NR==2 {print $4}')
log "guest $HOME: $(df -h $HOME | tail -n1)"
log "host   /mnt/c:   $(df -h /mnt/c | tail -n1)"
[[ -n "$guest_kb" && "$guest_kb" -ge "$MIN_FREE_KB" ]] \
    || abort "guest free space below 20 GiB (${guest_kb:-unknown} KiB available)"
[[ -n "$host_kb" && "$host_kb" -ge "$MIN_FREE_KB" ]] \
    || abort "host C: free space below 20 GiB (${host_kb:-unknown} KiB available)"

# ------------------------------------------------------------------ launch
cd "$REPO" || abort "cannot cd to $REPO"
[[ -x "$PY" ]] || abort "missing $PY"
mkdir -p "$EXT_RUN" || abort "cannot create $EXT_RUN"

log "launching extension: $PY -m moss_lora.trainer --config $EXT_CONFIG --resume $RESUME_CKPT"
cat > "$EXT_LAUNCHER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec 9>&- 2>/dev/null || true
ulimit -n 65536 || true
cd $REPO
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo \$\$ > $EXT_PID_FILE
exec $PY -m moss_lora.trainer --config $EXT_CONFIG --resume $RESUME_CKPT >> $EXT_LOG 2>&1
EOF
chmod +x "$EXT_LAUNCHER"
nohup setsid bash "$EXT_LAUNCHER" >/dev/null 2>&1 </dev/null &
sleep 10
ext_pid=$(tr -dc '0-9' < "$EXT_PID_FILE" 2>/dev/null)
if [[ -z "${ext_pid:-}" ]] || ! pid_is_ext_trainer "$ext_pid"; then
    log "ERROR: extension process is not running after 10s (pid file: ${ext_pid:-<missing>}); last log lines:"
    tail -n 20 "$EXT_LOG" 2>/dev/null
    exit 1
fi
log "extension launched detached: PID $ext_pid, log $EXT_LOG, pid file $EXT_PID_FILE"
log "expected saves: step_015000, step_015355 (epoch 5 end), step_018000, step_018426 (final, max_steps)"
