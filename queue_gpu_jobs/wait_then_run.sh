#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${1:?usage: wait_then_run.sh GPU_ID TARGET_PID EXPECTED_TEXT JOB_SCRIPT}"
TARGET_PID="${2:?usage: wait_then_run.sh GPU_ID TARGET_PID EXPECTED_TEXT JOB_SCRIPT}"
EXPECTED_TEXT="${3:?usage: wait_then_run.sh GPU_ID TARGET_PID EXPECTED_TEXT JOB_SCRIPT}"
JOB_SCRIPT="${4:?usage: wait_then_run.sh GPU_ID TARGET_PID EXPECTED_TEXT JOB_SCRIPT}"

INTERVAL="${INTERVAL:-60}"
IDLE_CHECKS="${IDLE_CHECKS:-5}"
LOCK_FILE="${LOCK_FILE:-/tmp/pkno_gpu_${GPU_ID}.queue.lock}"

if [[ ! -x "$JOB_SCRIPT" ]]; then
  echo "ERROR: job script is not executable: $JOB_SCRIPT" >&2
  exit 1
fi

if [[ ! -r "/proc/$TARGET_PID/stat" ]]; then
  echo "ERROR: target PID does not exist: $TARGET_PID" >&2
  exit 1
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: another queue runner already holds $LOCK_FILE" >&2
    exit 1
  fi
else
  echo "WARN: flock not found; duplicate queue protection is disabled" >&2
fi

TARGET_CMD="$(tr '\0' ' ' < "/proc/$TARGET_PID/cmdline" || true)"
if [[ "$TARGET_CMD" != *"$EXPECTED_TEXT"* ]]; then
  echo "ERROR: PID $TARGET_PID does not look like the expected task." >&2
  echo "Expected text: $EXPECTED_TEXT" >&2
  echo "Actual cmdline: $TARGET_CMD" >&2
  echo "Set ALLOW_PATTERN_MISMATCH=1 to bypass this check." >&2
  if [[ "${ALLOW_PATTERN_MISMATCH:-0}" != "1" ]]; then
    exit 1
  fi
fi

TARGET_START_TIME="$(awk '{print $22}' "/proc/$TARGET_PID/stat")"

same_process_running() {
  [[ -r "/proc/$TARGET_PID/stat" ]] || return 1
  [[ "$(awk '{print $22}' "/proc/$TARGET_PID/stat")" == "$TARGET_START_TIME" ]]
}

echo "[$(date '+%F %T')] Waiting for PID $TARGET_PID on cuda$GPU_ID"
echo "Expected text: $EXPECTED_TEXT"

while same_process_running; do
  sleep "$INTERVAL"
done

echo "[$(date '+%F %T')] Target PID finished. Checking cuda$GPU_ID idle state."

ok=0
while (( ok < IDLE_CHECKS )); do
  pids="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)"

  if [[ -z "$pids" ]]; then
    ok=$((ok + 1))
    echo "[$(date '+%F %T')] cuda$GPU_ID idle check $ok/$IDLE_CHECKS"
  else
    ok=0
    echo "[$(date '+%F %T')] cuda$GPU_ID still has compute process(es): $pids"
  fi

  sleep "$INTERVAL"
done

echo "[$(date '+%F %T')] cuda$GPU_ID appears idle. Starting $JOB_SCRIPT"
"$JOB_SCRIPT"
