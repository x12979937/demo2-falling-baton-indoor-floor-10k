#!/usr/bin/env bash
set -u

FS_ROOT="${1:?usage: monitor_state_replay_validation.sh FS_ROOT [sleep_seconds]}"
SLEEP_SECONDS="${2:-180}"
ARCHIVE_DIR="$FS_ROOT/archives"
META_DIR="$FS_ROOT/metadata"
VALIDATOR="$FS_ROOT/code/verify_replay_state.py"
TMP_BASE="/root/autodl-tmp/mingyu/state_replay_validation/tmp"
INDEX="$META_DIR/state_validation_index.jsonl"
STATUS="$META_DIR/state_validation_status.json"
LOG="$META_DIR/state_validation_monitor.log"
FAIL="$META_DIR/state_validation_failed.json"
PY="/root/autodl-tmp/mingyu/GieneSim_IsaacGym_IsaacSim_united/Conda/envs/isaacsim_py311/bin/python"
if [ ! -x "$PY" ]; then PY="python3"; fi

mkdir -p "$META_DIR" "$TMP_BASE"
touch "$INDEX"

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" >> "$LOG"
}

json_quote() {
  "$PY" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

has_ok_record() {
  local archive="$1" sha="$2"
  grep -F -- "\"archive\":\"$archive\"" "$INDEX" 2>/dev/null | grep -F -- "\"sha256\":\"$sha\"" | grep -F -- '"status":"ok"' >/dev/null 2>&1
}

write_status() {
  local ok_archives datasets latest checked_at
  ok_archives=$(grep -c '"status":"ok"' "$INDEX" 2>/dev/null || true)
  datasets=$(grep -o '"dataset_count":[0-9]*' "$INDEX" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
  latest=$(tail -1 "$INDEX" 2>/dev/null | sed 's/"/\\"/g')
  checked_at=$(date -Iseconds)
  printf '{"updated_at":"%s","fs_root":"%s","ok_archives":%d,"validated_datasets":%d,"latest_record":"%s"}\n' \
    "$checked_at" "$FS_ROOT" "${ok_archives:-0}" "${datasets:-0}" "$latest" > "$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
}

validate_one() {
  local archive="$1" base sha tmpout tmperr dataset_count bytes now
  [ -f "$archive" ] || return 0
  base=$(basename "$archive")
  sha=$(sha256sum "$archive" | awk '{print $1}')
  if has_ok_record "$archive" "$sha"; then
    return 0
  fi
  tmpout="$TMP_BASE/${base}.state_validation.json.$$"
  tmperr="$TMP_BASE/${base}.state_validation.err.$$"
  log "validating $base"
  if TMPDIR="$TMP_BASE" "$VALIDATOR" "$archive" --limit 0 --json > "$tmpout" 2> "$tmperr"; then
    dataset_count=$($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get("dataset_count", 0)))' "$tmpout")
    bytes=$(stat -c %s "$archive")
    now=$(date -Iseconds)
    printf '{"checked_at":"%s","archive":"%s","sha256":"%s","status":"ok","dataset_count":%d,"bytes":%d}\n' \
      "$now" "$archive" "$sha" "$dataset_count" "$bytes" >> "$INDEX"
    rm -f "$tmpout" "$tmperr"
    log "ok $base datasets=$dataset_count"
    write_status
    return 0
  fi
  now=$(date -Iseconds)
  printf '{"checked_at":"%s","archive":"%s","sha256":"%s","status":"failed","error":"see %s"}\n' \
    "$now" "$archive" "$sha" "$tmperr" > "$FAIL"
  cat "$tmperr" >> "$LOG"
  write_status
  log "failed $base"
  return 2
}

log "state validation monitor started fs_root=$FS_ROOT sleep=$SLEEP_SECONDS"
while true; do
  for archive in "$ARCHIVE_DIR"/*.tar.gz; do
    [ -e "$archive" ] || continue
    validate_one "$archive" || exit 2
  done
  write_status
  sleep "$SLEEP_SECONDS"
done
