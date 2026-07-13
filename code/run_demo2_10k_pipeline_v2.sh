#!/usr/bin/env bash
set -Eeuo pipefail

TOTAL_EPISODES=${TOTAL_EPISODES:-10000}
BATCH_SIZE=${BATCH_SIZE:-2}
SEED0=${SEED0:-202607110000}
RENDERER=${RENDERER:-RaytracedLighting}
FALLING_RODS=${FALLING_RODS:-8}
HEIGHT=${HEIGHT:-360}
WIDTH=${WIDTH:-640}
VIDEO_STRIDE=${VIDEO_STRIDE:-2}
FPS=${FPS:-30}
FRAMES=${FRAMES:-180}
FS_ROOT=/autodl-fs/data/mingyu/demo2_falling_baton_indoor_floor_10k
TMP_ROOT=/root/autodl-tmp/mingyu/demo2_falling_baton_indoor_floor_10k_work_v3_20260711
HELPER=/root/autodl-tmp/mingyu/IsaacLab/Projects/ThreeDemo/scripts/threedemo_annotation_export.py
SCRIPT=/root/autodl-tmp/mingyu/IsaacLab/Projects/ThreeDemo/scripts/render_formal_threedemo_physx_videos.py
PY=/root/autodl-tmp/mingyu/GieneSim_IsaacGym_IsaacSim_united/Conda/envs/isaacsim_py311/bin/python
VALIDATOR=$FS_ROOT/code/verify_replay_state.py
ARCHIVE_DIR=$FS_ROOT/archives
META_DIR=$FS_ROOT/metadata
LOG_DIR=$FS_ROOT/logs
REPO_DIR=$FS_ROOT/github_repo
TOKEN_FILE=$FS_ROOT/.github_token
ASKPASS=$FS_ROOT/github_askpass.sh
REPO_URL=https://github.com/x12979937/demo2-falling-baton-indoor-floor-10k.git
STATUS=$FS_ROOT/status.json
PIPELINE_LOG=$LOG_DIR/pipeline_v2.log
PIPELINE_SELF=$FS_ROOT/run_demo2_10k_pipeline_v2.sh

case "$FS_ROOT" in /autodl-fs/data/mingyu/*) ;; *) echo "unsafe FS_ROOT=$FS_ROOT" >&2; exit 10;; esac
case "$TMP_ROOT" in /root/autodl-tmp/mingyu/*|/root/autodl-tmp/data/mingyu/*) ;; *) echo "unsafe TMP_ROOT=$TMP_ROOT" >&2; exit 11;; esac
mkdir -p "$ARCHIVE_DIR" "$META_DIR" "$LOG_DIR" "$TMP_ROOT/batches" "$TMP_ROOT/tmp" "$TMP_ROOT/runtime_home" "$TMP_ROOT/replay_validation" "$FS_ROOT/code" "$FS_ROOT/git_tmp"

log() { printf '[%s] %s\n' "$(date '+%F %T %z')" "$*" | tee -a "$PIPELINE_LOG"; }

disk_json() {
  python3 - <<'PY'
import json, subprocess
rows = []
out = subprocess.check_output(['df', '-Pk', '/root/autodl-tmp', '/autodl-fs/data'], text=True).splitlines()[1:]
for line in out:
    parts = line.split()
    rows.append({'mount': parts[-1], 'avail_kb': int(parts[3]), 'used_pct': parts[4]})
print(','.join(json.dumps(row, separators=(',', ':')) for row in rows))
PY
}

archived_count() {
  if [ -f "$META_DIR/archive_index.jsonl" ]; then
    python3 - "$META_DIR/archive_index.jsonl" <<'PY_COUNT'
import json, sys
total = 0
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        total += int(json.loads(line).get("episode_count", 0))
    except Exception:
        pass
print(total)
PY_COUNT
  else
    echo 0
  fi
}

next_episode() {
  if [ -f "$META_DIR/archive_index.jsonl" ]; then
    python3 - "$META_DIR/archive_index.jsonl" <<'PY_NEXT'
import json, sys
next_ep = 0
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        item = json.loads(line)
        next_ep = max(next_ep, int(item.get("end_episode", -1)) + 1)
    except Exception:
        pass
print(next_ep)
PY_NEXT
    return
  fi
  local next=0 f base end
  shopt -s nullglob
  for f in "$ARCHIVE_DIR"/demo2_falling_baton_indoor_floor_episodes_*.tar.gz "$ARCHIVE_DIR"/demo2_falling_baton_indoor_floor_episodes_*.tar.gz.part000; do
    base=$(basename "$f")
    if [[ "$base" =~ episodes_([0-9]{5})_([0-9]{5})\.tar\.gz(\.part000)?$ ]]; then
      end=$((10#${BASH_REMATCH[2]}))
      if (( end + 1 > next )); then next=$((end + 1)); fi
    fi
  done
  shopt -u nullglob
  echo "$next"
}

write_status() {
  local phase=${1:-running}
  local current=${2:-0}
  local message=${3:-ok}
  local count disks
  count=$(archived_count)
  disks=$(disk_json)
  cat > "$STATUS.tmp" <<EOF
{"updated_at":"$(date -Iseconds)","phase":"$phase","current_episode":$current,"total_episodes":$TOTAL_EPISODES,"archived_episode_count":$count,"batch_size":$BATCH_SIZE,"tmp_root":"$TMP_ROOT","fs_root":"$FS_ROOT","github_repo":"https://github.com/x12979937/demo2-falling-baton-indoor-floor-10k","message":"$message","disks":[$disks]}
EOF
  mv "$STATUS.tmp" "$STATUS"
}

cleanup_tmp_intermediates() {
  find "$TMP_ROOT/tmp" -mindepth 1 -maxdepth 1 -print0 2>/dev/null | xargs -0r rm -rf --
  find "$TMP_ROOT/replay_validation" -mindepth 1 -maxdepth 1 -type d -mtime +1 -print0 2>/dev/null | xargs -0r rm -rf --
}

pack_frame_dirs() {
  local target="$1"
  find "$target" -type d -name '*_frames' -print0 2>/dev/null | while IFS= read -r -d '' frame_dir; do
    local parent base archive tmp_archive
    parent=$(dirname "$frame_dir")
    base=$(basename "$frame_dir")
    archive="$parent/${base}.tar.gz"
    tmp_archive="$archive.tmp"
    rm -f -- "$tmp_archive" "$archive"
    tar -C "$parent" -czf "$tmp_archive" "$base"
    tar -tzf "$tmp_archive" >/dev/null
    mv "$tmp_archive" "$archive"
    rm -rf --one-file-system "$frame_dir"
  done
}

require_tmp_space() {
  cleanup_tmp_intermediates
  local avail
  avail=$(df -Pk /root/autodl-tmp | awk 'NR==2{print $4}')
  if (( avail < 1048576 )); then
    log "tmp free space below 1GB: ${avail}KB"
    write_status "blocked" "$(next_episode)" "tmp_free_below_1GB"
    exit 20
  fi
}

validate_episode() {
  local ep_dir=$1
  "$PY" - "$ep_dir" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]) / 'falling_baton'
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
assert manifest.get('ok') is True, manifest
assert manifest.get('support_surface') == 'indoor_floor', manifest.get('support_surface')
assert int(manifest.get('object_count', 0)) == 8, manifest.get('object_count')
assert int(manifest.get('floor_contact_event_count', 0)) >= 8, manifest.get('floor_contact_event_count')
assert (root / 'dataset.npz').is_file()
views = ['overview', 'closeup', 'left_side', 'right_side', 'top', 'front', 'left_oblique', 'right_oblique']
for view in views:
    assert (root / f'falling_baton_{view}.mp4').is_file(), view
data = np.load(root / 'dataset.npz', allow_pickle=True)
for key in ['object_trajectory_w','object_pos_w','object_pose_w','object_quat_wxyz','object_lin_vel_w','object_ang_vel_w','object_momentum_kg_m_s','object_angular_momentum_kg_m2_rad_s','object_mass_kg','object_center_of_mass_local_m','object_inertia_diag_kg_m2','object_static_friction','object_dynamic_friction','object_restitution','object_text_description','object_standard_description','scene_text_description','scene_standard_description','task_text_description','task_standard_description','state_replay_import_contract_json','floor_contact','initial_object_pose_w','initial_object_lin_vel_w','initial_object_ang_vel_w','final_pose_w','object_metadata_json','object_visual_mesh_export_path','object_collision_mesh_export_path','object_mesh_export_dir','object_mesh_export_format','mesh_assets_manifest_path']:
    assert key in data.files, key
def scalar_str(x):
    a=np.asarray(x); v=a.item() if a.shape==() else a.reshape(-1)[0]
    if isinstance(v, bytes): v=v.decode('utf-8','replace')
    return str(v)
mesh_manifest = root / scalar_str(data['mesh_assets_manifest_path'])
assert mesh_manifest.is_file(), mesh_manifest
mesh_info = json.loads(mesh_manifest.read_text(encoding='utf-8'))
assert int(mesh_info.get('object_count', -1)) == 8, mesh_info.get('object_count')
visual_meshes = [scalar_str(x) for x in np.asarray(data['object_visual_mesh_export_path']).reshape(-1)]
collision_meshes = [scalar_str(x) for x in np.asarray(data['object_collision_mesh_export_path']).reshape(-1)]
formats = [scalar_str(x) for x in np.asarray(data['object_mesh_export_format']).reshape(-1)]
assert len(visual_meshes) == 8 and len(collision_meshes) == 8 and formats == ['obj'] * 8
for rel in visual_meshes + collision_meshes:
    fp = root / rel
    assert fp.is_file() and fp.stat().st_size > 0, fp
assert data['object_pos_w'].shape[1] == 8
assert data['floor_contact'].any(axis=0).sum() == 8
assert str(np.asarray(data['annotation_schema']).reshape(-1)[0]) == 'threedemo_object_2d_bbox_mask_v1'
assert bool(np.asarray(data['contains_2d_bboxes']).reshape(-1)[0])
assert bool(np.asarray(data['contains_pixel_masks']).reshape(-1)[0])
assert bool(np.asarray(data['videos_include_2d_bbox_and_pixel_mask_overlay']).reshape(-1)[0])
cam_views = [scalar_str(x) for x in np.asarray(data['camera_view_names']).reshape(-1)]
assert set(cam_views) == set(views), cam_views
ann_manifest = root / scalar_str(data['annotation_manifest_path'])
assert ann_manifest.is_file(), ann_manifest
ann = json.loads(ann_manifest.read_text(encoding='utf-8'))
assert ann.get('contains_2d_bboxes') and ann.get('contains_pixel_masks'), ann
bbox_paths = json.loads(scalar_str(data['object_2d_bbox_xyxy_path_json']))
mask_paths = json.loads(scalar_str(data['object_pixel_mask_rle_path_json']))
for view in views:
    bp = root / bbox_paths[view]; mp = root / mask_paths[view]
    assert bp.is_file() and bp.stat().st_size > 0, bp
    assert mp.is_file() and mp.stat().st_size > 0, mp
    b = np.load(bp, allow_pickle=True)
    assert b['object_2d_bbox_xyxy'].shape[1:] == (8, 4), b['object_2d_bbox_xyxy'].shape
    assert b['object_2d_bbox_valid'].any(), view
scene=json.loads(scalar_str(data['scene_standard_description']))
contract=json.loads(scalar_str(data['state_replay_import_contract_json']))
objstd=[json.loads(x.decode('utf-8','replace') if isinstance(x, bytes) else str(x)) for x in np.asarray(data['object_standard_description']).reshape(-1)]
objtxt=[x.decode('utf-8','replace') if isinstance(x, bytes) else str(x) for x in np.asarray(data['object_text_description']).reshape(-1)]
assert scene.get('engine') == 'IsaacSim PhysX'
assert int(scene.get('object_count', -1)) == 8
assert contract.get('required_engine') == 'IsaacSim PhysX'
assert len(objstd) == 8 and len(objtxt) == 8 and all(x.strip() for x in objtxt)
assert all(all(k in obj for k in ['mass_kg','center_of_mass_local_m','inertia_diag_kg_m2','collision','visual']) for obj in objstd)
PY
}

run_episode() {
  local ep=$1
  local ep_id seed ep_dir attempt rc
  ep_id=$(printf '%05d' "$ep")
  seed=$((SEED0 + ep))
  ep_dir=$2/falling_baton_$ep_id
  rm -rf -- "$ep_dir"
  for attempt in 1 2 3; do
    require_tmp_space
    mkdir -p "$ep_dir"
    log "episode $ep_id attempt $attempt start"
    set +e
    env \
      HOME="$TMP_ROOT/runtime_home" \
      XDG_CACHE_HOME="$TMP_ROOT/runtime_home/.cache" \
      XDG_CONFIG_HOME="$TMP_ROOT/runtime_home/.config" \
      XDG_DATA_HOME="$TMP_ROOT/runtime_home/.local/share" \
      TMPDIR="$TMP_ROOT/tmp" \
      ISAAC_ACTIVE_GPU="${ISAAC_ACTIVE_GPU:-0}" \
      "$PY" "$SCRIPT" \
        --task falling_baton \
        --out-dir "$ep_dir" \
        --frames "$FRAMES" \
        --fps "$FPS" \
        --video-stride "$VIDEO_STRIDE" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --falling-rods "$FALLING_RODS" \
        --falling-disturbances-per-rod 3 \
        --seed "$seed" \
        --renderer "$RENDERER" \
        --keep-video-frames \
        > "$ep_dir/run_attempt_${attempt}.log" 2>&1
    rc=$?
    set -e
    cleanup_tmp_intermediates
    if (( rc == 0 )) && validate_episode "$ep_dir" > "$ep_dir/validate.log" 2>&1 && TMPDIR="$TMP_ROOT/tmp" "$PY" "$VALIDATOR" "$ep_dir" --json --write-episode-validation >> "$ep_dir/validate.log" 2>&1; then
      pack_frame_dirs "$ep_dir"
      log "episode $ep_id ok state_replay_validation_pass_raw_rgb_frames_packed"
      return 0
    fi
    log "episode $ep_id attempt $attempt failed rc=$rc"
    tail -80 "$ep_dir/run_attempt_${attempt}.log" >> "$LOG_DIR/episode_${ep_id}_attempt_${attempt}_tail.log" 2>/dev/null || true
    rm -rf -- "$ep_dir"
    sleep $((attempt * 15))
  done
  log "episode $ep_id failed after retries"
  write_status "failed" "$ep" "episode_${ep_id}_failed"
  return 1
}

archive_batch() {
  local batch_dir=$1 start=$2 end=$3 expected=$4 got vids frame_archives validations mesh_manifests mesh_objs expected_mesh_objs ann_manifests bbox_files mask_files tarball tmp_tar size checksum batch_name validation_dir validation_log checked
  got=$(find "$batch_dir" -maxdepth 3 -type f -name dataset.npz | wc -l)
  vids=$(find "$batch_dir" -maxdepth 4 -type f -name '*.mp4' | wc -l)
  frame_archives=$(find "$batch_dir" -maxdepth 4 -type f -name '*_frames.tar.gz' | wc -l)
  validations=$(find "$batch_dir" -maxdepth 4 -type f -name 'state_replay_validation.json' | wc -l)
  mesh_manifests=$(find "$batch_dir" -path '*/mesh_assets/mesh_assets_manifest.json' -type f | wc -l)
  mesh_objs=$(find "$batch_dir" -path '*/mesh_assets/*.obj' -type f | wc -l)
  ann_manifests=$(find "$batch_dir" -path '*/annotations/annotation_manifest.json' -type f | wc -l)
  bbox_files=$(find "$batch_dir" -path '*/annotations/*_bbox_mask_stats.npz' -type f | wc -l)
  mask_files=$(find "$batch_dir" -path '*/annotations/*_mask_rle.jsonl.gz' -type f | wc -l)
  expected_mesh_objs=$((expected * FALLING_RODS * 2))
  if [ "$got" -ne "$expected" ] || [ "$vids" -ne $((expected * 8)) ] || [ "$frame_archives" -ne $((expected * 8)) ] || [ "$validations" -ne "$expected" ] || [ "$mesh_manifests" -ne "$expected" ] || [ "$mesh_objs" -ne "$expected_mesh_objs" ] || [ "$ann_manifests" -ne "$expected" ] || [ "$bbox_files" -ne $((expected * 8)) ] || [ "$mask_files" -ne $((expected * 8)) ]; then
    log "archive validation failed for $batch_dir got=$got vids=$vids frame_archives=$frame_archives validations=$validations mesh_manifests=$mesh_manifests mesh_objs=$mesh_objs ann_manifests=$ann_manifests bbox_files=$bbox_files mask_files=$mask_files expected=$expected expected_mesh_objs=$expected_mesh_objs"
    return 1
  fi
  batch_name="demo2_episodes_$(printf '%05d' "$start")_$(printf '%05d' "$end")"
  tarball="$ARCHIVE_DIR/demo2_falling_baton_indoor_floor_episodes_$(printf '%05d' "$start")_$(printf '%05d' "$end").tar.gz"
  validation_dir="$batch_dir/replay_state_exports"; validation_log="$TMP_ROOT/replay_validation/${batch_name}.json"
  rm -rf -- "$validation_dir"; mkdir -p "$validation_dir"
  if ! TMPDIR="$TMP_ROOT/tmp" "$PY" "$VALIDATOR" "$batch_dir" --limit 0 --json --export-dir "$validation_dir" > "$validation_log" 2>&1; then
    log "state replay export/validation failed for $batch_dir"; cp -f "$validation_log" "$LOG_DIR/${batch_name}_state_replay_validation_failed.json" 2>/dev/null || true; rm -rf -- "$validation_dir"; return 1
  fi
  checked=$(python3 - "$validation_log" <<'PYJSON'
import json, sys
p=sys.argv[1]
d=json.load(open(p,encoding='utf-8'))
print(int(d.get('checked',0)) if d.get('ok') else -1)
PYJSON
)
  if [ "$checked" -ne "$expected" ]; then log "state replay validation count mismatch for $batch_dir checked=$checked expected=$expected"; cp -f "$validation_log" "$LOG_DIR/${batch_name}_state_replay_validation_count_mismatch.json" 2>/dev/null || true; rm -rf -- "$validation_dir"; return 1; fi
  cp -f "$validation_log" "$META_DIR/last_state_replay_validation.json" 2>/dev/null || true
  tmp_tar="$tarball.tmp"
  rm -f -- "$tmp_tar" "$tarball" "$tarball.sha256" "$tarball".part* "$tarball.parts.sha256" "$tarball.parts.json"
  tar -C "$(dirname "$batch_dir")" -czf "$tmp_tar" "$(basename "$batch_dir")"
  tar -tzf "$tmp_tar" >/dev/null
  mv "$tmp_tar" "$tarball"
  checksum=$(sha256sum "$tarball" | awk '{print $1}')
  size=$(stat -c %s "$tarball")
  printf '%s  %s
' "$checksum" "$(basename "$tarball")" > "$tarball.sha256"
  if (( size > 95000000 )); then
    split -b 90M -d -a 3 "$tarball" "$tarball.part"
    rm -f -- "$tarball"
    sha256sum "$tarball".part* > "$tarball.parts.sha256"
    python3 - "$tarball" "$checksum" "$start" "$end" "$expected" "$size" "$mesh_manifests" "$mesh_objs" <<'PY_ARCHIVE' >> "$META_DIR/archive_index.jsonl"
import json, sys, hashlib
from pathlib import Path
base=Path(sys.argv[1]); checksum=sys.argv[2]
start=int(sys.argv[3]); end=int(sys.argv[4]); expected=int(sys.argv[5]); size=int(sys.argv[6]); mesh_manifests=int(sys.argv[7]); mesh_objs=int(sys.argv[8])
parts=[]
for part in sorted(base.parent.glob(base.name + '.part*')):
    h=hashlib.sha256()
    with part.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    parts.append({'file': str(part), 'name': part.name, 'bytes': part.stat().st_size, 'sha256': h.hexdigest()})
(base.parent/(base.name + '.parts.json')).write_text(json.dumps(parts, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'archive': str(base), 'archive_sha256': checksum, 'split': True, 'part_count': len(parts), 'parts': parts, 'start_episode': start, 'end_episode': end, 'episode_count': expected, 'bytes': size, 'state_replay_validated': True, 'contains_replay_state_exports': True, 'contains_episode_mesh_assets': True, 'mesh_assets_validated': True, 'mesh_manifest_count': mesh_manifests, 'mesh_obj_count': mesh_objs, 'state_replay_validator': 'code/verify_replay_state.py'}, ensure_ascii=False))
PY_ARCHIVE
    log "archived split strict episodes $(printf '%05d' "$start")-$(printf '%05d' "$end") size=$size parts=$(find "$ARCHIVE_DIR" -maxdepth 1 -type f -name "$(basename "$tarball").part*" | wc -l)"
  else
    printf '{"archive":"%s","sha256":"%s","split":false,"start_episode":%d,"end_episode":%d,"episode_count":%d,"bytes":%d,"state_replay_validated":true,"contains_replay_state_exports":true,"contains_episode_mesh_assets":true,"mesh_assets_validated":true,"mesh_manifest_count":%d,"mesh_obj_count":%d,"state_replay_validator":"code/verify_replay_state.py","created_at":"%s"}
' "$tarball" "$checksum" "$start" "$end" "$expected" "$size" "$mesh_manifests" "$mesh_objs" "$(date -Iseconds)" >> "$META_DIR/archive_index.jsonl"
    log "archived with replay_state_exports, episode mesh assets, and state-replay-validated episodes $(printf '%05d' "$start")-$(printf '%05d' "$end") -> $tarball"
  fi
  rm -rf -- "$batch_dir"; cleanup_tmp_intermediates
}
write_repo_files() {
  mkdir -p "$FS_ROOT/code" "$META_DIR"
  cp -f "$HELPER" "$FS_ROOT/code/threedemo_annotation_export.py"
  cp -f "$SCRIPT" "$FS_ROOT/code/render_formal_threedemo_physx_videos.py"
  cp -f "$PIPELINE_SELF" "$FS_ROOT/code/run_demo2_10k_pipeline_v2.sh" 2>/dev/null || true
  cp -f "$VALIDATOR" "$FS_ROOT/code/verify_replay_state.py" 2>/dev/null || true
  cat > "$META_DIR/generation_config.json" <<EOF
{"schema":"demo2_falling_baton_indoor_floor_10k_v4_replayable_state_mesh_assets_bbox_mask_multiview","total_episodes":$TOTAL_EPISODES,"batch_size":$BATCH_SIZE,"frames":180,"fps":30,"video_stride":2,"width":$WIDTH,"height":$HEIGHT,"falling_rods":8,"renderer":"RaytracedLighting","support_surface":"indoor_floor","dataset_note":"Renderer RGB frame folders are retained exactly as PNGs, packed per view as *_frames.tar.gz inside each episode and then removed from tmp as expanded directories. Archives also contain dataset.npz, collision_events.jsonl, manifest.json, per-episode state_replay_validation.json, eight multiview mp4 files with 2D boxes and pixel-mask overlays, annotations/*.npz bbox files, annotations/*.jsonl.gz pixel masks, per-episode mesh_assets/*.obj for visual/collision replay, and replay_state_exports/replay_state.npz plus replay_state_manifest.json for physics-engine import verification."}
EOF
  cat > "$FS_ROOT/README.md" <<'EOF'
# demo2 falling baton indoor floor 10k

Generated on remote machine 1 under allowed paths only.

Data layout:
- archives/: tar.gz batches of falling baton episodes, split to stay below GitHub's single-file size limit.
- metadata/archive_index.jsonl: archive ranges, sizes, and checksums.
- metadata/generation_config.json: simulation and rendering settings.
- code/render_formal_threedemo_physx_videos.py: collection script snapshot.
- status.json: latest generation/push status.
- each episode folder: mesh_assets/mesh_assets_manifest.json plus visual_mesh.obj and collision_mesh.obj for every baton.

Each episode contains the simulation dataset, collision events, manifest, logs, eight multiview videos with 2D boxes and pixel-mask overlays, per-view raw renderer RGB PNG frames packed as `*_frames.tar.gz`, and `state_replay_validation.json`. Each archive also contains replay_state_exports/<episode>/replay_state.npz and replay_state_manifest.json exported by the validator for IsaacSim PhysX import verification.
EOF
}

source_network_turbo() {
  if [ -f /etc/network_turbo ]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo >/dev/null 2>&1 || true
  fi
}

prepare_repo() {
  if [ ! -s "$TOKEN_FILE" ] && [ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]; then
    umask 077
    printf '%s' "${GITHUB_TOKEN:-${GH_TOKEN:-}}" > "$TOKEN_FILE"
  fi
  if [ ! -x "$ASKPASS" ] && [ -s "$TOKEN_FILE" ]; then
    umask 077
    cat > "$ASKPASS" <<'ASKPASS_EOF'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) cat /autodl-fs/data/mingyu/demo2_falling_baton_indoor_floor_10k/.github_token ;;
  *) printf '%s\n' ;;
esac
ASKPASS_EOF
    chmod 700 "$ASKPASS"
  fi
  [ -x "$ASKPASS" ] || { log "missing github askpass"; return 1; }
  [ -s "$TOKEN_FILE" ] || { log "missing github token"; return 1; }
  source_network_turbo
  export GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 HOME="$FS_ROOT/git_home" TMPDIR="$FS_ROOT/git_tmp"
  mkdir -p "$HOME" "$TMPDIR"
  if [ "${FORCE_PUSH_REWRITE:-0}" = "1" ]; then
    rm -rf -- "$REPO_DIR"; mkdir -p "$REPO_DIR"
    git -C "$REPO_DIR" init >> "$LOG_DIR/github_push.log" 2>&1
    git -C "$REPO_DIR" checkout -B main >> "$LOG_DIR/github_push.log" 2>&1
    git -C "$REPO_DIR" remote add origin "$REPO_URL" 2>/dev/null || git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
  elif [ ! -d "$REPO_DIR/.git" ]; then
    rm -rf -- "$REPO_DIR"; log "cloning github repo"; git clone "$REPO_URL" "$REPO_DIR" >> "$LOG_DIR/github_push.log" 2>&1 || return 1
  else
    git -C "$REPO_DIR" fetch origin main >> "$LOG_DIR/github_push.log" 2>&1 || true
    git -C "$REPO_DIR" checkout -B main >> "$LOG_DIR/github_push.log" 2>&1 || true
  fi
  git -C "$REPO_DIR" config user.name "x12979937"
  git -C "$REPO_DIR" config user.email "mingyu_xu9646@163.com"
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
}
sync_repo_and_push() {
  local reason=${1:-batch}; write_repo_files
  if ! prepare_repo; then log "github prepare failed for $reason"; return 1; fi
  rm -rf -- "$REPO_DIR/archives" "$REPO_DIR/metadata" "$REPO_DIR/code"
  mkdir -p "$REPO_DIR/archives" "$REPO_DIR/metadata" "$REPO_DIR/code"
  find "$ARCHIVE_DIR" -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.tar.gz.part*' -o -name '*.sha256' -o -name '*.parts.json' \) -print0 | sort -z | while IFS= read -r -d '' f; do
    ln -f "$f" "$REPO_DIR/archives/$(basename "$f")" 2>/dev/null || cp -f "$f" "$REPO_DIR/archives/$(basename "$f")"
  done
  cp -f "$META_DIR/archive_index.jsonl" "$REPO_DIR/metadata/archive_index.jsonl" 2>/dev/null || : > "$REPO_DIR/metadata/archive_index.jsonl"
  cp -f "$META_DIR/generation_config.json" "$REPO_DIR/metadata/generation_config.json"
  cp -f "$META_DIR/state_replay_schema.json" "$REPO_DIR/metadata/state_replay_schema.json" 2>/dev/null || true
  cp -f "$META_DIR/last_state_replay_validation.json" "$REPO_DIR/metadata/last_state_replay_validation.json" 2>/dev/null || true
  cp -f "$FS_ROOT/status.json" "$REPO_DIR/status.json" 2>/dev/null || true
  cp -f "$FS_ROOT/README.md" "$REPO_DIR/README.md"
  cp -f "$FS_ROOT/code/threedemo_annotation_export.py" "$REPO_DIR/code/threedemo_annotation_export.py"
  cp -f "$FS_ROOT/code/render_formal_threedemo_physx_videos.py" "$REPO_DIR/code/render_formal_threedemo_physx_videos.py"
  cp -f "$FS_ROOT/code/run_demo2_10k_pipeline_v2.sh" "$REPO_DIR/code/run_demo2_10k_pipeline_v2.sh" 2>/dev/null || true
  cp -f "$FS_ROOT/code/verify_replay_state.py" "$REPO_DIR/code/verify_replay_state.py" 2>/dev/null || true
  cp -f "$FS_ROOT/code/monitor_state_replay_validation.sh" "$REPO_DIR/code/monitor_state_replay_validation.sh" 2>/dev/null || true
  git -C "$REPO_DIR" add -A README.md status.json archives metadata code >> "$LOG_DIR/github_push.log" 2>&1
  if git -C "$REPO_DIR" diff --cached --quiet; then log "github no changes for $reason"; return 0; fi
  git -C "$REPO_DIR" commit -m "Update demo2 replayable state data ($reason)" >> "$LOG_DIR/github_push.log" 2>&1 || true
  source_network_turbo
  local push_rc=0 local_head remote_head push_attempt
  for push_attempt in 1 2 3 4 5; do
    push_rc=0
    source_network_turbo
    if [ "${FORCE_PUSH_REWRITE:-0}" = "1" ]; then
      git -C "$REPO_DIR" push --force origin HEAD:main >> "$LOG_DIR/github_push.log" 2>&1 || push_rc=$?
    else
      git -C "$REPO_DIR" push origin HEAD:main >> "$LOG_DIR/github_push.log" 2>&1 || push_rc=$?
    fi
    local_head=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)
    remote_head=$(git -C "$REPO_DIR" ls-remote origin refs/heads/main 2>> "$LOG_DIR/github_push.log" | awk '{print $1}' || true)
    if [ "$push_rc" -eq 0 ]; then log "github push ok for $reason attempt=$push_attempt"; return 0; fi
    if [ -n "$local_head" ] && [ "$remote_head" = "$local_head" ]; then
      log "github push verified on remote for $reason despite local rc=$push_rc attempt=$push_attempt"
      return 0
    fi
    log "github push attempt $push_attempt failed for $reason rc=$push_rc remote_head=${remote_head:-none} local_head=${local_head:-none}"
    sleep $((push_attempt * 30))
  done
  log "github push failed for $reason after retries remote_head=${remote_head:-none} local_head=${local_head:-none}"; return 1
}
finalize_credentials() {
  if [ "${KEEP_GITHUB_CREDENTIALS:-0}" = "0" ]; then
    rm -f -- "$TOKEN_FILE" "$ASKPASS"
  fi
}

reset_data_if_requested() {
  if [ "${RESET_DATA:-0}" != "1" ]; then return 0; fi
  log "RESET_DATA=1: removing old archives/metadata/github mirror/tmp work inside allowed roots"
  rm -rf -- "$REPO_DIR" "$TMP_ROOT/batches" "$TMP_ROOT/tmp" "$TMP_ROOT/replay_validation"
  mkdir -p "$ARCHIVE_DIR" "$META_DIR" "$LOG_DIR" "$TMP_ROOT/batches" "$TMP_ROOT/tmp" "$TMP_ROOT/runtime_home" "$TMP_ROOT/replay_validation" "$FS_ROOT/code" "$FS_ROOT/git_tmp"
  find "$ARCHIVE_DIR" -mindepth 1 -maxdepth 1 -type f -print0 2>/dev/null | xargs -0r rm -f --
  rm -f -- "$META_DIR/archive_index.jsonl" "$META_DIR/last_state_replay_validation.json"
}

main() {
  reset_data_if_requested
  write_repo_files
  cleanup_tmp_intermediates
  local ep start end batch_dir expected failures=0
  ep=$(next_episode)
  write_status "running" "$ep" "starting_from_${ep}"
  log "pipeline start total=$TOTAL_EPISODES batch_size=$BATCH_SIZE next_episode=$ep"
  sync_repo_and_push "startup" || true
  while (( ep < TOTAL_EPISODES )); do
    start=$ep
    end=$((start + BATCH_SIZE - 1))
    if (( end >= TOTAL_EPISODES )); then end=$((TOTAL_EPISODES - 1)); fi
    expected=$((end - start + 1))
    batch_dir="$TMP_ROOT/batches/batch_$(printf '%05d' "$start")_$(printf '%05d' "$end")"
    rm -rf -- "$batch_dir"
    mkdir -p "$batch_dir"
    log "batch $(printf '%05d' "$start")-$(printf '%05d' "$end") start"
    while (( ep <= end )); do
      write_status "running" "$ep" "generating_episode_$(printf '%05d' "$ep")"
      if ! run_episode "$ep" "$batch_dir"; then
        failures=$((failures + 1))
        if (( failures >= 3 )); then
          log "stopping after repeated failures"
          exit 30
        fi
        continue
      fi
      failures=0
      ep=$((ep + 1))
    done
    if ! archive_batch "$batch_dir" "$start" "$end" "$expected"; then write_status "failed" "$start" "archive_or_state_replay_validation_failed_$(printf '%05d' "$start")_$(printf '%05d' "$end")"; exit 31; fi
    write_status "running" "$ep" "archived_state_replay_validated_through_$(printf '%05d' "$end")"
    sync_repo_and_push "episodes_$(printf '%05d' "$start")_$(printf '%05d' "$end")" || true
  done
  write_status "finalizing" "$TOTAL_EPISODES" "generation_complete_push_final"
  for attempt in 1 2 3 4 5; do
    if sync_repo_and_push "final_attempt_${attempt}"; then
      write_status "complete" "$TOTAL_EPISODES" "complete_and_pushed"
      log "pipeline complete and pushed"
      finalize_credentials
      exit 0
    fi
    sleep $((attempt * 60))
  done
  write_status "complete_push_pending" "$TOTAL_EPISODES" "generation_complete_github_push_failed_after_retries"
  log "generation complete but final github push failed"
  exit 40
}

main "$@"