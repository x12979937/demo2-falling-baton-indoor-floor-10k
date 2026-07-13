#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

COMMON_REQUIRED = [
    'task','support_surface','dt','object_pose_w','object_pos_w','object_quat_wxyz',
    'object_lin_vel_w','object_ang_vel_w','object_mass_kg','object_inertia_diag_kg_m2',
    'object_center_of_mass_local_m','object_momentum_kg_m_s','object_angular_momentum_kg_m2_rad_s',
    'object_size_m','object_name','object_category','object_asset_source','object_visual_mesh_export_path',
    'object_collision_mesh_export_path','object_mesh_export_dir','object_mesh_export_format',
    'mesh_assets_manifest_path','initial_object_pose_w','initial_object_position_w',
    'initial_object_quat_wxyz','initial_object_lin_vel_w','initial_object_ang_vel_w','final_pose_w',
    'object_static_friction','object_dynamic_friction','object_restitution','object_text_description',
    'object_standard_description','scene_text_description','scene_standard_description',
    'state_replay_import_contract_json','camera_view_names','annotation_schema','annotation_manifest_path',
    'contains_2d_bboxes','contains_pixel_masks','videos_include_2d_bbox_and_pixel_mask_overlay',
    'object_2d_bbox_xyxy_path_json','object_pixel_mask_rle_path_json'
]
OPTIONAL_EXPORT = [
    'schema','physics_source','object_vel_w','object_center_of_mass_w','object_euler_xyz_rad',
    'object_roll_pitch_yaw_rad','object_tilt_angle_rad','object_rotation_angle_from_initial_rad',
    'object_trajectory_w','object_proxy_shape','object_visual_asset_path','object_collision_asset_path',
    'object_metadata_json','initial_object_euler_xyz_rad','initial_object_height_m',
    'sampled_initial_object_pose_w','sampled_initial_object_position_w','sampled_initial_object_quat_wxyz',
    'sampled_initial_object_euler_xyz_rad','sampled_initial_object_lin_vel_w','sampled_initial_object_ang_vel_w',
    'sampled_initial_object_height_m','object_long_axis_w',
    'object_color_rgba','disturbance_delta_lin_vel_w','disturbance_delta_ang_vel_w',
    'disturbance_force_w_N','disturbance_torque_w_Nm','bottom_z_minus_support_z',
    'bottom_z_minus_active_support_z','active_support_z','support_contact','floor_contact','table_contact',
    'pair_contact','first_support_contact_step','first_table_contact_step','first_ground_contact_after_leaving_table_step',
    'task_text_description','task_standard_description'
]
VIEW_MIN_COUNT = 6


def scalar(x):
    a = np.asarray(x)
    v = a.item() if a.shape == () else (a.reshape(-1)[0] if a.size else '')
    if isinstance(v, bytes):
        v = v.decode('utf-8', 'replace')
    return str(v)


def str_list(x):
    out = []
    for v in np.asarray(x).reshape(-1).tolist():
        if isinstance(v, bytes):
            v = v.decode('utf-8', 'replace')
        out.append(str(v))
    return out


def finite_array(d, name, tail=None, report=None):
    if name not in d.files:
        raise AssertionError(f'missing field {name}')
    a = np.asarray(d[name])
    if not np.issubdtype(a.dtype, np.number):
        raise AssertionError(f'{name} not numeric')
    if tail and tuple(a.shape[-len(tail):]) != tuple(tail):
        raise AssertionError(f'{name} shape {a.shape} does not end with {tail}')
    if not np.isfinite(a).all():
        raise AssertionError(f'{name} contains non-finite values')
    return a.astype(float, copy=False)


def safe_rel(base_dir, rel):
    rel = str(rel).strip()
    pp = PurePosixPath(rel)
    if not rel or pp.is_absolute() or '..' in pp.parts:
        raise AssertionError(f'unsafe relative path {rel!r}')
    return Path(base_dir).joinpath(*pp.parts)


def discover(path):
    p = Path(path)
    tmp = None
    if p.is_file() and p.name.endswith('.npz'):
        mf = p.with_name('manifest.json')
        return [(p.parent.parent.name or p.parent.name, p, mf if mf.exists() else None)], tmp
    if p.is_file() and p.name.endswith('.tar.gz'):
        base = Path(os.environ.get('TMPDIR', '/tmp'))
        base.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix='state_replay_validate_', dir=str(base)))
        with tarfile.open(p, 'r:gz') as tf:
            members = []
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if not (m.name.endswith('/dataset.npz') or m.name.endswith('/manifest.json') or '/mesh_assets/' in m.name or '/annotations/' in m.name):
                    continue
                pp = PurePosixPath(m.name)
                if pp.is_absolute() or '..' in pp.parts:
                    raise AssertionError(f'unsafe tar path {m.name}')
                members.append(m)
            tf.extractall(tmp, members)
        return [(ds.parent.parent.name or ds.parent.name, ds, ds.with_name('manifest.json') if ds.with_name('manifest.json').exists() else None) for ds in sorted(tmp.rglob('dataset.npz'))], tmp
    if p.is_dir():
        return [(ds.parent.parent.name or ds.parent.name, ds, ds.with_name('manifest.json') if ds.with_name('manifest.json').exists() else None) for ds in sorted(p.rglob('dataset.npz'))], tmp
    raise FileNotFoundError(str(path))


def validation_template(ds, task):
    return {
        'schema': 'state_replay_validation_v1',
        'episode': ds.parent.parent.name or ds.parent.name,
        'task': task,
        'engine': 'IsaacSim PhysX',
        'validation_scope': {
            'state_sequence_importable': False,
            'engine_replay_verified': False,
            'engine_replay_mode': 'state_sequence_playback_contract',
            'resimulate_from_initial_verified': False,
            'note': 'This Isaac/PhysX validator verifies that mesh assets and recorded rigid-body state can be imported for state-sequence playback, and checks recorded trajectory/contact/penetration/task constraints. It does not prove deterministic resimulation from only the initial state.'
        },
        'validation_pass': False,
        'failure_reasons': [],
        'thresholds': {
            'max_allowed_penetration_m': 0.05,
            'max_allowed_speed_m_s': 20.0,
            'max_allowed_angular_speed_rad_s': 150.0,
            'max_initial_position_error_m': 1e-4,
            'max_final_position_error_m': 1e-4,
            'max_quaternion_norm_error': 1e-3
        },
        'trajectory_error': {},
        'contact_error': {},
        'penetration': {},
        'task_constraints': {},
        'assets': {},
    }


def add_failure(report, reason):
    report['failure_reasons'].append(str(reason))


def validate(ds, mf=None, write_episode_validation=False):
    d = np.load(ds, allow_pickle=True)
    task = scalar(d['task']) if 'task' in d.files else 'unknown'
    report = validation_template(Path(ds), task)
    base_dir = Path(ds).parent
    manifest = None
    try:
        if mf and Path(mf).exists():
            manifest = json.loads(Path(mf).read_text(encoding='utf-8'))
        missing = [k for k in COMMON_REQUIRED if k not in d.files]
        if missing:
            raise AssertionError('missing required state fields: ' + ','.join(missing))
        pos = finite_array(d, 'object_pos_w', (3,))
        pose = finite_array(d, 'object_pose_w', (7,))
        quat = finite_array(d, 'object_quat_wxyz', (4,))
        lin = finite_array(d, 'object_lin_vel_w', (3,))
        ang = finite_array(d, 'object_ang_vel_w', (3,))
        if pos.ndim != 3 or pose.ndim != 3:
            raise AssertionError(f'state arrays must be [T,N,D], got {pos.shape} {pose.shape}')
        T, N = pos.shape[:2]
        if T <= 1 or N <= 0:
            raise AssertionError(f'invalid frames/object_count {T}/{N}')
        for name, arr in [('pose', pose), ('quat', quat), ('lin', lin), ('ang', ang)]:
            if arr.shape[:2] != (T, N):
                raise AssertionError(f'{name} shape mismatch {arr.shape}')
        init_pos = finite_array(d, 'initial_object_position_w', (3,))
        init_pose = finite_array(d, 'initial_object_pose_w', (7,))
        final_pose = finite_array(d, 'final_pose_w', (7,))
        mass = finite_array(d, 'object_mass_kg')
        inertia = finite_array(d, 'object_inertia_diag_kg_m2', (3,))
        com_local = finite_array(d, 'object_center_of_mass_local_m', (3,))
        size = finite_array(d, 'object_size_m', (3,))
        if init_pos.shape != (N, 3) or init_pose.shape != (N, 7) or final_pose.shape != (N, 7):
            raise AssertionError('initial/final state shape mismatch')
        if mass.shape != (N,) or inertia.shape != (N, 3) or com_local.shape != (N, 3) or size.shape != (N, 3):
            raise AssertionError('rigid-body parameter shape mismatch')
        if not (mass > 0).all() or not (inertia > 0).all() or not (size > 0).all():
            raise AssertionError('mass/inertia/size must be positive')
        dt = float(np.asarray(d['dt']).reshape(-1)[0])
        if (not np.isfinite(dt)) or dt <= 0:
            raise AssertionError('dt must be positive')
        support = scalar(d['support_surface'])
        if support != 'indoor_floor':
            add_failure(report, f'support_surface is {support}, expected indoor_floor')
        for k in ['object_static_friction', 'object_dynamic_friction', 'object_restitution']:
            v = finite_array(d, k)
            if v.shape != (N,) or not (v >= 0).all():
                raise AssertionError(f'{k} invalid')
        # Text, contract, and mesh importability.
        obj_text = str_list(d['object_text_description'])
        obj_std = str_list(d['object_standard_description'])
        if len(obj_text) != N or len(obj_std) != N or any(not x.strip() for x in obj_text):
            raise AssertionError('object descriptions invalid')
        for x in obj_std:
            j = json.loads(x)
            if 'mass_kg' not in j or 'collision' not in j or 'inertia_diag_kg_m2' not in j:
                raise AssertionError('object standard description incomplete')
        scene_std = json.loads(scalar(d['scene_standard_description']))
        contract = json.loads(scalar(d['state_replay_import_contract_json']))
        if int(scene_std.get('object_count', -1)) != N:
            raise AssertionError('scene object_count mismatch')
        if contract.get('required_engine') != 'IsaacSim PhysX':
            raise AssertionError('state replay contract must target IsaacSim PhysX')
        visual_meshes = str_list(d['object_visual_mesh_export_path'])
        collision_meshes = str_list(d['object_collision_mesh_export_path'])
        mesh_formats = str_list(d['object_mesh_export_format'])
        if not (len(visual_meshes) == len(collision_meshes) == len(mesh_formats) == N):
            raise AssertionError('mesh path count mismatch')
        mesh_files_ok = 0
        for rel in visual_meshes + collision_meshes:
            p = safe_rel(base_dir, rel)
            if not p.is_file() or p.stat().st_size <= 0:
                raise AssertionError(f'missing mesh asset {rel}')
            mesh_files_ok += 1
        if any(fmt.lower() != 'obj' for fmt in mesh_formats):
            raise AssertionError('mesh export format must be obj')
        mesh_manifest = safe_rel(base_dir, scalar(d['mesh_assets_manifest_path']))
        if not mesh_manifest.is_file() or mesh_manifest.stat().st_size <= 0:
            raise AssertionError('mesh assets manifest missing')
        mesh_manifest_json = json.loads(mesh_manifest.read_text(encoding='utf-8'))
        if int(mesh_manifest_json.get('object_count', -1)) != N:
            raise AssertionError('mesh assets manifest object_count mismatch')
        report['assets'] = {
            'mesh_manifest': scalar(d['mesh_assets_manifest_path']),
            'visual_mesh_count': len(visual_meshes),
            'collision_mesh_count': len(collision_meshes),
            'mesh_files_verified': mesh_files_ok,
            'mesh_format': 'obj'
        }
        # Annotation importability.
        views = str_list(d['camera_view_names'])
        if len(views) < VIEW_MIN_COUNT:
            raise AssertionError(f'expected at least {VIEW_MIN_COUNT} camera views, got {views}')
        if scalar(d['annotation_schema']) != 'threedemo_object_2d_bbox_mask_v1':
            raise AssertionError('unexpected annotation schema')
        if not bool(np.asarray(d['contains_2d_bboxes']).reshape(-1)[0]):
            raise AssertionError('contains_2d_bboxes false')
        if not bool(np.asarray(d['contains_pixel_masks']).reshape(-1)[0]):
            raise AssertionError('contains_pixel_masks false')
        ann_manifest = safe_rel(base_dir, scalar(d['annotation_manifest_path']))
        if not ann_manifest.is_file() or ann_manifest.stat().st_size <= 0:
            raise AssertionError('annotation manifest missing')
        bbox_paths = json.loads(scalar(d['object_2d_bbox_xyxy_path_json']))
        mask_paths = json.loads(scalar(d['object_pixel_mask_rle_path_json']))
        if set(bbox_paths) != set(views) or set(mask_paths) != set(views):
            raise AssertionError('annotation path view set mismatch')
        for view in views:
            bp = safe_rel(base_dir, bbox_paths[view])
            mp = safe_rel(base_dir, mask_paths[view])
            if not bp.is_file() or bp.stat().st_size <= 0:
                raise AssertionError(f'missing bbox npz for {view}')
            if not mp.is_file() or mp.stat().st_size <= 0:
                raise AssertionError(f'missing mask rle for {view}')
            b = np.load(bp, allow_pickle=True)
            if 'object_2d_bbox_xyxy' not in b.files or b['object_2d_bbox_xyxy'].shape[1:] != (N, 4):
                raise AssertionError(f'{view} bbox shape invalid')
            if not b['object_2d_bbox_valid'].any():
                raise AssertionError(f'{view} has no valid bboxes')
        report['validation_scope']['state_sequence_importable'] = True
        # Trajectory consistency.
        init_err = float(np.max(np.abs(pos[0] - init_pos)))
        final_err = float(np.max(np.abs(pos[-1] - final_pose[:, :3])))
        qnorm_err = float(np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1.0)))
        speeds = np.linalg.norm(lin, axis=-1)
        ang_speeds = np.linalg.norm(ang, axis=-1)
        report['trajectory_error'] = {
            'initial_position_error_m': init_err,
            'final_position_error_m': final_err,
            'max_quaternion_norm_error': qnorm_err,
            'max_linear_speed_m_s': float(np.max(speeds)),
            'max_angular_speed_rad_s': float(np.max(ang_speeds)),
        }
        if init_err > report['thresholds']['max_initial_position_error_m']:
            add_failure(report, f'initial position mismatch {init_err:.6g} m')
        if final_err > report['thresholds']['max_final_position_error_m']:
            add_failure(report, f'final position mismatch {final_err:.6g} m')
        if qnorm_err > report['thresholds']['max_quaternion_norm_error']:
            add_failure(report, f'quaternion norm error {qnorm_err:.6g}')
        if report['trajectory_error']['max_linear_speed_m_s'] > report['thresholds']['max_allowed_speed_m_s']:
            add_failure(report, f'linear speed too high {report["trajectory_error"]["max_linear_speed_m_s"]:.3f} m/s')
        if report['trajectory_error']['max_angular_speed_rad_s'] > report['thresholds']['max_allowed_angular_speed_rad_s']:
            add_failure(report, f'angular speed too high {report["trajectory_error"]["max_angular_speed_rad_s"]:.3f} rad/s')
        # Contact and penetration. Positive penetration means below active support by that many meters.
        support_gap = None
        for k in ('bottom_z_minus_active_support_z', 'bottom_z_minus_support_z'):
            if k in d.files:
                support_gap = finite_array(d, k)
                break
        if support_gap is not None:
            min_gap = float(np.min(support_gap))
            max_pen = max(0.0, -min_gap)
        else:
            min_gap = None
            max_pen = None
        contact = None
        for k in ('support_contact', 'floor_contact'):
            if k in d.files:
                contact = np.asarray(d[k]).astype(bool)
                break
        contact_objects = int(contact.any(axis=0).sum()) if contact is not None and contact.ndim >= 2 else 0
        contact_mismatch = 0
        if support_gap is not None and contact is not None and contact.shape == support_gap.shape:
            contact_mismatch = int(np.logical_and(contact, support_gap > 0.25).sum())
        report['penetration'] = {
            'min_bottom_z_minus_support_m': min_gap,
            'max_penetration_m': max_pen,
        }
        report['contact_error'] = {
            'objects_with_recorded_contact': contact_objects,
            'contact_without_near_surface_count': contact_mismatch,
        }
        if max_pen is not None and max_pen > report['thresholds']['max_allowed_penetration_m']:
            add_failure(report, f'penetration too large {max_pen:.6g} m')
        if contact_mismatch > 0:
            add_failure(report, f'contact flag inconsistent with support gap in {contact_mismatch} samples')
        # Task constraints.
        names = str_list(d['object_name']) if 'object_name' in d.files else []
        task_ok = True
        constraints = {'object_count': int(N), 'support_surface': support, 'camera_view_count': len(views)}
        if task == 'falling_baton':
            constraints['expected_object_count'] = 8
            constraints['all_objects_reached_floor_contact'] = (contact_objects == N == 8)
            if N != 8:
                add_failure(report, f'falling_baton expected 8 rods, got {N}')
            if contact_objects != N:
                add_failure(report, f'only {contact_objects}/{N} objects have floor/support contact')
        elif task == 'rolling_tabletop':
            constraints['expected_object_count'] = 1
            constraints['single_object_episode'] = (N == 1)
            constraints['object_category'] = str_list(d['object_category'])[0] if 'object_category' in d.files else ''
            constraints['asset_source'] = str_list(d['object_asset_source'])[0] if 'object_asset_source' in d.files else ''
            if N != 1:
                add_failure(report, f'rolling_tabletop expected 1 object, got {N}')
            if contact_objects < 1:
                add_failure(report, 'rolling object never contacted support surface')
        else:
            add_failure(report, f'unknown task {task}')
        if manifest:
            constraints['manifest_ok'] = bool(manifest.get('ok'))
            if manifest.get('ok') is not True:
                add_failure(report, 'manifest ok is not true')
        report['task_constraints'] = constraints
        report['validation_pass'] = bool(report['validation_scope']['state_sequence_importable'] and not report['failure_reasons'])
    except Exception as e:
        add_failure(report, str(e))
    if write_episode_validation:
        (Path(ds).parent / 'state_replay_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def export_state(ds, mf, out_root, episode, report):
    d = np.load(ds, allow_pickle=True)
    out = Path(out_root) / episode
    out.mkdir(parents=True, exist_ok=True)
    fields = {k: d[k] for k in list(dict.fromkeys(COMMON_REQUIRED + OPTIONAL_EXPORT)) if k in d.files}
    fields['replay_state_schema'] = np.array('threedemo_replay_state_v3')
    fields['replay_state_export_note'] = np.array('Import into IsaacSim PhysX for state-sequence playback: instantiate each rigid body from collision mesh/proxy, assign mass/COM/inertia/material, then set recorded pose and velocity at each frame. Use state_replay_validation.json to check consistency and constraints.')
    npz = out / 'replay_state.npz'
    np.savez_compressed(npz, **fields)
    manifest = {
        'schema': 'threedemo_replay_state_v3',
        'episode': episode,
        'source_dataset': str(ds),
        'source_manifest': str(mf) if mf else None,
        'state_replay_validation_json': str((Path(ds).parent / 'state_replay_validation.json').name),
        'validation_pass': bool(report.get('validation_pass')),
        'task': report.get('task'),
        'engine': 'IsaacSim PhysX',
        'replay_mode': report['validation_scope']['engine_replay_mode'],
        'files': {'replay_state_npz': npz.name},
    }
    mp = out / 'replay_state_manifest.json'
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'episode': episode, 'state_npz': str(npz), 'manifest': str(mp), 'validation_pass': bool(report.get('validation_pass'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--export-dir', default=None)
    ap.add_argument('--write-episode-validation', action='store_true')
    ap.add_argument('--allow-fail', action='store_true', help='write reports and exit zero even if validation_pass is false')
    args = ap.parse_args()
    tmp = None
    try:
        items, tmp = discover(args.input)
        if args.limit and args.limit > 0:
            items = items[:args.limit]
        if not items:
            raise AssertionError('no dataset.npz files found')
        reports, exports = [], []
        for ep, ds, mf in items:
            report = validate(ds, mf, write_episode_validation=args.write_episode_validation or bool(args.export_dir))
            report['episode'] = ep
            reports.append(report)
            if args.export_dir:
                exports.append(export_state(ds, mf, args.export_dir, ep, report))
        ok = all(bool(r.get('validation_pass')) for r in reports)
        payload = {
            'schema': 'state_replay_validation_batch_v1',
            'ok': ok,
            'checked': len(reports),
            'failed': sum(1 for r in reports if not r.get('validation_pass')),
            'results': reports,
            'exports': exports,
        }
        text = json.dumps(payload, ensure_ascii=False, separators=(',', ':')) if args.json else json.dumps(payload, ensure_ascii=False, indent=2)
        print(text)
        return 0 if (ok or args.allow_fail) else 1
    except Exception as e:
        payload = {'schema': 'state_replay_validation_batch_v1', 'ok': False, 'checked': 0, 'failed': 1, 'error': str(e)}
        print(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) if args.json else json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if args.allow_fail else 1
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
