#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np


def _safe_rel(path: str) -> PurePosixPath:
    pp = PurePosixPath(str(path))
    if pp.is_absolute() or '..' in pp.parts or not str(path).strip():
        raise ValueError(f'unsafe relative path: {path}')
    return pp


def _load_obj(path: Path):
    vertices = []
    faces = []
    with Path(path).open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                idx = []
                for token in line.strip().split()[1:]:
                    raw = token.split('/')[0]
                    if not raw:
                        continue
                    j = int(raw)
                    idx.append(j - 1 if j > 0 else len(vertices) + j)
                if len(idx) >= 3:
                    for k in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[k], idx[k + 1]])
    if not vertices:
        vertices = [[-0.02, -0.02, -0.02], [0.02, -0.02, -0.02], [0.02, 0.02, -0.02], [-0.02, 0.02, -0.02], [-0.02, -0.02, 0.02], [0.02, -0.02, 0.02], [0.02, 0.02, 0.02], [-0.02, 0.02, 0.02]]
        faces = [[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]]
    if not faces:
        faces = []
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _quat_to_matrix(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _camera_metadata(name, cam, fallback_shape):
    h, w = int(fallback_shape[0]), int(fallback_shape[1])
    eye = np.asarray(getattr(cam, '_dataset_eye', [2.0, -3.0, 1.5]), dtype=np.float64)
    target = np.asarray(getattr(cam, '_dataset_target', [0.0, 0.0, 0.0]), dtype=np.float64)
    res = getattr(cam, '_dataset_resolution', (w, h))
    try:
        w, h = int(res[0]), int(res[1])
    except Exception:
        pass
    focal = getattr(cam, '_dataset_focal_length', None)
    if focal is None:
        try:
            focal = float(cam.prim.GetAttribute('focalLength').Get())
        except Exception:
            focal = 18.0
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1.0e-12)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    if float(np.linalg.norm(right)) < 1.0e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = right / max(float(np.linalg.norm(right)), 1.0e-12)
    up = np.cross(forward, right)
    up = up / max(float(np.linalg.norm(up)), 1.0e-12)
    # Isaac/Usd default horizontal aperture is about 20.955 mm. This records the
    # projection model used for generated 2D boxes and masks so consumers can replay it.
    fx = float(w) * float(focal) / 20.955
    fy = fx
    return {
        'name': str(name),
        'eye': eye,
        'target': target,
        'resolution': [int(w), int(h)],
        'focal_length_mm': float(focal),
        'horizontal_aperture_mm_assumed': 20.955,
        'fx_px': float(fx),
        'fy_px': float(fy),
        'cx_px': float(w) * 0.5,
        'cy_px': float(h) * 0.5,
        'forward': forward,
        'right': right,
        'up': up,
    }


def _project(points_w, meta):
    rel = np.asarray(points_w, dtype=np.float64) - meta['eye'].reshape(1, 3)
    z = rel @ meta['forward']
    x = rel @ meta['right']
    y = rel @ meta['up']
    z_safe = np.maximum(z, 1.0e-6)
    u = meta['cx_px'] + meta['fx_px'] * (x / z_safe)
    v = meta['cy_px'] - meta['fy_px'] * (y / z_safe)
    valid = z > 1.0e-4
    return np.stack([u, v], axis=-1), valid


def _rle_encode(mask):
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order='F')
    if pixels.size == 0:
        return [0]
    binary = (pixels != 0).astype(np.uint8, copy=False)
    changes = np.flatnonzero(binary[1:] != binary[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), changes.astype(np.int64)))
    ends = np.concatenate((changes.astype(np.int64), np.array([binary.size], dtype=np.int64)))
    lengths = ends - starts
    if int(binary[0]) == 1:
        lengths = np.concatenate((np.array([0], dtype=np.int64), lengths))
    return [int(x) for x in lengths.tolist()]


def _draw_label(cv2, img, text, org, color):
    x, y = int(org[0]), int(org[1])
    y = max(12, y)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tuple(int(c) for c in color), 1, cv2.LINE_AA)


def generate_object_2d_annotations_and_overlays(
    task_dir,
    task,
    cameras,
    video_frames,
    object_pos_w,
    object_quat_wxyz,
    object_visual_mesh_export_paths,
    object_names,
    object_categories,
    object_colors_rgba,
    video_stride,
    fps,
):
    import cv2

    task_dir = Path(task_dir)
    ann_dir = task_dir / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)
    object_pos_w = np.asarray(object_pos_w, dtype=np.float32)
    object_quat_wxyz = np.asarray(object_quat_wxyz, dtype=np.float32)
    colors = np.asarray(object_colors_rgba, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] < 3:
        colors = np.tile(np.array([[0.1, 0.8, 0.2, 1.0]], dtype=np.float32), (object_pos_w.shape[1], 1))
    n = int(object_pos_w.shape[1])
    stride = max(1, int(video_stride))
    sim_steps = list(range(0, int(object_pos_w.shape[0]), stride))
    meshes = []
    for rel in list(object_visual_mesh_export_paths):
        pp = _safe_rel(str(rel))
        meshes.append(_load_obj(task_dir.joinpath(*pp.parts)))
    views = {}
    bbox_paths = {}
    mask_paths = {}
    camera_intrinsics = {}
    frame_count_by_view = {}
    for view_name, cam in cameras.items():
        frames = list(video_frames.get(view_name, []))
        if not frames:
            continue
        h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
        meta = _camera_metadata(view_name, cam, (h, w))
        camera_intrinsics[view_name] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in meta.items() if k not in {'forward', 'right', 'up', 'eye', 'target'}} | {
            'eye': meta['eye'].tolist(),
            'target': meta['target'].tolist(),
            'projection_axes': {'forward': meta['forward'].tolist(), 'right': meta['right'].tolist(), 'up': meta['up'].tolist()},
        }
        f_count = min(len(frames), len(sim_steps))
        bbox_xyxy = np.zeros((f_count, n, 4), dtype=np.float32)
        bbox_valid = np.zeros((f_count, n), dtype=np.bool_)
        mask_area = np.zeros((f_count, n), dtype=np.uint32)
        mask_jsonl = ann_dir / f'{task}_{view_name}_mask_rle.jsonl.gz'
        annotated_frames = []
        with gzip.open(mask_jsonl, 'wt', encoding='utf-8') as gz:
            for fi in range(f_count):
                step = int(sim_steps[fi])
                img = np.asarray(frames[fi]).copy()
                frame_record = {'frame_index': int(fi), 'sim_step': step, 'view': str(view_name), 'objects': []}
                masks_for_frame = []
                for oi in range(n):
                    verts, faces = meshes[oi]
                    rot = _quat_to_matrix(object_quat_wxyz[step, oi])
                    pts_w = verts.astype(np.float64) @ rot.T + object_pos_w[step, oi].astype(np.float64).reshape(1, 3)
                    pts_2d, valid = _project(pts_w, meta)
                    good = valid & np.isfinite(pts_2d).all(axis=1)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    if int(good.sum()) >= 3:
                        xy = pts_2d[good]
                        x1 = float(np.clip(np.floor(xy[:, 0].min()), 0, w - 1))
                        y1 = float(np.clip(np.floor(xy[:, 1].min()), 0, h - 1))
                        x2 = float(np.clip(np.ceil(xy[:, 0].max()), 0, w - 1))
                        y2 = float(np.clip(np.ceil(xy[:, 1].max()), 0, h - 1))
                        if x2 > x1 and y2 > y1:
                            if len(faces):
                                for face in faces:
                                    if not np.all(good[face]):
                                        continue
                                    poly = np.rint(pts_2d[face]).astype(np.int32)
                                    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
                                    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
                                    if len(np.unique(poly, axis=0)) >= 3:
                                        cv2.fillConvexPoly(mask, poly, 1)
                            if int(mask.sum()) == 0:
                                hull = cv2.convexHull(np.rint(xy).astype(np.int32))
                                cv2.fillConvexPoly(mask, hull, 1)
                            area = int(mask.sum())
                            if area > 0:
                                ys, xs = np.where(mask > 0)
                                x1 = float(xs.min()); x2 = float(xs.max()); y1 = float(ys.min()); y2 = float(ys.max())
                                bbox_xyxy[fi, oi] = [x1, y1, x2, y2]
                                bbox_valid[fi, oi] = True
                                mask_area[fi, oi] = np.uint32(area)
                    counts = _rle_encode(mask)
                    frame_record['objects'].append({
                        'object_index': int(oi),
                        'name': str(object_names[oi]),
                        'category': str(object_categories[oi]),
                        'bbox_xyxy': [float(x) for x in bbox_xyxy[fi, oi].tolist()],
                        'bbox_valid': bool(bbox_valid[fi, oi]),
                        'mask': {'size': [int(h), int(w)], 'counts': counts, 'area': int(mask_area[fi, oi])},
                    })
                    masks_for_frame.append(mask)
                gz.write(json.dumps(frame_record, ensure_ascii=False, separators=(',', ':')) + '\n')
                for oi, mask in enumerate(masks_for_frame):
                    if not bbox_valid[fi, oi]:
                        continue
                    color = np.clip(colors[oi, :3] * 255.0, 0, 255).astype(np.uint8)
                    img[mask.astype(bool)] = (0.56 * img[mask.astype(bool)] + 0.44 * color.reshape(1, 3)).astype(np.uint8)
                    x1, y1, x2, y2 = [int(round(x)) for x in bbox_xyxy[fi, oi].tolist()]
                    cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c) for c in color.tolist()), 2)
                    _draw_label(cv2, img, f'{oi}:{object_categories[oi]}', (x1, y1 - 4), color.tolist())
                annotated_frames.append(img)
        video_frames[view_name] = annotated_frames
        bbox_npz = ann_dir / f'{task}_{view_name}_bbox_mask_stats.npz'
        np.savez_compressed(
            bbox_npz,
            schema=np.array('threedemo_object_2d_bbox_mask_view_v1'),
            view=np.array(str(view_name)),
            image_width=np.int32(w),
            image_height=np.int32(h),
            sim_step=np.asarray(sim_steps[:f_count], dtype=np.int32),
            object_2d_bbox_xyxy=bbox_xyxy,
            object_2d_bbox_valid=bbox_valid,
            object_pixel_mask_area=mask_area,
            object_name=np.asarray([str(x) for x in object_names]),
            object_category=np.asarray([str(x) for x in object_categories]),
            mask_rle_jsonl_gz=np.array(mask_jsonl.relative_to(task_dir).as_posix()),
        )
        rel_bbox = bbox_npz.relative_to(task_dir).as_posix()
        rel_mask = mask_jsonl.relative_to(task_dir).as_posix()
        bbox_paths[view_name] = rel_bbox
        mask_paths[view_name] = rel_mask
        frame_count_by_view[view_name] = int(f_count)
        views[view_name] = {
            'bbox_npz': rel_bbox,
            'mask_rle_jsonl_gz': rel_mask,
            'frame_count': int(f_count),
            'image_size_hw': [int(h), int(w)],
        }
    manifest = {
        'schema': 'threedemo_object_2d_bbox_mask_v1',
        'task': str(task),
        'contains_2d_bboxes': True,
        'contains_pixel_masks': True,
        'videos_include_2d_bbox_and_pixel_mask_overlay': True,
        'mask_encoding': 'COCO-style uncompressed RLE, column-major, gzip JSONL, one record per view frame',
        'bbox_format': 'xyxy_pixels_float32_in_image_coordinates',
        'annotation_method': 'projected per-episode visual mesh triangles from recorded PhysX pose; mask is the projected mesh silhouette used for video overlay',
        'camera_views': list(views.keys()),
        'frame_count_by_view': frame_count_by_view,
        'object_count': int(n),
        'object_names': [str(x) for x in object_names],
        'object_categories': [str(x) for x in object_categories],
        'camera_intrinsics': camera_intrinsics,
        'views': views,
    }
    save_path = ann_dir / 'annotation_manifest.json'
    save_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {
        'schema': manifest['schema'],
        'manifest_path': save_path.relative_to(task_dir).as_posix(),
        'camera_views': list(views.keys()),
        'bbox_paths': bbox_paths,
        'mask_paths': mask_paths,
        'frame_count_by_view': frame_count_by_view,
        'camera_intrinsics': camera_intrinsics,
        'contains_2d_bboxes': True,
        'contains_pixel_masks': True,
        'videos_include_2d_bbox_and_pixel_mask_overlay': True,
    }

