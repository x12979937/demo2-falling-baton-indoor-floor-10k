#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_norm(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if n < 1e-12 else q / n


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return quat_norm(np.r_[math.cos(angle * 0.5), axis * math.sin(angle * 0.5)])


def quat_rotate(q, v):
    q = quat_norm(q)
    return quat_mul(quat_mul(q, np.r_[0.0, v]), np.array([q[0], -q[1], -q[2], -q[3]]))[1:]


def quat_to_euler_xyz(q):
    w, x, y, z = quat_norm(q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def look_at_quat_world(eye, target):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(up, forward)
    right = right / max(np.linalg.norm(right), 1e-12)
    up2 = np.cross(forward, right)
    rot = np.stack([forward, right, up2], axis=1)
    tr = np.trace(rot)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(rot)))
        if i == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            q = [(rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            q = [(rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            q = [(rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s]
    return quat_norm(np.asarray(q, dtype=np.float64))


def capsule_bottom_delta(pos, quat, radius, height, table_z):
    axis = quat_rotate(quat, [0.0, 0.0, 1.0])
    az = abs(float(axis[2]))
    support = 0.5 * max(0.0, height - 2.0 * radius) * az + radius
    return float(pos[2] - support - table_z)


def box_bottom_delta(pos, quat, size, table_z):
    axes = [quat_rotate(quat, np.eye(3)[i]) for i in range(3)]
    support = sum(abs(float(a[2])) * float(s) * 0.5 for a, s in zip(axes, size))
    return float(pos[2] - support - table_z)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def camera_frame(camera, world, app):
    try:
        world.render()
    except Exception:
        pass
    app.update()
    rgba = camera.get_rgba()
    if rgba is None:
        return None
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return arr if arr.std() > 0.5 else None


def encode_video(path: Path, frames, fps: int, keep_frames: bool = False):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = path.parent / f"{path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i, frame in enumerate(frames):
            cv2.imwrite(str(frames_dir / f"frame_{i:05d}.png"), cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        ffmpeg = shutil.which("ffmpeg") or "/root/ffmpeg-7.1-build/bin/ffmpeg"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)


def add_camera(Camera, path, name, eye, target, resolution, focal_length):
    cam = Camera(
        prim_path=path,
        name=name,
        position=np.asarray(eye, dtype=np.float64),
        orientation=look_at_quat_world(eye, target),
        resolution=resolution,
        frequency=30,
    )
    cam.prim.GetAttribute("focalLength").Set(float(focal_length))
    cam._dataset_eye = np.asarray(eye, dtype=np.float64)
    cam._dataset_target = np.asarray(target, dtype=np.float64)
    cam._dataset_resolution = (int(resolution[0]), int(resolution[1]))
    cam._dataset_focal_length = float(focal_length)
    return cam


def setup_world(
    World,
    FixedCuboid,
    PhysicsMaterial,
    UsdGeom,
    UsdLux,
    table_z,
    table_static_friction=0.88,
    table_dynamic_friction=0.65,
    table_scale=(1.35, 0.92, 0.05),
):
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    stage = world.stage
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    table_mat = PhysicsMaterial(
        "/World/PhysicsMaterials/table",
        static_friction=float(table_static_friction),
        dynamic_friction=float(table_dynamic_friction),
        restitution=0.08,
    )
    world.scene.add_default_ground_plane()
    FixedCuboid(
        "/World/Table",
        name="table",
        position=np.array([0.62, 0.0, table_z - 0.025]),
        scale=np.array(table_scale, dtype=np.float64),
        color=np.array([0.54, 0.43, 0.31]),
        physics_material=table_mat,
    )
    FixedCuboid(
        "/World/BackWall",
        name="backwall",
        position=np.array([0.62, 0.49, 0.68]),
        scale=np.array([1.45, 0.035, 0.76]),
        color=np.array([0.74, 0.77, 0.80]),
    )
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(1250.0)
    fill = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    fill.CreateIntensityAttr(320.0)
    return world


def write_indoor_floor_texture(texture_path):
    texture_path = Path(texture_path)
    if texture_path.exists():
        return str(texture_path)
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        h, w = 512, 512
        yy, xx = np.mgrid[0:h, 0:w]
        plank = ((xx // 64) % 2) * 18 + ((yy // 128) % 2) * 10
        grain = (10.0 * np.sin(xx / 17.0) + 6.0 * np.sin((xx + yy) / 31.0)).astype(np.float32)
        base = np.zeros((h, w, 3), dtype=np.float32)
        base[..., 0] = 146 + plank + grain
        base[..., 1] = 123 + plank * 0.55 + grain * 0.45
        base[..., 2] = 88 + plank * 0.28 + grain * 0.25
        # Dark seams make the floor read as an indoor tiled/laminate surface in RTX renders.
        seam = ((xx % 64) < 3) | ((yy % 128) < 3)
        base[seam] *= 0.55
        img = np.clip(base, 0, 255).astype(np.uint8)
        cv2.imwrite(str(texture_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return str(texture_path)
    except Exception:
        return None


def bind_texture_material(stage, prim, texture_path):
    if not texture_path:
        return False
    try:
        from pxr import Sdf, UsdShade

        mat = UsdShade.Material.Define(stage, "/World/Materials/Demo2IndoorFloor")
        shader = UsdShade.Shader.Define(stage, "/World/Materials/Demo2IndoorFloor/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.78)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        tex = UsdShade.Shader.Define(stage, "/World/Materials/Demo2IndoorFloor/FloorTexture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
        st_reader = UsdShade.Shader.Define(stage, "/World/Materials/Demo2IndoorFloor/StReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(mat)
        return True
    except Exception:
        return False


def create_textured_floor_mesh(stage, UsdGeom, path, center, size, z, texture_path):
    from pxr import Gf, Sdf, Vt

    cx, cy = float(center[0]), float(center[1])
    sx, sy = float(size[0]) * 0.5, float(size[1]) * 0.5
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(cx - sx, cy - sy, float(z)),
            Gf.Vec3f(cx + sx, cy - sy, float(z)),
            Gf.Vec3f(cx + sx, cy + sy, float(z)),
            Gf.Vec3f(cx - sx, cy + sy, float(z)),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.52, 0.38)])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(8.0, 0.0), Gf.Vec2f(8.0, 5.5), Gf.Vec2f(0.0, 5.5)]))
    bind_texture_material(stage, mesh.GetPrim(), texture_path)
    return mesh


def setup_falling_floor_world(World, FixedCuboid, PhysicsMaterial, UsdGeom, UsdLux, floor_z=0.0):
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    stage = world.stage
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    try:
        world.scene.add_default_ground_plane()
    except Exception:
        pass
    floor_mat = PhysicsMaterial(
        "/World/PhysicsMaterials/indoor_floor",
        static_friction=0.86,
        dynamic_friction=0.58,
        restitution=0.10,
    )
    floor_center = np.array([0.62, 0.0, floor_z - 0.010])
    floor_size = np.array([4.20, 3.00, 0.020], dtype=np.float64)
    FixedCuboid(
        "/World/FloorCollider",
        name="indoor_floor_collider",
        position=floor_center,
        scale=floor_size,
        color=np.array([0.58, 0.50, 0.39]),
        physics_material=floor_mat,
    )
    texture_path = write_indoor_floor_texture(
        "/root/autodl-tmp/mingyu/IsaacLab/Projects/ThreeDemo/assets/textures/demo2_indoor_floor_laminate.png"
    )
    create_textured_floor_mesh(stage, UsdGeom, "/World/IndoorFloorTexture", [0.62, 0.0], [4.20, 3.00], floor_z + 0.0015, texture_path)
    _visual_prim(stage, UsdGeom, "/World/InteriorBackWall", "cube", [0.78, 0.79, 0.76], translate=(0.62, 1.50, 0.82), scale=(4.20, 0.035, 1.64))
    _visual_prim(stage, UsdGeom, "/World/InteriorLeftWall", "cube", [0.72, 0.75, 0.74], translate=(-1.48, 0.0, 0.82), scale=(0.035, 3.00, 1.64))
    _visual_prim(stage, UsdGeom, "/World/InteriorRightWall", "cube", [0.72, 0.75, 0.74], translate=(2.72, 0.0, 0.82), scale=(0.035, 3.00, 1.64))
    _visual_prim(stage, UsdGeom, "/World/BackBaseboard", "cube", [0.48, 0.43, 0.35], translate=(0.62, 1.46, 0.055), scale=(4.18, 0.030, 0.070))
    _visual_prim(stage, UsdGeom, "/World/LeftBaseboard", "cube", [0.48, 0.43, 0.35], translate=(-1.44, 0.0, 0.055), scale=(0.030, 2.90, 0.070))
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(1800.0)
    fill = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    fill.CreateIntensityAttr(450.0)
    return world


def _visual_prim(stage, UsdGeom, path, kind, color, translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), radius=None, height=None):
    from pxr import Gf

    if kind == "cube":
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.0)
    elif kind == "sphere":
        geom = UsdGeom.Sphere.Define(stage, path)
        geom.CreateRadiusAttr(float(radius if radius is not None else 0.5))
    elif kind == "cylinder":
        geom = UsdGeom.Cylinder.Define(stage, path)
        geom.CreateRadiusAttr(float(radius if radius is not None else 0.5))
        geom.CreateHeightAttr(float(height if height is not None else 1.0))
    else:
        return None
    geom.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    xf = UsdGeom.Xformable(geom.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3f(float(translate[0]), float(translate[1]), float(translate[2])))
    xf.AddScaleOp().Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))
    return geom


def attach_category_visual(stage, UsdGeom, prim_path, category, size, color):
    """Attach lightweight visual-only details under the rigid proxy body.

    The parent prim remains the PhysX rigid body/collision proxy. These children
    are rendered with the body transform, but do not add unstable mesh collision.
    """
    sx, sy, sz = [float(x) for x in size]
    base = f"{prim_path}/VisualDetail"
    if category in {"bottle", "can"}:
        body_h = sz * 0.72
        neck_h = sz * 0.25
        _visual_prim(stage, UsdGeom, f"{base}/body", "cylinder", color, translate=(0, 0, -sz * 0.05), radius=max(sx, sy) * 0.42, height=body_h)
        _visual_prim(stage, UsdGeom, f"{base}/neck", "cylinder", (0.86, 0.92, 0.98), translate=(0, 0, body_h * 0.42), radius=max(sx, sy) * 0.23, height=neck_h)
        _visual_prim(stage, UsdGeom, f"{base}/cap", "cylinder", (0.08, 0.12, 0.16), translate=(0, 0, sz * 0.49), radius=max(sx, sy) * 0.26, height=sz * 0.08)
    elif category == "cup":
        _visual_prim(stage, UsdGeom, f"{base}/outer", "cylinder", color, radius=max(sx, sy) * 0.46, height=sz)
        _visual_prim(stage, UsdGeom, f"{base}/inside", "cylinder", (0.08, 0.07, 0.09), translate=(0, 0, sz * 0.04), radius=max(sx, sy) * 0.34, height=sz * 0.86)
        _visual_prim(stage, UsdGeom, f"{base}/handle", "cube", (0.92, 0.88, 0.96), translate=(sx * 0.42, 0, 0.0), scale=(sx * 0.12, sy * 0.08, sz * 0.42))
    elif category == "bowl":
        _visual_prim(stage, UsdGeom, f"{base}/outer", "sphere", color, scale=(sx * 0.52, sy * 0.52, sz * 0.28), radius=1.0)
        _visual_prim(stage, UsdGeom, f"{base}/inner_shadow", "sphere", (0.10, 0.08, 0.06), translate=(0, 0, sz * 0.08), scale=(sx * 0.40, sy * 0.40, sz * 0.16), radius=1.0)
    elif category == "plate":
        _visual_prim(stage, UsdGeom, f"{base}/plate", "cylinder", color, radius=max(sx, sy) * 0.48, height=max(0.006, sz * 0.55))
        _visual_prim(stage, UsdGeom, f"{base}/rim", "cylinder", (0.96, 0.95, 0.88), translate=(0, 0, sz * 0.25), radius=max(sx, sy) * 0.51, height=max(0.004, sz * 0.22))
    elif category == "hammer":
        _visual_prim(stage, UsdGeom, f"{base}/handle", "cube", (0.54, 0.32, 0.14), scale=(sx * 0.46, sy * 0.18, sz * 0.28))
        _visual_prim(stage, UsdGeom, f"{base}/head", "cube", (0.12, 0.12, 0.12), translate=(sx * 0.38, 0, sz * 0.08), scale=(sx * 0.18, sy * 0.55, sz * 0.42))
    elif category == "tray":
        _visual_prim(stage, UsdGeom, f"{base}/base", "cube", color, scale=(sx, sy, sz * 0.45))
        _visual_prim(stage, UsdGeom, f"{base}/rim_l", "cube", (0.20, 0.16, 0.11), translate=(0, sy * 0.46, sz * 0.28), scale=(sx, sy * 0.06, sz * 0.7))
        _visual_prim(stage, UsdGeom, f"{base}/rim_r", "cube", (0.20, 0.16, 0.11), translate=(0, -sy * 0.46, sz * 0.28), scale=(sx, sy * 0.06, sz * 0.7))


def make_follow_visual(stage, UsdGeom, root_path, category, size, color):
    from pxr import Gf

    if category not in {"bottle", "bowl", "plate", "hammer", "tray", "cup", "can"}:
        return None
    root = UsdGeom.Xform.Define(stage, root_path)
    xf = UsdGeom.Xformable(root.GetPrim())
    translate_op = xf.AddTranslateOp()
    orient_op = xf.AddOrientOp()
    attach_category_visual(stage, UsdGeom, root_path, category, size, color)
    translate_op.Set(Gf.Vec3f(0.0, 0.0, -10.0))
    orient_op.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return translate_op, orient_op


def robotwin_asset_paths(object_id: str, variant: int | None = 0):
    base = Path("/autodl-fs/data/mingyu/robotwin2.0/robotwin_work/RoboTwin/assets/objects") / object_id
    if variant is None:
        return str(base / "visual"), str(base / "collision")
    visual = base / "visual" / f"base{variant}.glb"
    collision = base / "collision" / f"base{variant}.glb"
    return str(visual if visual.exists() else base / "visual"), str(collision if collision.exists() else base / "collision")


def robotwin_available_variants(object_id: str):
    base = Path("/autodl-fs/data/mingyu/robotwin2.0/robotwin_work/RoboTwin/assets/objects") / object_id
    visual = {p.stem.replace("base", "") for p in (base / "visual").glob("base*.glb")}
    collision = {p.stem.replace("base", "") for p in (base / "collision").glob("base*.glb")}
    variants = sorted(int(x) for x in visual.intersection(collision) if x.isdigit())
    return variants


def load_mesh_as_fit_vertices(asset_path: str, target_size):
    import trimesh

    path = Path(asset_path)
    if path.is_dir():
        candidates = sorted(path.glob("base*.glb"))
        if not candidates:
            candidates = sorted(path.glob("*.glb"))
        if not candidates:
            raise FileNotFoundError(f"No GLB mesh in {path}")
        path = candidates[0]
    mesh = trimesh.load(str(path), force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Empty mesh: {path}")
    bounds = np.vstack([vertices.min(axis=0), vertices.max(axis=0)])
    center = bounds.mean(axis=0)
    extents = np.maximum(bounds[1] - bounds[0], 1e-9)
    target = np.asarray(target_size, dtype=np.float64)
    scale = target / extents
    fitted = (vertices - center) * scale
    return fitted, faces, str(path), extents.tolist(), scale.tolist()


def load_collision_mesh_as_fit_vertices(collision_asset_path: str, target_size):
    return load_mesh_as_fit_vertices(collision_asset_path, target_size)



def dataset_mesh_safe_name(value):
    text = str(value or "object")
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    out = "".join(keep).strip("_")
    return out[:80] or "object"


def dataset_mesh_write_obj(path: Path, vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"empty mesh export for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# threedemo per-episode mesh export\n")
        for v in vertices:
            f.write(f"v {float(v[0]):.9g} {float(v[1]):.9g} {float(v[2]):.9g}\n")
        for face in faces:
            a, b, c = [int(x) + 1 for x in face]
            f.write(f"f {a} {b} {c}\n")
    return int(len(vertices)), int(len(faces))


def dataset_mesh_box(size, center=(0.0, 0.0, 0.0)):
    sx, sy, sz = [float(x) for x in size]
    cx, cy, cz = [float(x) for x in center]
    xs = [cx - sx * 0.5, cx + sx * 0.5]
    ys = [cy - sy * 0.5, cy + sy * 0.5]
    zs = [cz - sz * 0.5, cz + sz * 0.5]
    v = np.array([
        [xs[0], ys[0], zs[0]], [xs[1], ys[0], zs[0]], [xs[1], ys[1], zs[0]], [xs[0], ys[1], zs[0]],
        [xs[0], ys[0], zs[1]], [xs[1], ys[0], zs[1]], [xs[1], ys[1], zs[1]], [xs[0], ys[1], zs[1]],
    ], dtype=np.float64)
    f = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]], dtype=np.int64)
    return v, f


def dataset_mesh_uv_sphere(size, segments=24, rings=12):
    sx, sy, sz = [float(x) for x in size]
    vertices = []
    for r in range(rings + 1):
        theta = math.pi * r / rings
        z = math.cos(theta) * sz * 0.5
        rr = math.sin(theta)
        for s in range(segments):
            phi = 2.0 * math.pi * s / segments
            vertices.append([math.cos(phi) * sx * 0.5 * rr, math.sin(phi) * sy * 0.5 * rr, z])
    faces = []
    for r in range(rings):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + (s + 1) % segments
            c = (r + 1) * segments + (s + 1) % segments
            d = (r + 1) * segments + s
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def dataset_mesh_cylinder(size, segments=32):
    sx, sy, sz = [float(x) for x in size]
    z0, z1 = -sz * 0.5, sz * 0.5
    vertices = []
    for z in (z0, z1):
        for s in range(segments):
            phi = 2.0 * math.pi * s / segments
            vertices.append([math.cos(phi) * sx * 0.5, math.sin(phi) * sy * 0.5, z])
    bottom_center = len(vertices); vertices.append([0.0, 0.0, z0])
    top_center = len(vertices); vertices.append([0.0, 0.0, z1])
    faces = []
    for s in range(segments):
        a = s; b = (s + 1) % segments; c = segments + (s + 1) % segments; d = segments + s
        faces.append([a, b, c]); faces.append([a, c, d])
        faces.append([bottom_center, b, a]); faces.append([top_center, d, c])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def dataset_mesh_capsule(size, segments=24, hemi_rings=6):
    sx, sy, sz = [float(x) for x in size]
    radius = max(1e-5, min(sx, sy) * 0.5)
    cyl_half = max(0.0, sz - 2.0 * radius) * 0.5
    ring_defs = []
    for k in range(hemi_rings + 1):
        theta = -math.pi * 0.5 + (math.pi * 0.5) * k / hemi_rings
        ring_defs.append((-cyl_half + radius * math.sin(theta), radius * math.cos(theta)))
    for k in range(1, hemi_rings + 1):
        theta = (math.pi * 0.5) * k / hemi_rings
        ring_defs.append((cyl_half + radius * math.sin(theta), radius * math.cos(theta)))
    vertices = []
    for z, rr in ring_defs:
        for s in range(segments):
            phi = 2.0 * math.pi * s / segments
            vertices.append([math.cos(phi) * rr, math.sin(phi) * rr, z])
    faces = []
    for r in range(len(ring_defs) - 1):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + (s + 1) % segments
            c = (r + 1) * segments + (s + 1) % segments
            d = (r + 1) * segments + s
            faces.append([a, b, c]); faces.append([a, c, d])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def dataset_mesh_tetrahedron(size):
    sx, sy, sz = [float(x) for x in size]
    v = np.array([[0.0, 0.0, sz * 0.5], [-sx * 0.5, -sy * 0.5, -sz * 0.5], [sx * 0.5, -sy * 0.5, -sz * 0.5], [0.0, sy * 0.5, -sz * 0.5]], dtype=np.float64)
    f = np.array([[0,1,2],[0,2,3],[0,3,1],[1,3,2]], dtype=np.int64)
    return v, f


def dataset_mesh_l_shape(size):
    sx, sy, sz = [float(x) for x in size]
    v1, f1 = dataset_mesh_box([sx, sy * 0.42, sz], center=(0.0, -sy * 0.29, 0.0))
    v2, f2 = dataset_mesh_box([sx * 0.42, sy, sz], center=(-sx * 0.29, 0.0, 0.0))
    return np.vstack([v1, v2]), np.vstack([f1, f2 + len(v1)])


def dataset_mesh_from_proxy(shape, category, size):
    shape = str(shape or "").lower()
    category = str(category or "").lower()
    if shape == "sphere" or category == "sphere":
        return dataset_mesh_uv_sphere(size)
    if shape == "capsule" or category in {"baton", "rod", "markpen"}:
        return dataset_mesh_capsule(size)
    if category == "flat_cylinder" or "cylinder" in shape:
        return dataset_mesh_cylinder(size)
    if category == "tetrahedron" or "tetra" in shape:
        return dataset_mesh_tetrahedron(size)
    if category == "l_shape" or "l_shape" in shape:
        return dataset_mesh_l_shape(size)
    return dataset_mesh_box(size)


def export_episode_mesh_assets(task_dir: Path, meta, shapes, sizes):
    mesh_root = task_dir / "mesh_assets"
    mesh_root.mkdir(parents=True, exist_ok=True)
    objects = []
    visual_paths, collision_paths, object_dirs, formats = [], [], [], []
    for i, item in enumerate(meta):
        name = dataset_mesh_safe_name(item.get("name", f"object_{i:02d}"))
        category = str(item.get("category", ""))
        shape = str(shapes[i])
        size = np.asarray(sizes[i], dtype=np.float64)
        obj_dir = mesh_root / f"object_{i:02d}_{name}"
        obj_dir.mkdir(parents=True, exist_ok=True)
        visual_vertices = visual_faces = collision_vertices = collision_faces = None
        visual_source = str(item.get("resolved_visual_asset_path") or item.get("visual_asset_path") or "")
        collision_source = str(item.get("collision_asset_path") or "")
        if shape.startswith("robotwin_collision_mesh"):
            try:
                collision_vertices, collision_faces, resolved_collision, _, _ = load_collision_mesh_as_fit_vertices(collision_source, size)
                collision_source = resolved_collision
            except Exception:
                collision_vertices, collision_faces = dataset_mesh_from_proxy(shape, category, size)
            try:
                visual_vertices, visual_faces, resolved_visual, _, _ = load_mesh_as_fit_vertices(visual_source, size)
                visual_source = resolved_visual
            except Exception:
                visual_vertices, visual_faces = collision_vertices, collision_faces
        elif shape.startswith("procedural_mesh") and "procedural_mesh_vertices_faces" in globals():
            kind = item.get("procedural_mesh_kind", item.get("mesh_kind", category))
            visual_vertices, visual_faces = procedural_mesh_vertices_faces(kind, size)
            collision_vertices, collision_faces = visual_vertices, visual_faces
        else:
            visual_vertices, visual_faces = dataset_mesh_from_proxy(shape, category, size)
            collision_vertices, collision_faces = visual_vertices, visual_faces
        visual_path = obj_dir / "visual_mesh.obj"
        collision_path = obj_dir / "collision_mesh.obj"
        vv, vf = dataset_mesh_write_obj(visual_path, visual_vertices, visual_faces)
        cv, cf = dataset_mesh_write_obj(collision_path, collision_vertices, collision_faces)
        rel_visual = visual_path.relative_to(task_dir).as_posix()
        rel_collision = collision_path.relative_to(task_dir).as_posix()
        rel_dir = obj_dir.relative_to(task_dir).as_posix()
        item["visual_mesh_export_path"] = rel_visual
        item["collision_mesh_export_path"] = rel_collision
        item["mesh_export_dir"] = rel_dir
        item["mesh_export_format"] = "obj"
        objects.append({
            "index": int(i),
            "name": str(item.get("name", "")),
            "category": category,
            "source": str(item.get("source", "")),
            "proxy_shape": shape,
            "size_m": [float(x) for x in size.tolist()],
            "visual_mesh": {"path": rel_visual, "format": "obj", "vertex_count": vv, "face_count": vf, "source_asset_path": visual_source},
            "collision_mesh": {"path": rel_collision, "format": "obj", "vertex_count": cv, "face_count": cf, "source_asset_path": collision_source},
        })
        visual_paths.append(rel_visual)
        collision_paths.append(rel_collision)
        object_dirs.append(rel_dir)
        formats.append("obj")
    manifest = {
        "schema": "threedemo_episode_mesh_assets_v1",
        "description": "Per-episode object meshes used by this simulation. Paths are relative to the episode task directory containing dataset.npz.",
        "object_count": int(len(objects)),
        "objects": objects,
    }
    save_json(mesh_root / "mesh_assets_manifest.json", manifest)
    return {
        "manifest_path": "mesh_assets/mesh_assets_manifest.json",
        "objects": objects,
        "visual_paths": visual_paths,
        "collision_paths": collision_paths,
        "object_dirs": object_dirs,
        "formats": formats,
    }


def mesh_bottom_delta(pos, quat, local_vertices, table_z):
    rot_z = np.array(
        [
            quat_rotate(quat, [1.0, 0.0, 0.0])[2],
            quat_rotate(quat, [0.0, 1.0, 0.0])[2],
            quat_rotate(quat, [0.0, 0.0, 1.0])[2],
        ],
        dtype=np.float64,
    )
    min_local_z = float(np.min(np.asarray(local_vertices, dtype=np.float64) @ rot_z))
    return float(pos[2] + min_local_z - table_z)


def create_robotwin_mesh_rigid(
    world,
    UsdGeom,
    prim_path,
    name,
    position,
    orientation,
    vertices,
    faces,
    visual_vertices,
    visual_faces,
    color,
    physics_material,
    mass,
    diagonal_inertia,
    center_of_mass,
    linear_velocity,
    angular_velocity,
):
    try:
        from isaacsim.core.prims import SingleRigidPrim as RigidPrim
    except Exception:
        from omni.isaac.core.prims import RigidPrim
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage = world.stage
    root = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(root.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    xf.AddOrientOp().Set(Gf.Quatf(float(orientation[0]), Gf.Vec3f(float(orientation[1]), float(orientation[2]), float(orientation[3]))))

    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/CollisionMesh")
    mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * int(len(faces)))
    mesh.CreateFaceVertexIndicesAttr([int(x) for x in np.asarray(faces, dtype=np.int64).reshape(-1)])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()

    mesh_prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision_api.CreateApproximationAttr().Set("convexDecomposition")
    PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)
    try:
        physics_material.apply_to(mesh_prim)
    except Exception:
        pass

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(float(mass))
    if diagonal_inertia is not None:
        inertia = np.asarray(diagonal_inertia, dtype=np.float64)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(float(inertia[0]), float(inertia[1]), float(inertia[2])))
    if center_of_mass is not None:
        com = np.asarray(center_of_mass, dtype=np.float64)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(float(com[0]), float(com[1]), float(com[2])))

    visual_mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/RoboTwinVisualMesh")
    visual_mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in visual_vertices])
    visual_mesh.CreateFaceVertexCountsAttr([3] * int(len(visual_faces)))
    visual_mesh.CreateFaceVertexIndicesAttr([int(x) for x in np.asarray(visual_faces, dtype=np.int64).reshape(-1)])
    visual_mesh.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])

    obj = RigidPrim(
        prim_path=prim_path,
        name=name,
        position=np.asarray(position, dtype=np.float64),
        orientation=np.asarray(orientation, dtype=np.float64),
        mass=float(mass),
        reset_xform_properties=False,
    )
    try:
        world.scene.add(obj)
    except Exception:
        pass
    obj.set_linear_velocity(np.asarray(linear_velocity, dtype=np.float64))
    obj.set_angular_velocity(np.asarray(angular_velocity, dtype=np.float64))
    return obj


def run_falling(args, app, imports, out):
    World, PhysicsMaterial, DynamicCapsule, DynamicCuboid, DynamicSphere, FixedCuboid, Camera, UsdGeom, UsdLux = imports
    rng = random.Random(args.seed)
    table_z = 0.0
    world = setup_falling_floor_world(World, FixedCuboid, PhysicsMaterial, UsdGeom, UsdLux, table_z)
    colors = np.array([[0.90, 0.08, 0.05], [0.06, 0.32, 0.92], [0.10, 0.74, 0.22], [0.95, 0.66, 0.07], [0.58, 0.15, 0.82], [0.05, 0.68, 0.70]], dtype=np.float32)
    rods, meta, rod_sizes, rod_masses, rod_colors, disturbances = [], [], [], [], [], []
    for i in range(args.falling_rods):
        radius = rng.uniform(0.010, 0.020)
        height = rng.uniform(0.24, 0.46)
        mass = rng.uniform(0.075, 0.185)
        static_friction = rng.uniform(0.42, 0.78)
        dynamic_friction = rng.uniform(0.30, min(0.66, static_friction - 0.03))
        restitution = rng.uniform(0.08, 0.26)
        rod_mat = PhysicsMaterial(
            f"/World/PhysicsMaterials/rod_{i}",
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        yaw = math.pi * 0.5 + rng.uniform(-0.82, 0.82)
        pitch = rng.uniform(-0.46, 0.46)
        roll = rng.uniform(-0.36, 0.36)
        q = quat_mul(quat_mul(quat_axis_angle([0, 0, 1], yaw), quat_axis_angle([0, 1, 0], pitch)), quat_axis_angle([1, 0, 0], roll))
        pos = np.array([0.28 + 0.12 * i + rng.uniform(-0.055, 0.055), rng.uniform(-0.27, 0.24), rng.uniform(0.88, 1.42)])
        lin = np.array([rng.uniform(-0.38, 0.36), rng.uniform(-0.36, 0.34), rng.uniform(-0.55, 0.04)])
        ang = np.array([rng.uniform(-10.5, 10.5), rng.uniform(-10.5, 10.5), rng.uniform(-12.0, 12.0)])
        obj = DynamicCapsule(
            f"/World/FallingRod_{i}",
            name=f"falling_rod_{i}",
            position=pos,
            orientation=q,
            radius=radius,
            height=height,
            color=colors[i % len(colors)],
            physics_material=rod_mat,
            mass=mass,
            linear_velocity=lin,
            angular_velocity=ang,
        )
        rods.append(obj)
        rod_sizes.append([radius * 2.0, radius * 2.0, height])
        rod_masses.append(mass)
        rod_colors.append([*colors[i % len(colors)].tolist(), 1.0])
        per_rod_disturbances = []
        for _ in range(args.falling_disturbances_per_rod):
            step = rng.randint(max(4, args.frames // 12), max(5, int(args.frames * 0.78)))
            dv = np.array([rng.uniform(-0.10, 0.10), rng.uniform(-0.10, 0.10), rng.uniform(-0.035, 0.055)], dtype=np.float32)
            dw = np.array([rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(-2.6, 2.6)], dtype=np.float32)
            per_rod_disturbances.append({"step": int(step), "delta_linear_velocity_w": dv.tolist(), "delta_angular_velocity_w": dw.tolist()})
        disturbances.append(per_rod_disturbances)
        meta.append(
            {
                "name": obj.name,
                "category": "falling_baton",
                "source": "threedemo_parametric",
                "initial_position_w": pos.tolist(),
                "initial_height_m": float(pos[2]),
                "initial_quat_wxyz": q.tolist(),
                "initial_euler_xyz_rad": quat_to_euler_xyz(q).tolist(),
                "initial_lin_vel_w": lin.tolist(),
                "initial_ang_vel_w": ang.tolist(),
                "mass_kg": mass,
                "radius_m": radius,
                "height_m": height,
                "static_friction": static_friction,
                "dynamic_friction": dynamic_friction,
                "restitution": restitution,
                "color_rgb": colors[i % len(colors)].tolist(),
                "disturbance_impulses": per_rod_disturbances,
                "support_surface": "indoor_floor",
                "support_z_m": float(table_z),
            }
        )
    cameras = {
        # Eight RTX views. Videos are later overlaid with per-object 2D boxes and pixel masks.
        "overview": add_camera(Camera, "/World/Cameras/FallingOverview", "falling_overview", [3.20, -3.80, 2.10], [0.72, 0.00, 0.48], (args.width, args.height), 12.5),
        "closeup": add_camera(Camera, "/World/Cameras/FallingCloseup", "falling_closeup", [2.05, -2.30, 1.18], [0.72, 0.00, 0.34], (args.width, args.height), 18.0),
        "left_side": add_camera(Camera, "/World/Cameras/FallingLeftSide", "falling_left_side", [0.72, -4.05, 1.28], [0.72, 0.00, 0.42], (args.width, args.height), 12.0),
        "right_side": add_camera(Camera, "/World/Cameras/FallingRightSide", "falling_right_side", [0.72, 4.05, 1.28], [0.72, 0.00, 0.42], (args.width, args.height), 12.0),
        "top": add_camera(Camera, "/World/Cameras/FallingTop", "falling_top", [0.72, -0.28, 5.35], [0.72, 0.00, 0.12], (args.width, args.height), 10.0),
        "front": add_camera(Camera, "/World/Cameras/FallingFront", "falling_front", [4.20, 0.00, 1.38], [0.72, 0.00, 0.42], (args.width, args.height), 12.0),
        "left_oblique": add_camera(Camera, "/World/Cameras/FallingLeftOblique", "falling_left_oblique", [3.30, -3.20, 1.82], [0.72, 0.00, 0.42], (args.width, args.height), 12.5),
        "right_oblique": add_camera(Camera, "/World/Cameras/FallingRightOblique", "falling_right_oblique", [3.30, 3.20, 1.82], [0.72, 0.00, 0.42], (args.width, args.height), 12.5),
    }
    world.reset()
    for cam in cameras.values():
        cam.initialize()
    rod_sizes_arr = np.array(rod_sizes, np.float32)
    return collect_task(
        args,
        app,
        world,
        rods,
        cameras,
        out,
        "falling_baton",
        table_z,
        meta,
        lambda p, q, idx: capsule_bottom_delta(p, q, rod_sizes_arr[idx, 0] * 0.5, rod_sizes_arr[idx, 2], table_z),
        rod_sizes_arr,
        np.array(["capsule"] * len(rods)),
        np.array(rod_masses, np.float32),
        np.array(rod_colors, np.float32),
        disturbances=disturbances,
    )


def run_rolling(args, app, imports, out):
    World, PhysicsMaterial, DynamicCapsule, DynamicCuboid, DynamicSphere, FixedCuboid, Camera, UsdGeom, UsdLux = imports
    rng = random.Random(args.seed + 1009)
    frames = int(args.frames)
    table_z = 0.0
    world = setup_falling_floor_world(World, FixedCuboid, PhysicsMaterial, UsdGeom, UsdLux, table_z)
    basic_catalog = [
        {"name": "rolling_ball", "category": "sphere", "shape": "sphere", "mass": 0.09, "size": [0.070, 0.070, 0.070], "color": [0.95, 0.30, 0.08], "source": "builtin"},
        {"name": "rolling_box", "category": "box", "shape": "box", "mass": 0.13, "size": [0.090, 0.055, 0.045], "color": [0.08, 0.52, 0.92], "source": "builtin"},
        {"name": "rolling_bar", "category": "bar", "shape": "box", "mass": 0.11, "size": [0.150, 0.038, 0.038], "color": [0.12, 0.72, 0.24], "source": "builtin"},
        {"name": "rolling_l_block_proxy", "category": "l_shape", "shape": "box", "mass": 0.15, "size": [0.105, 0.082, 0.044], "color": [0.94, 0.74, 0.12], "source": "builtin_proxy"},
    ]
    robotwin_catalog = [
        {"name": "robotwin_bottle_mesh", "category": "bottle", "shape": "robotwin_collision_mesh_convex", "mass": 0.16, "size": [0.055, 0.055, 0.155], "color": [0.10, 0.55, 0.88], "source": "robotwin", "robotwin_object_id": "001_bottle"},
        {"name": "robotwin_bowl_mesh", "category": "bowl", "shape": "robotwin_collision_mesh_convex", "mass": 0.12, "size": [0.105, 0.105, 0.105], "color": [0.92, 0.42, 0.14], "source": "robotwin", "robotwin_object_id": "002_bowl"},
        {"name": "robotwin_plate_mesh", "category": "plate", "shape": "robotwin_collision_mesh_convex", "mass": 0.10, "size": [0.145, 0.145, 0.026], "color": [0.86, 0.86, 0.80], "source": "robotwin", "robotwin_object_id": "003_plate"},
        {"name": "robotwin_hammer_mesh", "category": "hammer", "shape": "robotwin_collision_mesh_convex", "mass": 0.24, "size": [0.185, 0.050, 0.043], "color": [0.22, 0.22, 0.22], "source": "robotwin", "robotwin_object_id": "020_hammer"},
        {"name": "robotwin_tray_mesh", "category": "tray", "shape": "robotwin_collision_mesh_convex", "mass": 0.18, "size": [0.165, 0.115, 0.030], "color": [0.42, 0.34, 0.24], "source": "robotwin", "robotwin_object_id": "008_tray"},
        {"name": "robotwin_cup_mesh", "category": "cup", "shape": "robotwin_collision_mesh_convex", "mass": 0.13, "size": [0.075, 0.075, 0.110], "color": [0.58, 0.30, 0.76], "source": "robotwin", "robotwin_object_id": "021_cup"},
        {"name": "robotwin_mug_mesh", "category": "mug", "shape": "robotwin_collision_mesh_convex", "mass": 0.15, "size": [0.082, 0.082, 0.105], "color": [0.78, 0.78, 0.84], "source": "robotwin", "robotwin_object_id": "039_mug"},
        {"name": "robotwin_can_mesh", "category": "can", "shape": "robotwin_collision_mesh_convex", "mass": 0.15, "size": [0.064, 0.064, 0.122], "color": [0.90, 0.18, 0.26], "source": "robotwin", "robotwin_object_id": "071_can"},
        {"name": "robotwin_woodenmallet_mesh", "category": "woodenmallet", "shape": "robotwin_collision_mesh_convex", "mass": 0.26, "size": [0.180, 0.058, 0.050], "color": [0.58, 0.34, 0.16], "source": "robotwin", "robotwin_object_id": "084_woodenmallet"},
        {"name": "robotwin_dumbbell_mesh", "category": "dumbbell", "shape": "robotwin_collision_mesh_convex", "mass": 0.30, "size": [0.155, 0.055, 0.055], "color": [0.16, 0.16, 0.18], "source": "robotwin", "robotwin_object_id": "052_dumbbell"},
        {"name": "robotwin_brush_mesh", "category": "brush", "shape": "robotwin_collision_mesh_convex", "mass": 0.10, "size": [0.155, 0.042, 0.036], "color": [0.18, 0.52, 0.72], "source": "robotwin", "robotwin_object_id": "083_brush"},
        {"name": "robotwin_remotecontrol_mesh", "category": "remotecontrol", "shape": "robotwin_collision_mesh_convex", "mass": 0.11, "size": [0.155, 0.045, 0.022], "color": [0.10, 0.10, 0.12], "source": "robotwin", "robotwin_object_id": "079_remotecontrol"},
        {"name": "robotwin_stapler_mesh", "category": "stapler", "shape": "robotwin_collision_mesh_convex", "mass": 0.17, "size": [0.125, 0.042, 0.048], "color": [0.22, 0.22, 0.26], "source": "robotwin", "robotwin_object_id": "048_stapler"},
        {"name": "robotwin_markpen_mesh", "category": "markpen", "shape": "robotwin_collision_mesh_convex", "mass": 0.055, "size": [0.150, 0.022, 0.022], "color": [0.05, 0.05, 0.05], "source": "robotwin", "robotwin_object_id": "058_markpen"},
    ]
    catalog = basic_catalog if args.rolling_object_set == "basic" else robotwin_catalog
    if args.rolling_object_set == "basic":
        catalog = basic_catalog
    if args.rolling_category != "random":
        catalog = [x for x in catalog if x["category"] == args.rolling_category]
        if not catalog:
            raise ValueError(f"Unknown rolling category: {args.rolling_category}")
    # Demo1 is defined as a single-object episode; run repeated episodes for dataset coverage.
    specs = [dict(rng.choice(catalog))]
    count = len(specs)
    for i, sp in enumerate(specs):
        sp["x0"] = rng.uniform(-0.72, 0.22)
        sp["y0"] = rng.uniform(-0.46, 0.46)
        sp["initial_position_w"] = [sp["x0"], sp["y0"], 0.0]
        easy_motion = sp["category"] in {"box", "bar", "l_shape", "plate", "hammer", "tray", "woodenmallet", "dumbbell", "brush", "remotecontrol", "stapler", "markpen"}
        if easy_motion:
            sp["static_friction"] = rng.uniform(0.38, 0.72)
            sp["dynamic_friction"] = rng.uniform(0.24, min(0.58, sp["static_friction"] - 0.05))
        else:
            sp["static_friction"] = rng.uniform(0.46, 0.86)
            sp["dynamic_friction"] = rng.uniform(0.30, min(0.68, sp["static_friction"] - 0.06))
        sp["restitution"] = rng.uniform(0.02, 0.12)

    def random_rolling_orientation(category):
        yaw = rng.uniform(-math.pi, math.pi)
        if category in {"hammer", "woodenmallet", "dumbbell", "brush", "remotecontrol", "stapler", "markpen", "bar", "l_shape"}:
            roll = rng.uniform(-1.25, 1.25)
            pitch = rng.uniform(-1.10, 1.10)
        elif category in {"cup", "mug", "bottle", "can"} and rng.random() < 0.45:
            roll = rng.choice([-1.0, 1.0]) * rng.uniform(0.82, 1.44)
            pitch = rng.uniform(-0.62, 0.62)
        elif category in {"plate", "tray", "bowl"}:
            roll = rng.uniform(-0.42, 0.42)
            pitch = rng.uniform(-0.42, 0.42)
        else:
            roll = rng.uniform(-0.62, 0.62)
            pitch = rng.uniform(-0.62, 0.62)
        return quat_mul(quat_mul(quat_axis_angle([0, 0, 1], yaw), quat_axis_angle([1, 0, 0], roll)), quat_axis_angle([0, 1, 0], pitch))

    def center_of_mass_offset(category, size):
        frac = {
            "hammer": [0.22, -0.02, 0.05],
            "woodenmallet": [0.18, 0.00, 0.03],
            "dumbbell": [0.00, 0.00, 0.00],
            "brush": [0.08, 0.06, 0.02],
            "remotecontrol": [0.04, 0.00, 0.05],
            "stapler": [0.12, 0.00, 0.10],
            "markpen": [0.04, 0.00, 0.00],
            "cup": [0.02, 0.00, 0.10],
            "mug": [0.06, 0.08, 0.06],
            "bottle": [0.00, 0.00, 0.12],
            "can": [0.00, 0.00, 0.06],
            "bowl": [0.00, 0.00, -0.04],
            "plate": [0.00, 0.00, -0.02],
            "tray": [0.00, 0.00, -0.04],
        }.get(category, [0.0, 0.0, 0.0])
        jitter = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.015, 0.015)], dtype=np.float64)
        offset = (np.asarray(frac, dtype=np.float64) + jitter) * np.asarray(size, dtype=np.float64)
        limit = np.asarray(size, dtype=np.float64) * 0.38
        return np.clip(offset, -limit, limit)

    def inertia_diag_for(category, size, mass):
        sx, sy, sz = [float(x) for x in size]
        base = np.array([mass * (sy * sy + sz * sz) / 12.0, mass * (sx * sx + sz * sz) / 12.0, mass * (sx * sx + sy * sy) / 12.0], dtype=np.float64)
        scale = {
            "hammer": [0.72, 1.36, 1.48],
            "woodenmallet": [0.78, 1.30, 1.42],
            "dumbbell": [0.92, 1.18, 1.22],
            "brush": [0.78, 1.20, 1.24],
            "remotecontrol": [0.70, 1.14, 1.18],
            "stapler": [0.82, 1.16, 1.24],
            "markpen": [0.64, 1.18, 1.18],
            "cup": [1.10, 1.10, 0.76],
            "mug": [1.16, 1.10, 0.82],
            "bottle": [1.12, 1.12, 0.70],
            "can": [1.05, 1.05, 0.72],
            "bowl": [1.02, 1.02, 0.86],
            "plate": [1.20, 1.20, 0.58],
            "tray": [1.16, 1.02, 0.64],
        }.get(category, [1.0, 1.0, 1.0])
        jitter = np.array([rng.uniform(0.92, 1.10), rng.uniform(0.92, 1.10), rng.uniform(0.92, 1.10)], dtype=np.float64)
        return np.maximum(base * np.asarray(scale, dtype=np.float64) * jitter, 1e-6)

    objs, meta, visual_ops, mesh_vertices_for_bottom = [], [], [], []
    for i, sp in enumerate(specs):
        size = np.array(sp["size"], dtype=np.float64)
        elongated = sp["category"] in {"bar", "hammer", "woodenmallet", "dumbbell", "brush", "remotecontrol", "stapler", "markpen"}
        speed_xy = rng.uniform(0.68, 1.58 if elongated else 1.28)
        heading = rng.uniform(-math.pi, math.pi)
        spin_scale = 1.55 if elongated else 1.15
        lin = np.array([speed_xy * math.cos(heading), speed_xy * math.sin(heading), rng.uniform(-0.010, 0.018)])
        ang = np.array(
            [
                rng.uniform(-8.5, 8.5) * spin_scale,
                rng.uniform(-8.5, 8.5) * spin_scale,
                rng.uniform(-10.0, 10.0) * spin_scale,
            ]
        )
        q = random_rolling_orientation(sp["category"])
        sp["disturbances"] = []
        for _ in range(max(0, int(args.rolling_disturbances_per_object))):
            step = rng.randint(max(2, frames // 10), max(3, int(frames * 0.82)))
            kick_heading = heading + rng.uniform(-1.35, 1.35)
            impulse_speed = rng.uniform(0.04, 0.20)
            sp["disturbances"].append(
                {
                    "step": int(step),
                    "kind": "force_torque_impulse_proxy",
                    "delta_linear_velocity_w": [float(impulse_speed * math.cos(kick_heading)), float(impulse_speed * math.sin(kick_heading)), float(rng.uniform(-0.015, 0.045))],
                    "delta_angular_velocity_w": [float(rng.uniform(-3.8, 3.8)), float(rng.uniform(-3.8, 3.8)), float(rng.uniform(-4.6, 4.6))],
                }
            )
        sp["disturbances"].sort(key=lambda item: int(item["step"]))
        mesh_vertices = None
        visual_vertices = None
        visual_faces = None
        if str(sp["shape"]).startswith("robotwin_collision_mesh"):
            variants = robotwin_available_variants(sp["robotwin_object_id"])
            sp["robotwin_mesh_variant"] = int(rng.choice(variants)) if variants else None
            visual_path, collision_path = robotwin_asset_paths(sp["robotwin_object_id"], sp["robotwin_mesh_variant"])
            vertices, faces, resolved_collision_path, original_extents, mesh_scale = load_collision_mesh_as_fit_vertices(collision_path, size)
            visual_vertices, visual_faces, resolved_visual_path, visual_original_extents, visual_mesh_scale = load_mesh_as_fit_vertices(visual_path, size)
            sp["visual_asset_path"] = visual_path
            sp["resolved_visual_asset_path"] = resolved_visual_path
            sp["collision_asset_path"] = resolved_collision_path
            sp["collision_mesh_original_extents"] = original_extents
            sp["collision_mesh_fit_scale"] = mesh_scale
            sp["collision_mesh_vertex_count"] = int(len(vertices))
            sp["collision_mesh_face_count"] = int(len(faces))
            sp["visual_mesh_original_extents"] = visual_original_extents
            sp["visual_mesh_fit_scale"] = visual_mesh_scale
            sp["visual_mesh_vertex_count"] = int(len(visual_vertices))
            sp["visual_mesh_face_count"] = int(len(visual_faces))
            mesh_vertices = vertices
            axes = [quat_rotate(q, np.eye(3)[j]) for j in range(3)]
            support = max(0.0, -float(np.min(vertices @ np.array([a[2] for a in axes], dtype=np.float64))))
        elif sp["shape"] == "sphere":
            support = size[0] * 0.5
        elif sp["shape"] == "capsule":
            axis = quat_rotate(q, [0.0, 0.0, 1.0])
            support = 0.5 * max(0.0, size[2] - size[0]) * abs(float(axis[2])) + size[0] * 0.5
        else:
            axes = [quat_rotate(q, np.eye(3)[j]) for j in range(3)]
            support = sum(abs(float(a[2])) * float(s) * 0.5 for a, s in zip(axes, size))
        z = table_z + float(support) + 0.004
        sp["initial_position_w"] = [sp["x0"], sp["y0"], float(z)]
        sp["initial_euler_xyz_rad"] = quat_to_euler_xyz(q).tolist()
        sp["initial_speed_xy_m_s"] = float(speed_xy)
        sp["initial_heading_rad"] = float(heading)
        roll_mat = PhysicsMaterial(
            f"/World/PhysicsMaterials/rolling_{i}",
            static_friction=float(sp["static_friction"]),
            dynamic_friction=float(sp["dynamic_friction"]),
            restitution=float(sp["restitution"]),
        )
        com_offset = center_of_mass_offset(sp["category"], size)
        inertia_diag = inertia_diag_for(sp["category"], size, float(sp["mass"])).tolist()
        sp["center_of_mass_offset_m"] = com_offset.tolist()
        sp["inertia_diag_kg_m2"] = inertia_diag
        if str(sp["shape"]).startswith("robotwin_collision_mesh"):
            obj = create_robotwin_mesh_rigid(
                world,
                UsdGeom,
                f"/World/Rolling_{i}",
                sp["name"],
                np.array([sp["x0"], sp["y0"], z]),
                q,
                mesh_vertices,
                faces,
                visual_vertices,
                visual_faces,
                sp["color"],
                roll_mat,
                sp["mass"],
                inertia_diag,
                com_offset,
                lin,
                ang,
            )
        elif sp["shape"] == "sphere":
            obj = DynamicSphere(
                f"/World/Rolling_{i}",
                name=sp["name"],
                position=np.array([sp["x0"], sp["y0"], z]),
                radius=size[0] * 0.5,
                color=np.array(sp["color"]),
                physics_material=roll_mat,
                mass=sp["mass"],
                linear_velocity=lin,
                angular_velocity=ang,
            )
        elif sp["shape"] == "capsule":
            obj = DynamicCapsule(
                f"/World/Rolling_{i}",
                name=sp["name"],
                position=np.array([sp["x0"], sp["y0"], z]),
                orientation=q,
                radius=size[0] * 0.5,
                height=size[2],
                color=np.array(sp["color"]),
                physics_material=roll_mat,
                mass=sp["mass"],
                linear_velocity=lin,
                angular_velocity=ang,
            )
        else:
            obj = DynamicCuboid(
                f"/World/Rolling_{i}",
                name=sp["name"],
                position=np.array([sp["x0"], sp["y0"], z]),
                orientation=q,
                scale=size,
                color=np.array(sp["color"]),
                physics_material=roll_mat,
                mass=sp["mass"],
                linear_velocity=lin,
                angular_velocity=ang,
            )
        objs.append(obj)
        visual_ops.append(None if str(sp["shape"]).startswith("robotwin_collision_mesh") else make_follow_visual(world.stage, UsdGeom, f"/World/RollingVisual_{i}", sp["category"], size, sp["color"]))
        mesh_vertices_for_bottom.append(mesh_vertices)
        meta.append({"name": obj.name, **sp, "initial_lin_vel_w": lin.tolist(), "initial_ang_vel_w": ang.tolist(), "initial_quat_wxyz": q.tolist()})
    cameras = {
        "overview": add_camera(Camera, "/World/Cameras/RollingOverview", "rolling_overview", [2.35, -3.05, 1.46], [0.08, 0.00, 0.20], (args.width, args.height), 14.0),
        "closeup": add_camera(Camera, "/World/Cameras/RollingCloseup", "rolling_closeup", [1.38, -2.02, 0.86], [0.00, 0.00, 0.16], (args.width, args.height), 21.0),
    }
    world.reset()
    for cam in cameras.values():
        cam.initialize()

    sizes = np.array([s["size"] for s in specs], np.float32)
    shapes = np.array([s["shape"] for s in specs])
    masses = np.array([s["mass"] for s in specs], np.float32)
    colors = np.array([[*s["color"], 1.0] for s in specs], np.float32)

    def bottom_fn(p, q, idx):
        sp = specs[idx]
        if sp["shape"] == "sphere":
            return float(p[2] - sp["size"][0] * 0.5 - table_z)
        if sp["shape"] == "capsule":
            return capsule_bottom_delta(p, q, sp["size"][0] * 0.5, sp["size"][2], table_z)
        if str(sp["shape"]).startswith("robotwin_collision_mesh"):
            return mesh_bottom_delta(p, q, mesh_vertices_for_bottom[idx], table_z)
        return box_bottom_delta(p, q, np.array(sp["size"], dtype=np.float32), table_z)

    return collect_task(
        args,
        app,
        world,
        objs,
        cameras,
        out,
        "rolling_tabletop",
        table_z,
        meta,
        bottom_fn,
        sizes,
        shapes,
        masses,
        colors,
        visual_ops=visual_ops,
        disturbances=[s.get("disturbances", []) for s in specs],
    )


def collect_task(args, app, world, objects, cameras, out, task, table_z, meta, bottom_fn, sizes, shapes, masses, colors, visual_ops=None, disturbances=None):
    from pxr import Gf

    frames = int(args.frames)
    n = len(objects)
    table_center_xy = np.array([0.62, 0.0], dtype=np.float32)
    support_surface = "indoor_floor" if float(table_z) <= 0.001 else "table"
    support_contact_event_type = "floor_contact_begin" if support_surface == "indoor_floor" else "table_contact_begin"
    if support_surface == "indoor_floor":
        table_half_xy = np.array([4.20, 3.00], dtype=np.float32) * 0.5
    elif task == "rolling_tabletop":
        table_half_xy = np.array([2.80, 1.05], dtype=np.float32) * 0.5
    else:
        table_half_xy = np.array([1.35, 0.92], dtype=np.float32) * 0.5
    pos = np.zeros((frames, n, 3), np.float32)
    quat = np.zeros((frames, n, 4), np.float32)
    lin = np.zeros((frames, n, 3), np.float32)
    ang = np.zeros((frames, n, 3), np.float32)
    euler = np.zeros((frames, n, 3), np.float32)
    tilt_angle = np.zeros((frames, n), np.float32)
    rotation_from_initial = np.zeros((frames, n), np.float32)
    center_of_mass_w = np.zeros((frames, n, 3), np.float32)
    bottom = np.zeros((frames, n), np.float32)
    over_table = np.zeros((frames, n), np.bool_)
    active_support_z = np.zeros((frames, n), np.float32)
    bottom_active = np.zeros((frames, n), np.float32)
    table_contact = np.zeros((frames, n), np.bool_)
    ground_contact = np.zeros((frames, n), np.bool_)
    pair_contact = np.zeros((frames, n, n), np.bool_)
    disturbance_delta_lin_vel = np.zeros((frames, n, 3), np.float32)
    disturbance_delta_ang_vel = np.zeros((frames, n, 3), np.float32)
    disturbance_events = [[] for _ in range(frames)]
    if disturbances:
        for obj_i, items in enumerate(disturbances):
            for item in items:
                step = int(item.get("step", -1))
                if 0 <= step < frames:
                    disturbance_delta_lin_vel[step, obj_i] += np.asarray(item.get("delta_linear_velocity_w", [0, 0, 0]), dtype=np.float32)
                    disturbance_delta_ang_vel[step, obj_i] += np.asarray(item.get("delta_angular_velocity_w", [0, 0, 0]), dtype=np.float32)
                    disturbance_events[step].append({"object_i": int(obj_i), **item})
    center_of_mass_local = np.array([x.get("center_of_mass_offset_m", [0.0, 0.0, 0.0]) for x in meta], dtype=np.float32)
    initial_quat_ref = np.array([x.get("initial_quat_wxyz", [1.0, 0.0, 0.0, 0.0]) for x in meta], dtype=np.float32)
    for i, obj in enumerate(objects):
        try:
            obj.set_linear_velocity(np.asarray(meta[i].get("initial_lin_vel_w", [0, 0, 0]), dtype=np.float64))
            obj.set_angular_velocity(np.asarray(meta[i].get("initial_ang_vel_w", [0, 0, 0]), dtype=np.float64))
        except Exception:
            pass
    first_table = [-1] * n
    first_ground = [-1] * n
    first_pair = {}
    events = []
    video_frames = {k: [] for k in cameras}
    for t in range(frames):
        if disturbance_events[t]:
            for item in disturbance_events[t]:
                i = int(item["object_i"])
                try:
                    objects[i].set_linear_velocity(objects[i].get_linear_velocity() + disturbance_delta_lin_vel[t, i])
                    objects[i].set_angular_velocity(objects[i].get_angular_velocity() + disturbance_delta_ang_vel[t, i])
                except Exception:
                    pass
                events.append({"step": t, "type": "random_disturbance_impulse", **item})
        world.step(render=True)
        for i, obj in enumerate(objects):
            p, q = obj.get_world_pose()
            v = obj.get_linear_velocity()
            w = obj.get_angular_velocity()
            pos[t, i], quat[t, i], lin[t, i], ang[t, i] = p, q, v, w
            euler[t, i] = quat_to_euler_xyz(q)
            local_z_w = quat_rotate(q, [0.0, 0.0, 1.0])
            tilt_angle[t, i] = math.acos(float(np.clip(abs(local_z_w[2]), 0.0, 1.0)))
            qn = quat_norm(q)
            q0n = quat_norm(initial_quat_ref[i])
            rotation_from_initial[t, i] = 2.0 * math.acos(float(np.clip(abs(float(np.dot(qn, q0n))), 0.0, 1.0)))
            center_of_mass_w[t, i] = np.asarray(p, dtype=np.float32) + quat_rotate(q, center_of_mass_local[i])
            bottom[t, i] = bottom_fn(p, q, i)
            over_table[t, i] = bool(np.all(np.abs(pos[t, i, :2] - table_center_xy) <= table_half_xy))
            active_support_z[t, i] = table_z if over_table[t, i] else 0.0
            bottom_active[t, i] = bottom[t, i] + (table_z - active_support_z[t, i])
            if over_table[t, i] and bottom[t, i] < 0.018:
                table_contact[t, i] = True
                if first_table[i] < 0:
                    first_table[i] = t
                    events.append({"step": t, "type": support_contact_event_type, "object_i": i, "support_surface": support_surface})
            if (not over_table[t, i]) and bottom_active[t, i] < 0.018:
                ground_contact[t, i] = True
                if first_ground[i] < 0:
                    first_ground[i] = t
                    events.append({"step": t, "type": "ground_contact_begin_after_leaving_table", "object_i": i})
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(pos[t, i] - pos[t, j]))
                thresh = float(max(np.linalg.norm(sizes[i]), np.linalg.norm(sizes[j])) * 0.55)
                if d < thresh:
                    pair_contact[t, i, j] = pair_contact[t, j, i] = True
                    if (i, j) not in first_pair:
                        first_pair[(i, j)] = t
                        events.append({"step": t, "type": "object_pair_near_contact", "object_i": i, "object_j": j, "distance_m": d})
        if visual_ops:
            for i, ops in enumerate(visual_ops):
                if ops is None:
                    continue
                translate_op, orient_op = ops
                translate_op.Set(Gf.Vec3f(float(pos[t, i, 0]), float(pos[t, i, 1]), float(pos[t, i, 2])))
                orient_op.Set(Gf.Quatf(float(quat[t, i, 0]), Gf.Vec3f(float(quat[t, i, 1]), float(quat[t, i, 2]), float(quat[t, i, 3]))))
        if t % max(1, args.video_stride) == 0:
            for name, cam in cameras.items():
                frame = camera_frame(cam, world, app)
                if frame is not None:
                    video_frames[name].append(frame)

    inertia = np.zeros((n, 3), np.float32)
    for i in range(n):
        sx, sy, sz = sizes[i]
        m = masses[i]
        if "inertia_diag_kg_m2" in meta[i]:
            inertia[i] = np.asarray(meta[i]["inertia_diag_kg_m2"], dtype=np.float32)
        elif shapes[i] == "sphere":
            inertia[i] = [0.4 * m * (sx * 0.5) ** 2] * 3
        elif shapes[i] == "capsule":
            inertia[i] = [m * (3 * (sx * 0.5) ** 2 + sz**2) / 12, m * (3 * (sx * 0.5) ** 2 + sz**2) / 12, 0.5 * m * (sx * 0.5) ** 2]
        else:
            inertia[i] = [m * (sy * sy + sz * sz) / 12, m * (sx * sx + sz * sz) / 12, m * (sx * sx + sy * sy) / 12]
    long_axis = np.zeros((frames, n, 3), np.float32)
    for t in range(frames):
        for i in range(n):
            long_axis[t, i] = quat_rotate(quat[t, i], [0, 0, 1] if shapes[i] == "capsule" else [1, 0, 0])
    categories = np.array([str(x.get("category", x.get("name", ""))) for x in meta])
    names = np.array([str(x.get("name", "")) for x in meta])
    sources = np.array([str(x.get("source", "")) for x in meta])
    visual_assets = np.array([str(x.get("visual_asset_path", "")) for x in meta])
    collision_assets = np.array([str(x.get("collision_asset_path", "")) for x in meta])
    initial_pose = np.array(
        [
            [*x.get("initial_position_w", pos[0, i].tolist()), *x.get("initial_quat_wxyz", quat[0, i].tolist())]
            for i, x in enumerate(meta)
        ],
        dtype=np.float32,
    )
    initial_lin_vel = np.array([x.get("initial_lin_vel_w", lin[0, i].tolist()) for i, x in enumerate(meta)], dtype=np.float32)
    initial_ang_vel = np.array([x.get("initial_ang_vel_w", ang[0, i].tolist()) for i, x in enumerate(meta)], dtype=np.float32)
    initial_euler = np.array([x.get("initial_euler_xyz_rad", quat_to_euler_xyz(initial_pose[i, 3:7]).tolist()) for i, x in enumerate(meta)], dtype=np.float32)
    static_friction = np.array([float(x.get("static_friction", np.nan)) for x in meta], dtype=np.float32)
    dynamic_friction = np.array([float(x.get("dynamic_friction", np.nan)) for x in meta], dtype=np.float32)
    restitution = np.array([float(x.get("restitution", np.nan)) for x in meta], dtype=np.float32)
    initial_height = np.array([float(x.get("initial_height_m", initial_pose[i, 2])) for i, x in enumerate(meta)], dtype=np.float32)
    def _safe_float(value, default=0.0):
        try:
            value = float(value)
        except Exception:
            return float(default)
        return value if np.isfinite(value) else float(default)

    def _safe_vec(values, count=3):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[:count] if arr.size >= count else np.pad(arr, (0, count - arr.size))
        return [_safe_float(x) for x in arr.tolist()]

    task_dir = out / task
    task_dir.mkdir(parents=True, exist_ok=True)
    mesh_export_summary = export_episode_mesh_assets(task_dir, meta, shapes, sizes)
    visual_mesh_exports = np.array(mesh_export_summary["visual_paths"], dtype=str)
    collision_mesh_exports = np.array(mesh_export_summary["collision_paths"], dtype=str)
    mesh_export_dirs = np.array(mesh_export_summary["object_dirs"], dtype=str)
    mesh_export_formats = np.array(mesh_export_summary["formats"], dtype=str)
    mesh_assets_manifest_path = np.array(mesh_export_summary["manifest_path"])
    from threedemo_annotation_export import generate_object_2d_annotations_and_overlays
    annotation_summary = generate_object_2d_annotations_and_overlays(
        task_dir=task_dir,
        task=task,
        cameras=cameras,
        video_frames=video_frames,
        object_pos_w=pos,
        object_quat_wxyz=quat,
        object_visual_mesh_export_paths=visual_mesh_exports.tolist(),
        object_names=names.tolist(),
        object_categories=categories.tolist(),
        object_colors_rgba=np.asarray(colors, dtype=np.float32),
        video_stride=max(1, int(args.video_stride)),
        fps=max(1, int(args.fps // max(1, int(args.video_stride)))),
    )
    annotation_manifest_path = np.array(annotation_summary["manifest_path"])
    object_2d_bbox_xyxy_path_json = np.array(json.dumps(annotation_summary["bbox_paths"], sort_keys=True))
    object_pixel_mask_rle_path_json = np.array(json.dumps(annotation_summary["mask_paths"], sort_keys=True))
    camera_intrinsics_json = np.array(json.dumps(annotation_summary.get("camera_intrinsics", {}), sort_keys=True))

    object_text_descriptions = np.array(
        [
            (
                f"{names[i]} is a {categories[i]} rigid object from {sources[i] or 'unknown_source'} with proxy shape {shapes[i]}. "
                f"It uses visual asset {visual_assets[i] or 'procedural_visual'} and collision asset {collision_assets[i] or 'procedural_collision_from_proxy'}. "
                f"The collected per-episode visual mesh is {visual_mesh_exports[i]} and collision mesh is {collision_mesh_exports[i]}. "
                f"Its size is {_safe_vec(sizes[i])} m, mass is {_safe_float(masses[i]):.4f} kg, center of mass offset is {_safe_vec(center_of_mass_local[i])} m, "
                f"inertia diagonal is {_safe_vec(inertia[i])} kg*m^2, static/dynamic friction is {_safe_float(static_friction[i]):.4f}/{_safe_float(dynamic_friction[i]):.4f}, "
                f"and restitution is {_safe_float(restitution[i]):.4f}."
            )
            for i in range(n)
        ],
        dtype=str,
    )
    object_standard_descriptions = np.array(
        [
            json.dumps(
                {
                    "schema": "geniesim_like_object_v1",
                    "index": int(i),
                    "name": str(names[i]),
                    "category": str(categories[i]),
                    "source": str(sources[i]),
                    "visual": {"asset_path": str(visual_assets[i]), "mesh_export_path": str(visual_mesh_exports[i]), "mesh_format": str(mesh_export_formats[i])},
                    "collision": {"asset_path": str(collision_assets[i]), "mesh_export_path": str(collision_mesh_exports[i]), "proxy_shape": str(shapes[i]), "mesh_format": str(mesh_export_formats[i])},
                    "geometry": {"size_m": _safe_vec(sizes[i])},
                    "mass_kg": _safe_float(masses[i]),
                    "center_of_mass_local_m": _safe_vec(center_of_mass_local[i]),
                    "inertia_diag_kg_m2": _safe_vec(inertia[i]),
                    "material": {
                        "static_friction": _safe_float(static_friction[i]),
                        "dynamic_friction": _safe_float(dynamic_friction[i]),
                        "restitution": _safe_float(restitution[i]),
                    },
                    "initial_state": {
                        "pose_w_xyz_qwqxqyqz": _safe_vec(initial_pose[i], 7),
                        "linear_velocity_w_m_s": _safe_vec(initial_lin_vel[i]),
                        "angular_velocity_w_rad_s": _safe_vec(initial_ang_vel[i]),
                    },
                },
                sort_keys=True,
            )
            for i in range(n)
        ],
        dtype=str,
    )
    if task == "rolling_tabletop":
        scene_text_description = (
            "A single rigid object starts on a textured indoor floor with randomized pose, linear velocity, angular velocity, "
            "mass, center of mass, inertia, friction, restitution, and disturbance force/torque impulses. The object slides, "
            "rolls, spins, and may tip or tumble under IsaacSim PhysX gravity and floor contact."
        )
        camera_names = list(cameras.keys())
    elif task == "falling_baton":
        scene_text_description = (
            "Eight rigid baton objects fall inside an indoor room above a textured floor. Each baton has randomized pose, velocity, "
            "angular velocity, mass, size, inertia, friction, restitution, and disturbance impulses, then falls under gravity and contacts the floor."
        )
        camera_names = list(cameras.keys())
    else:
        scene_text_description = "Rigid objects move in an indoor PhysX scene with recorded assets, collision, mass properties, contacts, and trajectories."
        camera_names = sorted(list(cameras.keys())) if 'cameras' in locals() else []
    scene_standard_description = json.dumps(
        {
            "schema": "geniesim_like_scene_task_v1",
            "engine": "IsaacSim PhysX",
            "task": str(task),
            "support_surface": str(support_surface),
            "environment": {"type": "indoor_room", "floor": "textured indoor floor", "background": "indoor walls", "gravity_m_s2": [0.0, 0.0, -9.81]},
            "units": "SI",
            "world_frame": "right_handed_world_xyz",
            "pose_convention": "position_xyz_m + quaternion_wxyz",
            "object_count": int(n),
            "objects": [json.loads(x) for x in object_standard_descriptions.tolist()],
            "cameras": camera_names,
            "recorded_state_fields": ["object_pose_w", "object_pos_w", "object_quat_wxyz", "object_lin_vel_w", "object_ang_vel_w", "object_momentum_kg_m_s", "object_angular_momentum_kg_m2_rad_s", "object_mass_kg", "object_center_of_mass_local_m", "object_inertia_diag_kg_m2", "object_static_friction", "object_dynamic_friction", "object_restitution", "object_trajectory_w", "contact_flags", "initial_state", "final_pose_w"],
        },
        sort_keys=True,
    )
    state_replay_import_contract = json.dumps(
        {
            "schema": "physx_replay_import_contract_v1",
            "required_engine": "IsaacSim PhysX",
            "support_surface": str(support_surface),
            "dt": float(1.0 / 120.0),
            "object_count": int(n),
            "pose_convention": "position_xyz_m + quaternion_wxyz",
            "import_steps": [
                "Create the indoor floor/support surface named by support_surface.",
                "Instantiate each rigid body from object_collision_mesh_export_path in the per-episode mesh assets; use object_visual_mesh_export_path for rendering, and only fall back to object_collision_asset_path or object_proxy_shape if explicitly testing fallback import.",
                "Assign object_mass_kg, object_center_of_mass_local_m, object_inertia_diag_kg_m2, object_static_friction, object_dynamic_friction, and object_restitution.",
                "Set initial_object_pose_w, initial_object_lin_vel_w, and initial_object_ang_vel_w.",
                "Simulate with dt, then compare object_pose_w, object_lin_vel_w, object_ang_vel_w, and contact fields against the recorded trajectory.",
            ],
        },
        sort_keys=True,
    )
    task_dir = out / task
    task_dir.mkdir(parents=True, exist_ok=True)
    dataset = {
        "schema": "threedemo_formal_physx_task_v1",
        "task": task,
        "physics_source": "isaacsim_physx_direct",
        "support_surface": np.array(support_surface),
        "dt": np.float32(1.0 / 120.0),
        "table_z": np.float32(table_z),
        "support_z": np.float32(table_z),
        "object_pose_w": np.concatenate([pos, quat], axis=-1),
        "object_pos_w": pos,
        "object_quat_wxyz": quat,
        "object_lin_vel_w": lin,
        "object_ang_vel_w": ang,
        "object_vel_w": np.concatenate([lin, ang], axis=-1),
        "object_center_of_mass_w": center_of_mass_w,
        "object_center_of_mass_local_m": center_of_mass_local,
        "object_long_axis_w": long_axis,
        "object_euler_xyz_rad": euler,
        "object_roll_pitch_yaw_rad": euler,
        "object_tilt_angle_rad": tilt_angle,
        "object_rotation_angle_from_initial_rad": rotation_from_initial,
        "object_trajectory_w": pos,
        "object_mass_kg": masses,
        "object_inertia_diag_kg_m2": inertia,
        "object_momentum_kg_m_s": lin * masses.reshape(1, n, 1),
        "object_angular_momentum_kg_m2_rad_s": ang * inertia.reshape(1, n, 3),
        "object_size_m": sizes,
        "object_color_rgba": colors,
        "object_proxy_shape": shapes,
        "object_name": names,
        "object_category": categories,
        "object_asset_source": sources,
        "object_visual_asset_path": visual_assets,
        "object_collision_asset_path": collision_assets,
        "object_visual_mesh_export_path": visual_mesh_exports,
        "object_collision_mesh_export_path": collision_mesh_exports,
        "object_mesh_export_dir": mesh_export_dirs,
        "object_mesh_export_format": mesh_export_formats,
        "mesh_assets_manifest_path": mesh_assets_manifest_path,
        "annotation_schema": np.array(annotation_summary["schema"]),
        "contains_2d_bboxes": np.array(True),
        "contains_pixel_masks": np.array(True),
        "videos_include_2d_bbox_and_pixel_mask_overlay": np.array(True),
        "camera_view_names": np.array(annotation_summary["camera_views"], dtype=str),
        "camera_intrinsics_json": camera_intrinsics_json,
        "annotation_manifest_path": annotation_manifest_path,
        "object_2d_bbox_xyxy_path_json": object_2d_bbox_xyxy_path_json,
        "object_pixel_mask_rle_path_json": object_pixel_mask_rle_path_json,
        "object_text_description": object_text_descriptions,
        "object_standard_description": object_standard_descriptions,
        "scene_text_description": np.array(scene_text_description),
        "scene_standard_description": np.array(scene_standard_description),
        "task_text_description": np.array(scene_text_description),
        "task_standard_description": np.array(scene_standard_description),
        "state_replay_import_contract_json": np.array(state_replay_import_contract),
        # Replay contract starts from the first recorded physics frame.  The originally
        # sampled pose/velocity are preserved separately so validation can compare replay
        # state without one-step initialization or contact-settling drift.
        "initial_object_pose_w": np.concatenate([pos[0], quat[0]], axis=-1),
        "initial_object_position_w": pos[0],
        "initial_object_quat_wxyz": quat[0],
        "initial_object_euler_xyz_rad": euler[0],
        "initial_object_lin_vel_w": lin[0],
        "initial_object_ang_vel_w": ang[0],
        "initial_object_height_m": pos[0, :, 2],
        "sampled_initial_object_pose_w": initial_pose,
        "sampled_initial_object_position_w": initial_pose[:, :3],
        "sampled_initial_object_quat_wxyz": initial_pose[:, 3:7],
        "sampled_initial_object_euler_xyz_rad": initial_euler,
        "sampled_initial_object_lin_vel_w": initial_lin_vel,
        "sampled_initial_object_ang_vel_w": initial_ang_vel,
        "sampled_initial_object_height_m": initial_height,
        "object_static_friction": static_friction,
        "object_dynamic_friction": dynamic_friction,
        "object_restitution": restitution,
        "disturbance_delta_lin_vel_w": disturbance_delta_lin_vel,
        "disturbance_delta_ang_vel_w": disturbance_delta_ang_vel,
        "bottom_z_minus_table_z": bottom,
        "bottom_z_minus_support_z": bottom,
        "object_over_table_xy": over_table,
        "object_over_support_xy": over_table,
        "active_support_z": active_support_z,
        "bottom_z_minus_active_support_z": bottom_active,
        "table_contact": table_contact,
        "support_contact": table_contact,
        "floor_contact": table_contact if support_surface == "indoor_floor" else ground_contact,
        "ground_contact_after_leaving_table": ground_contact,
        "pair_contact": pair_contact,
        "first_table_contact_step": np.array(first_table, np.int32),
        "first_support_contact_step": np.array(first_table, np.int32),
        "first_ground_contact_after_leaving_table_step": np.array(first_ground, np.int32),
        "final_pose_w": np.concatenate([pos[-1], quat[-1]], axis=-1),
        "object_metadata_json": np.array([json.dumps(x, sort_keys=True) for x in meta]),
    }
    np.savez_compressed(task_dir / "dataset.npz", **dataset)
    with (task_dir / "collision_events.jsonl").open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, sort_keys=True) + "\n")
    videos = {}
    for name, frames_ in video_frames.items():
        if frames_:
            path = task_dir / f"{task}_{name}.mp4"
            encode_video(path, frames_, max(1, args.fps // args.video_stride), keep_frames=bool(getattr(args, "keep_video_frames", False)))
            videos[name] = str(path)
    manifest = {
        "ok": bool(videos),
        "task": task,
        "dataset": str(task_dir / "dataset.npz"),
        "collision_events": str(task_dir / "collision_events.jsonl"),
        "videos": videos,
        "camera_views": annotation_summary["camera_views"],
        "contains_2d_bboxes": True,
        "contains_pixel_masks": True,
        "videos_include_2d_bbox_and_pixel_mask_overlay": True,
        "annotation_manifest": str(task_dir / annotation_summary["manifest_path"]),
        "annotation_summary": annotation_summary,
        "frames": frames,
        "captured_video_frames": {k: len(v) for k, v in video_frames.items()},
        "object_count": n,
        "support_surface": support_surface,
        "support_contact_event_type": support_contact_event_type,
        "scene_text_description": scene_text_description,
        "scene_standard_description": scene_standard_description,
        "state_replay_import_contract_json": state_replay_import_contract,
        "mesh_assets_manifest": str(task_dir / str(mesh_assets_manifest_path.item())),
        "mesh_export_count": int(len(mesh_export_summary["objects"])),
        "mesh_exports_present": bool(len(mesh_export_summary["objects"]) == n),
        "object_mesh_exports": mesh_export_summary["objects"],
        "object_text_descriptions": object_text_descriptions.tolist(),
        "object_standard_descriptions": object_standard_descriptions.tolist(),
        "single_object_episode": bool(task == "rolling_tabletop"),
        "robotwin_visual_and_collision_mesh_count": len([x for x in meta if str(x.get("source", "")) == "robotwin" and x.get("resolved_visual_asset_path") and x.get("collision_asset_path")]),
        "center_of_mass_offsets_m": center_of_mass_local.tolist(),
        "first_table_contact_step": first_table,
        "first_support_contact_step": first_table,
        "first_ground_contact_after_leaving_table_step": first_ground,
        "table_contact_event_count": len([e for e in events if e["type"] == "table_contact_begin"]),
        "floor_contact_event_count": len([e for e in events if e["type"] == "floor_contact_begin"]),
        "support_contact_event_count": len([e for e in events if e["type"] == support_contact_event_type]),
        "ground_contact_after_leaving_table_event_count": len([e for e in events if e["type"] == "ground_contact_begin_after_leaving_table"]),
        "pair_contact_event_count": len([e for e in events if e["type"] == "object_pair_near_contact"]),
        "max_linear_speed_m_s": float(np.linalg.norm(lin, axis=-1).max()),
        "max_angular_speed_rad_s": float(np.linalg.norm(ang, axis=-1).max()),
        "min_bottom_z_minus_table_z": float(bottom.min()),
        "min_bottom_z_minus_active_support_z": float(bottom_active.min()),
        "final_mean_speed_m_s": float(np.linalg.norm(lin[-1], axis=-1).mean()),
    }
    save_json(task_dir / "manifest.json", manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", choices=["falling_baton", "rolling_tabletop", "both"], default="both")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--video-stride", type=int, default=2)
    ap.add_argument("--keep-video-frames", action="store_true", help="Keep intermediate PNG frame folders after mp4 encoding for debugging.")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--falling-rods", type=int, default=5)
    ap.add_argument("--falling-disturbances-per-rod", type=int, default=3)
    ap.add_argument("--rolling-object-set", choices=["basic", "extended"], default="extended")
    ap.add_argument("--rolling-object-count", type=int, default=10)
    ap.add_argument("--rolling-single-object", action="store_true", help="Use exactly one object per rolling_tabletop episode; batch coverage should be done across repeated runs.")
    ap.add_argument("--rolling-disturbances-per-object", type=int, default=3, help="Number of random force/torque impulse proxies applied to each demo1 object.")
    ap.add_argument(
        "--rolling-category",
        default="random",
        choices=[
            "random",
            "sphere",
            "box",
            "bar",
            "l_shape",
            "bottle",
            "bowl",
            "plate",
            "hammer",
            "tray",
            "cup",
            "mug",
            "can",
            "woodenmallet",
            "dumbbell",
            "brush",
            "remotecontrol",
            "stapler",
            "markpen",
        ],
    )
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--renderer", default="RaytracedLighting")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.task == "both":
        run = {
            "ok": True,
            "out_dir": str(out),
            "tasks": {},
            "errors": [],
            "note": "Each task rendered in a separate IsaacSim process to avoid USD stage carry-over.",
        }
        common = [
            "--out-dir",
            str(out),
            "--frames",
            str(args.frames),
            "--fps",
            str(args.fps),
            "--video-stride",
            str(args.video_stride),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--falling-rods",
            str(args.falling_rods),
            "--falling-disturbances-per-rod",
            str(args.falling_disturbances_per_rod),
            "--rolling-object-set",
            str(args.rolling_object_set),
            "--rolling-object-count",
            str(args.rolling_object_count),
            "--rolling-category",
            str(args.rolling_category),
            "--rolling-disturbances-per-object",
            str(args.rolling_disturbances_per_object),
            "--seed",
            str(args.seed),
            "--renderer",
            str(args.renderer),
        ]
        if args.rolling_single_object:
            common.append("--rolling-single-object")
        if args.keep_video_frames:
            common.append("--keep-video-frames")
        for task in ("falling_baton", "rolling_tabletop"):
            log_path = out / f"{task}_run.log"
            cmd = [sys.executable, __file__, "--task", task, *common]
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                run["ok"] = False
                run["errors"].append({"task": task, "returncode": proc.returncode, "log": str(log_path)})
                continue
            manifest_path = out / task / "manifest.json"
            run["tasks"][task] = json.loads(manifest_path.read_text(encoding="utf-8"))
            run["ok"] = run["ok"] and bool(run["tasks"][task].get("ok"))
        save_json(out / "run_manifest.json", run)
        print(json.dumps(run, indent=2, sort_keys=True), flush=True)
        sys.exit(0 if run["ok"] else 2)
    run = {"ok": False, "out_dir": str(out), "tasks": {}, "errors": []}
    app = None
    try:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        os.environ.setdefault("ACCEPT_EULA", "Y")
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "width": args.width, "height": args.height, "renderer": args.renderer, "enable_cameras": True, "multi_gpu": False, "active_gpu": int(os.environ.get("ISAAC_ACTIVE_GPU", "0"))})
        from isaacsim.core.api import World
        from isaacsim.core.api.materials.physics_material import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCapsule, DynamicCuboid, DynamicSphere, FixedCuboid
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom, UsdLux

        imports = (World, PhysicsMaterial, DynamicCapsule, DynamicCuboid, DynamicSphere, FixedCuboid, Camera, UsdGeom, UsdLux)
        if args.task in ("falling_baton", "both"):
            run["tasks"]["falling_baton"] = run_falling(args, app, imports, out)
        if args.task in ("rolling_tabletop", "both"):
            run["tasks"]["rolling_tabletop"] = run_rolling(args, app, imports, out)
        run["ok"] = all(v.get("ok") for v in run["tasks"].values())
        save_json(out / "run_manifest.json", run)
        print(json.dumps(run, indent=2, sort_keys=True), flush=True)
        app.close()
        sys.exit(0 if run["ok"] else 2)
    except BaseException as exc:
        run["ok"] = False
        run["error"] = repr(exc)
        run["traceback"] = traceback.format_exc()
        save_json(out / "run_manifest.json", run)
        print(json.dumps(run, indent=2, sort_keys=True), flush=True)
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
        sys.exit(2)


if __name__ == "__main__":
    main()
