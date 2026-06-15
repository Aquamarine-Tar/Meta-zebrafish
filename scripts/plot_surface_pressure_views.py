#!/usr/bin/env python3
"""
每个时刻单独出图：3 行子图（+x 右视 / +y 俯视 / -x 左视），
鱼头在左、鱼尾在右（体轴 Z），仅显示朝向相机一侧的表面压力。

用法:
  python3 scripts/plot_surface_pressure_views.py --dir build/fsi_snapshots_ramp010
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, SymLogNorm

# 体轴 Z：head 在 z 较小一端 → 横轴为 Z，左=head，右=tail
BODY_AXIS = 2
HEAD_AT_MIN_Z = True

# (视图名, 法向分量轴, 符号, 垂直轴)
VIEWS = (
    ("+x view (right surface)", 0, +1, 1),   # horiz=Z, vert=Y
    ("+y view (top surface)", 1, +1, 0),    # horiz=Z, vert=X
    ("-x view (left surface)", 0, -1, 1),   # horiz=Z, vert=Y
)


def read_csv_dict(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out: dict[str, list] = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            out[k].append(int(v) if k in ("vtx", "f") else float(v))
    return {k: np.array(v) for k, v in out.items()}


def list_snapshot_times(snapshot_dir: Path) -> list[float]:
    pat = re.compile(r"^t([0-9.]+)_positions\.csv$")
    times = sorted({float(pat.match(p.name).group(1)) for p in snapshot_dir.iterdir() if pat.match(p.name)})
    if not times:
        raise FileNotFoundError(f"no snapshots in {snapshot_dir}")
    return times


def load_snapshot(snapshot_dir: Path, sim_t: float):
    tag = f"t{sim_t:.2f}"
    pos = read_csv_dict(snapshot_dir / f"{tag}_positions.csv")
    forces = read_csv_dict(snapshot_dir / f"{tag}_forces.csv")
    vertices = np.stack([pos["x"], pos["y"], pos["z"]], axis=1)
    hydro = np.stack([forces["hx"], forces["hy"], forces["hz"]], axis=1)
    return vertices, hydro


def load_meta_com(snapshot_dir: Path, sim_t: float) -> np.ndarray | None:
    meta_path = snapshot_dir / f"t{sim_t:.2f}_meta.txt"
    if not meta_path.exists():
        return None
    for line in meta_path.read_text().splitlines():
        if line.startswith("com="):
            vals = [float(x) for x in line.split("=", 1)[1].split()]
            if len(vals) == 3:
                return np.array(vals)
    return None


def surface_nonzero_hydro_ratio(
    hydro: np.ndarray,
    faces: np.ndarray,
    eps: float = 1e-12,
) -> tuple[int, int, float]:
    """表面顶点 |F_hydro|>eps 的数量 / 表面顶点总数"""
    surf = set(faces.ravel().tolist())
    mag = np.linalg.norm(hydro[list(surf)], axis=1)
    nonzero = int(np.sum(mag > eps))
    total = len(surf)
    pct = 100.0 * nonzero / max(total, 1)
    return nonzero, total, pct


def load_faces(snapshot_dir: Path) -> np.ndarray:
    d = read_csv_dict(snapshot_dir / "surface_faces.csv")
    return np.stack([d["i0"], d["i1"], d["i2"]], axis=1).astype(int)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(vertices)
    for f in faces:
        p0, p1, p2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        fn = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(fn)
        if ln < 1e-14:
            continue
        fn /= ln
        acc[f[0]] += fn
        acc[f[1]] += fn
        acc[f[2]] += fn
    ln = np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    return acc / ln


def compute_pressure_field(vertices: np.ndarray, faces: np.ndarray, hydro: np.ndarray) -> np.ndarray:
    """signed pressure proxy: (F_hydro · n) / A_vtx"""
    normals = vertex_normals(vertices, faces)
    fn = np.sum(hydro * normals, axis=1)
    v_area = np.zeros(len(vertices))
    for f in faces:
        p0, p1, p2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        share = area / 3.0
        v_area[f[0]] += share
        v_area[f[1]] += share
        v_area[f[2]] += share
    return fn / np.maximum(v_area, 1e-10)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-14)
    return n / ln


def project_point(p: np.ndarray, horiz: int, vert: int) -> tuple[float, float]:
    """体轴 Z 横放：head 在左。若 head 在 min(z)，直接 z 作横轴。"""
    h = p[horiz]
    if horiz == BODY_AXIS and HEAD_AT_MIN_Z:
        pass  # z 小在左
    return h, p[vert]


def plot_view_panel(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    p_field: np.ndarray,
    view_axis: int,
    view_sign: int,
    vert_axis: int,
    title: str,
    norm: Normalize,
    normal_thresh: float = 0.35,
):
    """仅绘制法向朝向相机一侧的三角面，按面心压力着色。"""
    fn = face_normals(vertices, faces)
    visible = view_sign * fn[:, view_axis] > normal_thresh

    polys = []
    vals = []
    horiz = BODY_AXIS
    for f, ok in zip(faces, visible):
        if not ok:
            continue
        tri = vertices[f]
        poly = np.array([project_point(tri[i], horiz, vert_axis) for i in range(3)])
        polys.append(poly)
        vals.append(p_field[f].mean())

    if polys:
        vals_arr = np.array(vals)
        colors = cm.RdBu_r(norm(vals_arr))
        ax.add_collection(
            PolyCollection(
                polys,
                facecolors=colors,
                edgecolors=(0, 0, 0, 0.08),
                linewidths=0.12,
            )
        )
        xs = np.concatenate([p[:, 0] for p in polys])
        ys = np.concatenate([p[:, 1] for p in polys])
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
    else:
        ax.text(0.5, 0.5, "no visible faces", ha="center", va="center", transform=ax.transAxes)

    ax.set_aspect("equal")
    ax.set_xlabel("Z [m]  (head $\\leftarrow$  tail $\\rightarrow$)")
    vert_name = ["X", "Y", "Z"][vert_axis]
    ax.set_ylabel(f"{vert_name} [m]")
    ax.set_title(title, fontsize=10)
    # head 在左：min z 在左侧（默认 xlim 已满足）


def phase_label(t: float, cycle: float = 1.0) -> str:
    for n in range(1, 7):
        if abs(t - n / 6.0 * cycle) < 0.025:
            return f"{n}/6 T  (t={t:.2f}s)"
    return f"t={t:.2f}s"


def main():
    parser = argparse.ArgumentParser(description="Per-time surface pressure orthographic views")
    parser.add_argument("--dir", type=Path, default=Path("build/fsi_snapshots_ramp010"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cycle", type=float, default=1.0)
    parser.add_argument("--normal-thresh", type=float, default=0.35)
    parser.add_argument(
        "--linthresh",
        type=float,
        default=0.0,
        help="SymLog 线性区阈值；0=自动取可见面 |p| 的 15%% 分位",
    )
    args = parser.parse_args()

    snap_dir = args.dir
    out_dir = args.out_dir or snap_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    faces = load_faces(snap_dir)
    times = list_snapshot_times(snap_dir)

    # 全局色标：所有时刻、所有可见面压力
    all_vals = []
    cache = []
    for t in times:
        v, h = load_snapshot(snap_dir, t)
        p = compute_pressure_field(v, faces, h)
        fn = face_normals(v, faces)
        cache.append((t, v, p, fn))
        for _, view_axis, view_sign, vert_axis in VIEWS:
            vis = view_sign * fn[:, view_axis] > args.normal_thresh
            if vis.any():
                all_vals.append(p[faces[vis]].mean(axis=1))
    combined = np.concatenate(all_vals) if all_vals else np.array([0.0])
    abs_combined = np.abs(combined)
    vmax = max(float(np.percentile(abs_combined, 99.5)), 1e-3)
    if args.linthresh > 0:
        linthresh = args.linthresh
    else:
        linthresh = max(float(np.percentile(abs_combined, 15)), 1.0)
        linthresh = min(linthresh, vmax * 0.2)
    norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
    print(f"color scale: SymLog  linthresh={linthresh:.3g}  vmax={vmax:.3g}")

    for t, v, p_field, _fn in cache:
        lbl = phase_label(t, args.cycle)
        _, h = load_snapshot(snap_dir, t)
        nz, nt, pct = surface_nonzero_hydro_ratio(h, faces)
        com = load_meta_com(snap_dir, t)
        if com is not None:
            print(
                f"t={t:.2f}s  surface_nonzero_hydro={nz}/{nt} ({pct:.1f}%)  "
                f"com=({com[0]:.6f},{com[1]:.6f},{com[2]:.6f})"
            )
        else:
            com = v.mean(axis=0)
            print(
                f"t={t:.2f}s  surface_nonzero_hydro={nz}/{nt} ({pct:.1f}%)  "
                f"com=({com[0]:.6f},{com[1]:.6f},{com[2]:.6f}) [vertex mean, no meta]"
            )

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        for ax, (vtitle, vaxis, vsign, vert_axis) in zip(axes, VIEWS):
            plot_view_panel(
                ax, v, faces, p_field,
                view_axis=vaxis,
                view_sign=vsign,
                vert_axis=vert_axis,
                title=vtitle,
                norm=norm,
                normal_thresh=args.normal_thresh,
            )

        sm = cm.ScalarMappable(cmap=cm.RdBu_r, norm=norm)
        sm.set_array([])

        fig.suptitle(
            f"{lbl}\nHead left, tail right  |  each panel: pressure on the face toward the viewer",
            fontsize=12,
            y=0.98,
        )
        # 先留右侧给 colorbar，避免色标压住鱼体
        fig.subplots_adjust(left=0.09, right=0.74, top=0.91, bottom=0.06, hspace=0.32)
        cbar_ax = fig.add_axes([0.77, 0.12, 0.022, 0.72])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(
            r"Surface pressure (SymLog)  $(F_h \cdot \hat{n})/A_{vtx}$  (+P / -S)",
            fontsize=9,
        )

        out_path = out_dir / f"pressure_views_t{t:.2f}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
