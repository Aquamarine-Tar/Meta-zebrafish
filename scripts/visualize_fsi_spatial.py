#!/usr/bin/env python3
"""
FSI 力在鱼体空间的分布可视化（比 3D mesh 着色更易读）

输出布局（每个时刻一行，共 5 行）：
  [1] 体轴展开图  Z × θ（θ=绕体轴方位角，颜色=|F|）
  [2] 侧视 Y–Z（颜色=|F|，看 pitch / +Y 分布）
  [3] 侧视 Y–Z（发散色=F_y，正=+Y 推力）
  [4] 侧视力矢量（子采样箭头，(F_z,F_y) 在 Z–Y 平面）

用法:
  python3 scripts/visualize_fsi_spatial.py --dir build/fsi_snapshots --kind total
  python3 scripts/visualize_fsi_spatial.py --dir build/fsi_snapshots --kind hydro --per-time
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm


def read_csv_dict(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    cols = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k in cols:
            cols[k].append(float(row[k]) if k not in ("vtx", "f") else int(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


def load_faces(snapshot_dir: Path) -> np.ndarray:
    d = read_csv_dict(snapshot_dir / "surface_faces.csv")
    return np.stack([d["i0"], d["i1"], d["i2"]], axis=1).astype(int)


def force_vec(forces: dict[str, np.ndarray], kind: str) -> np.ndarray:
    if kind == "contact":
        return np.stack([forces["cx"], forces["cy"], forces["cz"]], axis=1)
    if kind == "hydro":
        return np.stack([forces["hx"], forces["hy"], forces["hz"]], axis=1)
    return np.stack([forces["fx"], forces["fy"], forces["fz"]], axis=1)


def surface_vertex_indices(faces: np.ndarray) -> np.ndarray:
    return np.unique(faces.ravel())


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray, n_vtx: int) -> np.ndarray:
    """面积加权顶点法向（用于 F·n 分量）。"""
    vn = np.zeros((n_vtx, 3))
    for tri in faces:
        p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        if ln < 1e-14:
            continue
        n /= ln
        for idx in tri:
            vn[idx] += n
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    vn /= np.maximum(ln, 1e-12)
    return vn


def binned_mean(z: np.ndarray, theta: np.ndarray, values: np.ndarray, z_bins: int, t_bins: int):
    """在 (Z, θ) 网格上对 values 求平均，用于展开图。"""
    t_min, t_max = -np.pi, np.pi
    z_lo, z_hi = z.min(), z.max()
    sum_v, _, _ = np.histogram2d(z, theta, bins=[z_bins, t_bins], range=[[z_lo, z_hi], [t_min, t_max]], weights=values)
    cnt, z_edges, t_edges = np.histogram2d(z, theta, bins=[z_bins, t_bins], range=[[z_lo, z_hi], [t_min, t_max]])
    mean = sum_v / np.maximum(cnt, 1)
    mean[cnt == 0] = np.nan
    return mean, z_edges, t_edges


def load_snapshot(snap_dir: Path, tag: str):
    pos = read_csv_dict(snap_dir / f"{tag}_positions.csv")
    forces = read_csv_dict(snap_dir / f"{tag}_forces.csv")
    vertices = np.stack([pos["x"], pos["y"], pos["z"]], axis=1)
    return vertices, forces


def plot_spatial_row(
    axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    fvec: np.ndarray,
    vmax_mag: float,
    vmax_fy: float,
    tag: str,
    sim_t: float,
):
    """一行四列：展开 |F| / 侧视|F| / 侧视F_y / 侧视矢量。"""
    ax_u, ax_mag, ax_fy, ax_vec = axes
    n_vtx = len(vertices)
    surf = surface_vertex_indices(faces)
    p = vertices[surf]
    f = fvec[surf]
    mag = np.linalg.norm(f, axis=1)
    fy = f[:, 1]

    # 体轴 +Z：用表面点 xy 质心作为截圆心
    x_c, y_c = p[:, 0].mean(), p[:, 1].mean()
    theta = np.arctan2(p[:, 1] - y_c, p[:, 0] - x_c)
    z = p[:, 2]

    # --- [1] Z–θ 展开 ---
    grid, z_edges, t_edges = binned_mean(z, theta, mag, z_bins=48, t_bins=48)
    im = ax_u.pcolormesh(
        t_edges,
        z_edges,
        grid,
        cmap="inferno",
        norm=Normalize(0, vmax_mag),
        shading="auto",
    )
    ax_u.set_xlabel(r"$\theta$ [rad] (around +Z)")
    ax_u.set_ylabel("Z [m] (body axis)")
    ax_u.set_title(f"{tag}  t={sim_t:.2f}s  unwrap |F|")
    ax_u.axhline(y_c, color="w", ls=":", lw=0.4, alpha=0.5)

    # --- [2] 侧视 Y–Z |F| ---
    sc1 = ax_mag.scatter(
        z, p[:, 1], c=mag, s=3, cmap="inferno", norm=Normalize(0, vmax_mag), rasterized=True
    )
    ax_mag.set_xlabel("Z [m]")
    ax_mag.set_ylabel("Y [m]")
    ax_mag.set_title("side view  |F|")
    ax_mag.set_aspect("equal", adjustable="box")

    # --- [3] 侧视 F_y（+Y 向上推为红）---
    norm_fy = TwoSlopeNorm(vmin=-vmax_fy, vcenter=0.0, vmax=vmax_fy)
    ax_fy.scatter(z, p[:, 1], c=fy, s=3, cmap="RdBu_r", norm=norm_fy, rasterized=True)
    ax_fy.set_xlabel("Z [m]")
    ax_fy.set_ylabel("Y [m]")
    ax_fy.set_title(r"side view  $F_y$ (+Y up)")
    ax_fy.set_aspect("equal", adjustable="box")

    # --- [4] 侧视力矢量 (F_z, F_y) @ (Z, Y) ---
    step = max(1, len(surf) // 400)
    idx = surf[::step]
    p_sub = vertices[idx]
    f_sub = fvec[idx]
    fz, fyy = f_sub[:, 2], f_sub[:, 1]
    scale = 0.015 / (np.percentile(np.hypot(fz, fyy), 95) + 1e-9)
    ax_vec.quiver(
        p_sub[:, 2],
        p_sub[:, 1],
        fz * scale,
        fyy * scale,
        np.linalg.norm(f_sub, axis=1),
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap="viridis",
        width=0.002,
        alpha=0.85,
    )
    ax_vec.set_xlabel("Z [m]")
    ax_vec.set_ylabel("Y [m]")
    ax_vec.set_title("side vectors  (F_z, F_y)")
    ax_vec.set_aspect("equal", adjustable="box")

    return im, sc1


def render_figure(snap_dir: Path, kind: str, tags: list[str], out_path: Path, per_time: bool):
    faces = load_faces(snap_dir)
    sim_times = [float(t[1:]) for t in tags]

    # 全局色标
    mags, fys = [], []
    for tag in tags:
        if not (snap_dir / f"{tag}_forces.csv").exists():
            continue
        vertices, forces = load_snapshot(snap_dir, tag)
        fvec = force_vec(forces, kind)
        surf = surface_vertex_indices(faces)
        mags.append(np.linalg.norm(fvec[surf], axis=1).max())
        fys.append(np.abs(fvec[surf, 1]).max())
    vmax_mag = max(mags) if mags else 1.0
    vmax_fy = max(fys) if fys else 1.0

    for tag, sim_t in zip(tags, sim_times):
        fpath = snap_dir / f"{tag}_forces.csv"
        if not fpath.exists():
            continue
        vertices, forces = load_snapshot(snap_dir, tag)
        fvec = force_vec(forces, kind)

        if per_time:
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
            plot_spatial_row(axes, vertices, faces, fvec, vmax_mag, vmax_fy, tag, sim_t)
            fig.subplots_adjust(right=0.92)
            cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
            sm = cm.ScalarMappable(norm=Normalize(0, vmax_mag), cmap="inferno")
            sm.set_array([])
            fig.colorbar(sm, cax=cax, label=f"|F_{kind}| [N]")
            fig.suptitle(f"FSI spatial distribution ({kind})  t={sim_t:.2f}s", fontsize=11)
            pt_out = snap_dir / f"fsi_spatial_{tag}_{kind}.png"
            fig.savefig(pt_out, dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {pt_out}")

    # 汇总大图：5 行 × 4 列
    n = sum(1 for t in tags if (snap_dir / f"{t}_forces.csv").exists())
    fig, axes = plt.subplots(n, 4, figsize=(18, 3.2 * n), squeeze=False)
    row = 0
    for tag, sim_t in zip(tags, sim_times):
        fpath = snap_dir / f"{tag}_forces.csv"
        if not fpath.exists():
            continue
        vertices, forces = load_snapshot(snap_dir, tag)
        fvec = force_vec(forces, kind)
        plot_spatial_row(axes[row], vertices, faces, fvec, vmax_mag, vmax_fy, tag, sim_t)
        row += 1

    fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.28)
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.76])
    sm = cm.ScalarMappable(norm=Normalize(0, vmax_mag), cmap="inferno")
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=f"|F_{kind}| [N]")
    fig.suptitle(
        f"FSI force on fish surface ({kind}) — unwrap Z×θ & side Y–Z\n"
        r"columns: unwrap |F| | side |F| | side $F_y$ | side vectors ($F_z$,$F_y$)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Spatial FSI force visualization")
    parser.add_argument("--dir", type=Path, default=Path("build/fsi_snapshots"))
    parser.add_argument("--kind", choices=("total", "contact", "hydro"), default="total")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--per-time", action="store_true", help="also save one PNG per snapshot time")
    args = parser.parse_args()

    tags = ["t2.00", "t2.25", "t2.50", "t2.75", "t3.00"]
    out = args.out or (args.dir / f"fsi_spatial_{args.kind}.png")
    render_figure(args.dir, args.kind, tags, out, args.per_time)


if __name__ == "__main__":
    main()
