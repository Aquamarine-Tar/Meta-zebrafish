#!/usr/bin/env python3
"""
Fig.10 风格：游动周期内鱼体形态与表面「压力代理」（水压力沿法向分量）的关系。

用法:
  python3 scripts/plot_fig10_body_pressure_cycle.py \\
      --dir build/fsi_snapshots_ramp010 \\
      --cycle 1.0 \\
      --phases 1/6 4/6 \\
      --out build/fsi_snapshots_ramp010/fig10_body_pressure.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import TwoSlopeNorm


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


def load_faces(snapshot_dir: Path) -> np.ndarray:
    d = read_csv_dict(snapshot_dir / "surface_faces.csv")
    return np.stack([d["i0"], d["i1"], d["i2"]], axis=1).astype(int)


def nearest_time_tag(snapshot_dir: Path, t_target: float) -> tuple[float, str]:
    """在目录中找最接近 t_target 的快照 tag（如 t0.67）。"""
    pat = re.compile(r"^t([0-9.]+)_(positions|forces)\.csv$")
    times: set[float] = set()
    for p in snapshot_dir.iterdir():
        m = pat.match(p.name)
        if m:
            times.add(float(m.group(1)))
    if not times:
        raise FileNotFoundError(f"no snapshots in {snapshot_dir}")
    t_best = min(times, key=lambda t: abs(t - t_target))
    return t_best, f"t{t_best:.2f}"


def load_snapshot(snapshot_dir: Path, sim_t: float):
    tag = f"t{sim_t:.2f}"
    pos_path = snapshot_dir / f"{tag}_positions.csv"
    force_path = snapshot_dir / f"{tag}_forces.csv"
    if not pos_path.exists():
        t_best, tag = nearest_time_tag(snapshot_dir, sim_t)
        sim_t = t_best
        pos_path = snapshot_dir / f"{tag}_positions.csv"
        force_path = snapshot_dir / f"{tag}_forces.csv"
    pos = read_csv_dict(pos_path)
    forces = read_csv_dict(force_path)
    vertices = np.stack([pos["x"], pos["y"], pos["z"]], axis=1)
    hydro = np.stack([forces["hx"], forces["hy"], forces["hz"]], axis=1)
    total = np.stack([forces["fx"], forces["fy"], forces["fz"]], axis=1)
    return sim_t, vertices, hydro, total


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """表面顶点外法向（面积加权平均）。"""
    n_v = len(vertices)
    acc = np.zeros((n_v, 3))
    for f in faces:
        i0, i1, i2 = f
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
        fn = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(fn)
        if ln < 1e-14:
            continue
        fn /= ln
        acc[i0] += fn
        acc[i1] += fn
        acc[i2] += fn
    norms = np.linalg.norm(acc, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return acc / norms


def compute_pressure_field(vertices: np.ndarray, faces: np.ndarray, hydro: np.ndarray) -> np.ndarray:
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
    v_area = np.maximum(v_area, 1e-10)
    return fn / v_area  # 近似 Pa（量级用于着色）


def plot_projected_mesh(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    surf_mask: np.ndarray,
    proj: tuple[int, int],
    title: str,
    norm: TwoSlopeNorm,
    xlabel: str,
    ylabel: str,
):
    """2D 投影三角面片，按面心标量着色。"""
    ix, iy = proj
    polys = []
    face_vals = []
    for f in faces:
        if not np.all(surf_mask[f]):
            continue
        tri = vertices[f][:, [ix, iy]]
        polys.append(tri)
        face_vals.append(values[f].mean())
    if not polys:
        ax.set_title(title + " (empty)")
        return
    face_vals = np.array(face_vals)
    colors = cm.RdBu_r(norm(face_vals))
    coll = PolyCollection(polys, facecolors=colors, edgecolors=(0, 0, 0, 0.06), linewidths=0.15)
    ax.add_collection(coll)
    xs = vertices[surf_mask][:, ix]
    ys = vertices[surf_mask][:, iy]
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)


def plot_side_deformation_forces(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    hydro: np.ndarray,
    surf: np.ndarray,
    y_com: float,
    title: str,
    z_markers: list[tuple[float, str, str]] | None = None,
):
    """侧视 Y–Z：鱼体轮廓 + 表面水压力箭头（jet/推力方向示意）。"""
    # 轮廓：表面三角边在 Y–Z 投影
    segs = []
    for f in faces:
        tri = vertices[f]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            segs.append(tri[[a, b]][:, [1, 2]])
    lc = LineCollection(segs, colors=(0.35, 0.35, 0.35, 0.25), linewidths=0.25)
    ax.add_collection(lc)

    p = vertices[surf]
    h = hydro[surf]
    # 子采样，避免箭头过密
    step = max(1, len(p) // 180)
    y, z = p[::step, 1], p[::step, 2]
    hy, hz = h[::step, 1], h[::step, 2]
    mag = np.sqrt(hy**2 + hz**2)
    scale = 0.015 / max(np.percentile(mag, 95), 1e-6)
    ax.quiver(
        z, y, hz * scale, hy * scale,
        angles="xy", scale_units="xy", scale=1,
        color="0.15", width=0.002, headwidth=3, alpha=0.65,
    )

    ax.set_xlabel("Z [m] (body axis)")
    ax.set_ylabel("Y [m]")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axhline(y_com, color="0.6", ls=":", lw=0.8)

    if z_markers:
        for z0, lbl, c in z_markers:
            ax.axvline(z0, color=c, ls="--", lw=0.9, alpha=0.75)
            ax.text(z0, ax.get_ylim()[1], lbl, ha="center", va="bottom", fontsize=7, color=c)


def main():
    parser = argparse.ArgumentParser(description="Fig.10 风格 形态–表面压力关系图")
    parser.add_argument("--dir", type=Path, default=Path("build/fsi_snapshots_ramp010"))
    parser.add_argument("--cycle", type=float, default=1.0, help="游动周期 T [s]")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["1/6", "4/6"],
        help='周期相位，如 1/6 4/6 或 0.25 0.5',
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    def parse_phase(s: str) -> float:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b) * args.cycle
        return float(s)

    t_targets = [parse_phase(p) for p in args.phases]
    snap_dir = args.dir
    faces = load_faces(snap_dir)
    surf = np.unique(faces.ravel())

    snapshots = []
    all_p = []
    for t_tgt in t_targets:
        t_act, v, h, tot = load_snapshot(snap_dir, t_tgt)
        p_field = compute_pressure_field(v, faces, h)
        snapshots.append((t_tgt, t_act, v, h, tot, p_field))
        all_p.append(p_field[surf])

    combined = np.concatenate(all_p)
    vmax = float(np.percentile(np.abs(combined), 98))
    vmax = max(vmax, 1e-3)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_cols = len(snapshots)
    fig, axes = plt.subplots(3, n_cols, figsize=(5.2 * n_cols, 11))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    phase_labels = []
    for col, (t_tgt, t_act, v, h, tot, p_field) in enumerate(snapshots):
        y_com = v[:, 1].mean()
        dorsal = v[:, 1] >= y_com
        ventral = v[:, 1] < y_com
        surf_d = dorsal & np.isin(np.arange(len(v)), surf)
        surf_v = ventral & np.isin(np.arange(len(v)), surf)

        z = v[:, 2]
        p_dorsal = np.full(len(v), np.nan)
        p_dorsal[surf_d] = p_field[surf_d]
        idx_p = int(np.nanargmax(p_dorsal))
        idx_s = int(np.nanargmin(p_dorsal))
        z_markers = [
            (z[idx_p], "P", "#c0392b"),
            (z[idx_s], "S", "#2980b9"),
        ]

        # 相位标签（如 1/6 T）
        phase_lbl = None
        for num in (1, 2, 3, 4, 5, 6):
            if abs(t_tgt - num / 6.0 * args.cycle) < 0.02:
                phase_lbl = f"{num}/6 T"
                break
        if phase_lbl is None:
            phase_lbl = f"t={t_act:.2f}s"
        phase_labels.append(phase_lbl)

        # 行 0：背侧俯视 (X–Z)，P/S 色图
        plot_projected_mesh(
            axes[0, col], v, faces, p_field, surf_d,
            proj=(0, 2), title=f"Dorsal  P(+)/S(-)  [{phase_lbl}]",
            norm=norm, xlabel="X [m]", ylabel="Z [m]",
        )

        plot_side_deformation_forces(
            axes[1, col], v, faces, h, surf, y_com,
            title=f"Side: shape + hydro force  [{phase_lbl}]",
            z_markers=z_markers,
        )

        plot_projected_mesh(
            axes[2, col], v, faces, p_field, surf_v,
            proj=(0, 2), title=f"Ventral  P(+)/S(-)  [{phase_lbl}]",
            norm=norm, xlabel="X [m]", ylabel="Z [m]",
        )

        # 列标题
        axes[0, col].text(
            0.5, 1.06, phase_lbl, transform=axes[0, col].transAxes,
            ha="center", fontsize=11, fontweight="bold",
        )

    # 统一 colorbar
    sm = cm.ScalarMappable(cmap=cm.RdBu_r, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.55, pad=0.02)
    cbar.set_label(r"Pressure proxy  $F_h \cdot \hat{n}/A_{vtx}$  (+P / -S)")

    fig.suptitle(
        "Body shape vs surface pressure within one swim cycle (Fig.10 style)\n"
        f"Data: {snap_dir.name}  |  red=high P, blue=suction S  |  dashed lines mark P/S along body axis",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.94])

    out = args.out or (snap_dir / "fig10_body_pressure_cycle.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
