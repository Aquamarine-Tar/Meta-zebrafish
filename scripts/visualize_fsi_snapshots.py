#!/usr/bin/env python3
"""读取 fsi_force_snapshot 输出，生成表面力模长着色图（5 个时刻）。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_csv_dict(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    cols = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k in cols:
            cols[k].append(float(row[k]) if k != "vtx" and k != "f" else int(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


def read_meta(path: Path) -> dict[str, str]:
    meta = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def load_faces(snapshot_dir: Path) -> np.ndarray:
    d = read_csv_dict(snapshot_dir / "surface_faces.csv")
    return np.stack([d["i0"], d["i1"], d["i2"]], axis=1).astype(int)


def force_magnitude(forces: dict[str, np.ndarray], kind: str) -> np.ndarray:
    if kind == "contact":
        f = np.stack([forces["cx"], forces["cy"], forces["cz"]], axis=1)
    elif kind == "hydro":
        f = np.stack([forces["hx"], forces["hy"], forces["hz"]], axis=1)
    else:
        f = np.stack([forces["fx"], forces["fy"], forces["fz"]], axis=1)
    return np.linalg.norm(f, axis=1)


def render_mesh_force(ax, vertices, faces, vtx_mag, title, vmax):
    face_vals = vtx_mag[faces].mean(axis=1)
    polys = vertices[faces]
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-12))
    colors = cm.plasma(norm(face_vals))
    mesh = Poly3DCollection(polys, facecolors=colors, edgecolors=(0, 0, 0, 0.05), linewidths=0.1)
    ax.add_collection3d(mesh)
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=20, azim=-60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("build/fsi_snapshots"))
    parser.add_argument("--kind", choices=("total", "contact", "hydro"), default="total")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    snap_dir = args.dir
    times = ["t2.00", "t2.25", "t2.50", "t2.75", "t3.00"]
    faces = load_faces(snap_dir)

    mags = []
    for tag in times:
        fpath = snap_dir / f"{tag}_forces.csv"
        if fpath.exists():
            mags.append(force_magnitude(read_csv_dict(fpath), args.kind).max())
    vmax = max(mags) if mags else 1.0

    fig = plt.figure(figsize=(16, 9))
    summary_path = snap_dir / "summary.csv"
    summary = read_csv_dict(summary_path) if summary_path.exists() else {}

    for i, tag in enumerate(times):
        fpath = snap_dir / f"{tag}_forces.csv"
        if not fpath.exists():
            continue
        pos = read_csv_dict(snap_dir / f"{tag}_positions.csv")
        forces = read_csv_dict(fpath)
        meta = read_meta(snap_dir / f"{tag}_meta.txt")
        vertices = np.stack([pos["x"], pos["y"], pos["z"]], axis=1)
        mag = force_magnitude(forces, args.kind)
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        tau = meta.get("tau_x_surface_about_com", "?")
        title = f"{tag}  |F|max={mag.max():.4f} N\nτ_x(COM)={tau} N·m"
        render_mesh_force(ax, vertices, faces, mag, title, vmax)

    ax6 = fig.add_subplot(2, 3, 6)
    if len(summary):
        ax6.plot(summary["sim_time"], summary["tau_x_surface_com"], "o-", label="τ_x @ COM (surface)")
        ax6.plot(summary["sim_time"], summary["tau_x_surface_center"], "s--", label="τ_x @ center (surface)")
        ax6.set_xlabel("sim time [s]")
        ax6.set_ylabel("torque_x [N·m]")
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_title("绕 x 轴合力矩")
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(norm=Normalize(0, vmax), cmap="plasma")
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=f"|F_{args.kind}| [N]")
    fig.suptitle(f"FSI surface force ({args.kind})", fontsize=12)
    out = args.out or (snap_dir / f"fsi_force_{args.kind}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
