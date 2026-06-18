#!/usr/bin/env python3
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def read_positions(path):
    rows = list(csv.DictReader(open(path, newline="")))
    pos = np.zeros((len(rows), 3), dtype=float)
    for r in rows:
        i = int(r["vtx"])
        pos[i] = (float(r["x"]), float(r["y"]), float(r["z"]))
    return pos


def read_forces(path):
    rows = list(csv.DictReader(open(path, newline="")))
    hydro = np.zeros((len(rows), 3), dtype=float)
    for r in rows:
        i = int(r["vtx"])
        hydro[i] = (float(r["hx"]), float(r["hy"]), float(r["hz"]))
    return hydro


def read_faces(path):
    faces = []
    for r in csv.DictReader(open(path, newline="")):
        faces.append((int(r["i0"]), int(r["i1"]), int(r["i2"])))
    return np.asarray(faces, dtype=int)


def equal_axes(ax, x, y):
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    r = 0.52 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_aspect("equal", adjustable="box")


def plot_snapshot(out_dir, tag, pos, faces, hydro, hz_lim):
    hmag = np.linalg.norm(hydro, axis=1)
    nonzero = hmag > 1e-12
    pct = 100.0 * np.count_nonzero(nonzero) / max(len(nonzero), 1)

    projections = [
        ("zy", pos[:, 2], pos[:, 1], "z", "y"),
        ("zx", pos[:, 2], pos[:, 0], "z", "x"),
    ]
    for proj_name, px, py, xlabel, ylabel in projections:
        tri = mtri.Triangulation(px, py, faces)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
        c0 = axes[0].tripcolor(
            tri, hydro[:, 2], shading="gouraud", cmap="coolwarm",
            vmin=-hz_lim, vmax=hz_lim)
        axes[0].set_title(f"{tag} hydro z-force, {proj_name}")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        equal_axes(axes[0], px, py)
        fig.colorbar(c0, ax=axes[0], label="hydro force z component (N)")

        vmax = np.percentile(hmag[nonzero], 98) if np.any(nonzero) else 1.0
        vmax = max(vmax, 1e-8)
        c1 = axes[1].tripcolor(
            tri, hmag, shading="gouraud", cmap="viridis",
            vmin=0.0, vmax=vmax)
        axes[1].set_title(f"{tag} hydro |force|, nonzero={pct:.1f}%")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        equal_axes(axes[1], px, py)
        fig.colorbar(c1, ax=axes[1], label="hydro force magnitude (N)")

        path = os.path.join(out_dir, f"{tag}_pressure_{proj_name}.png")
        fig.savefig(path, dpi=220)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot_dir")
    ap.add_argument("--tags", nargs="+", default=["t0.50", "t1.00", "t1.50"])
    args = ap.parse_args()

    faces = read_faces(os.path.join(args.snapshot_dir, "surface_faces.csv"))
    all_hz = []
    data = []
    for tag in args.tags:
        pos = read_positions(os.path.join(args.snapshot_dir, f"{tag}_positions.csv"))
        hydro = read_forces(os.path.join(args.snapshot_dir, f"{tag}_forces.csv"))
        data.append((tag, pos, hydro))
        all_hz.append(hydro[:, 2])

    hz = np.concatenate(all_hz)
    hz_lim = np.percentile(np.abs(hz), 98)
    hz_lim = max(float(hz_lim), 1e-8)

    for tag, pos, hydro in data:
        plot_snapshot(args.snapshot_dir, tag, pos, faces, hydro, hz_lim)


if __name__ == "__main__":
    main()
