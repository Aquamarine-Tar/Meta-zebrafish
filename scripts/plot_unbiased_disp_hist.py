#!/usr/bin/env python3
"""
读取 fsi_unbiased_disp 输出的 CSV，对 t∈[2,3]s 的无偏位移（相对 t=2s 参考）做直方图对比。

用法:
  python3 scripts/plot_unbiased_disp_hist.py --dir build/fsi_disp_compare
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {k: np.array([float(r[k]) for r in rows]) for k in keys}
    return out


def displacement_from_ref(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """相对首样本（≈t=2s）的无偏位移。"""
    ref = np.array([data["ub_x"][0], data["ub_y"][0], data["ub_z"][0]])
    ub = np.stack([data["ub_x"], data["ub_y"], data["ub_z"]], axis=1)
    d = ub - ref
    mag = np.linalg.norm(d, axis=1)
    return d[:, 0], d[:, 1], d[:, 2], mag


def plot_compare(
    out_path: Path,
    both: dict[str, np.ndarray],
    hydro: dict[str, np.ndarray],
    xlim: tuple[float, float],
):
    dx_b, dy_b, dz_b, mag_b = displacement_from_ref(both)
    dx_h, dy_h, dz_h, mag_h = displacement_from_ref(hydro)
    bins = np.linspace(xlim[0], xlim[1], 45)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        (axes[0, 0], dx_b, dx_h, r"$\Delta x$ [m]", "x"),
        (axes[0, 1], dy_b, dy_h, r"$\Delta y$ [m]", "y"),
        (axes[1, 0], dz_b, dz_h, r"$\Delta z$ [m]", "z"),
        (axes[1, 1], mag_b, mag_h, r"$|\Delta \mathbf{t}|$ [m]", "|t|"),
    ]

    for ax, db, dh, xlab, tag in panels:
        ax.hist(db, bins=bins, alpha=0.55, color="C0", label=f"both (n={len(db)})", edgecolor="none")
        ax.hist(dh, bins=bins, alpha=0.55, color="C2", label=f"hydro-only (n={len(dh)})", edgecolor="none")
        ax.axvline(float(np.mean(db)), color="C0", ls="--", lw=1.2, label=rf"both mean={np.mean(db):.5f}")
        ax.axvline(float(np.mean(dh)), color="C2", ls="--", lw=1.2, label=rf"hydro mean={np.mean(dh):.5f}")
        ax.set_xlim(xlim)
        ax.set_xlabel(xlab)
        ax.set_ylabel("sample count")
        ax.set_title(f"unbiased disp {tag}  (ref @ t≈2s)")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Unbiased body translation (green trajectory) during t∈[2,3]s\n"
        r"$\Delta \mathbf{t}(t) = \mathbf{t}(t) - \mathbf{t}(t=2\,\mathrm{s})$",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    stats_path = out_path.with_suffix(".txt")
    with stats_path.open("w") as f:
        f.write("# unbiased displacement stats t in [2,3]s, ref=first sample\n")
        for name, dx, dy, dz, mag in [
            ("both", dx_b, dy_b, dz_b, mag_b),
            ("hydro", dx_h, dy_h, dz_h, mag_h),
        ]:
            f.write(f"\n## {name}\n")
            f.write(f"  mean dx={dx.mean():.6f} dy={dy.mean():.6f} dz={dz.mean():.6f} |dt|={mag.mean():.6f}\n")
            f.write(f"  max  |dt|={mag.max():.6f}\n")
    print(f"saved {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("build/fsi_disp_compare"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--xmin", type=float, default=-0.05)
    parser.add_argument("--xmax", type=float, default=0.05)
    args = parser.parse_args()

    both_path = args.dir / "both.csv"
    hydro_path = args.dir / "hydro.csv"
    if not both_path.exists() or not hydro_path.exists():
        raise SystemExit(f"missing CSV in {args.dir} (run fsi_unbiased_disp --out-dir {args.dir} first)")

    both = load_csv(both_path)
    hydro = load_csv(hydro_path)
    out = args.out or (args.dir / "unbiased_disp_hist.png")
    plot_compare(out, both, hydro, (args.xmin, args.xmax))


if __name__ == "__main__":
    main()
