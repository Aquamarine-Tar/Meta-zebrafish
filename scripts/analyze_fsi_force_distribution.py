#!/usr/bin/env python3
"""
分析单个时刻 FSI 顶点力分布，识别尖峰及其对 tau_x 的贡献。
用法:
  python3 scripts/analyze_fsi_force_distribution.py --dir build/fsi_snapshots --time 3.0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_csv_dict(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k in cols:
            cols[k].append(float(row[k]) if k not in ("vtx", "f") else int(row[k]))
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


def surface_indices(faces: np.ndarray) -> np.ndarray:
    return np.unique(faces.ravel())


def load_snapshot(snap_dir: Path, sim_t: float):
    tag = f"t{sim_t:.2f}"
    pos = read_csv_dict(snap_dir / f"{tag}_positions.csv")
    forces = read_csv_dict(snap_dir / f"{tag}_forces.csv")
    meta = read_meta(snap_dir / f"{tag}_meta.txt")
    vertices = np.stack([pos["x"], pos["y"], pos["z"]], axis=1)
    contact = np.stack([forces["cx"], forces["cy"], forces["cz"]], axis=1)
    hydro = np.stack([forces["hx"], forces["hy"], forces["hz"]], axis=1)
    total = np.stack([forces["fx"], forces["fy"], forces["fz"]], axis=1)
    return vertices, contact, hydro, total, meta


def parse_vec3(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.replace(",", " ").split()])


def compute_tau_contributions(
    positions: np.ndarray,
    forces: np.ndarray,
    surf: np.ndarray,
    pivot: np.ndarray,
) -> np.ndarray:
    """返回各表面顶点 tau_x 分量（关于 pivot 的叉乘 x 分量）。"""
    p = positions[surf]
    f = forces[surf]
    r = p - pivot
    return np.cross(r, f)[:, 0]


def percentile_stats(x: np.ndarray) -> dict:
    ps = [50, 90, 95, 99, 99.9, 100]
    out = {"mean": float(np.mean(x)), "std": float(np.std(x)), "max": float(np.max(x))}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(x, p))
    out["n"] = len(x)
    out["n_gt_1N"] = int(np.sum(x > 1.0))
    out["n_gt_2N"] = int(np.sum(x > 2.0))
    out["n_at_clip_3N"] = int(np.sum(x >= 2.999))
    return out


def write_stats_report(path: Path, sim_t: float, stats: dict, top_spike_rows: list, top_tau_rows: list):
    with path.open("w") as f:
        f.write(f"# FSI force distribution at t={sim_t:.2f}s (surface vertices only)\n\n")
        for name, s in stats.items():
            f.write(f"## |F| {name}\n")
            for k, v in s.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
        f.write("## Top-15 |F_total| spikes (vtx, |F|, Fy, tau_x contribution)\n")
        for row in top_spike_rows:
            f.write(f"  {row}\n")
        f.write("\n## Top-15 |tau_x| contributors\n")
        for row in top_tau_rows:
            f.write(f"  {row}\n")


def plot_distribution_figure(
    out_path: Path,
    sim_t: float,
    mag_total: np.ndarray,
    mag_contact: np.ndarray,
    mag_hydro: np.ndarray,
    tau_x: np.ndarray,
    ref_t: float | None,
    ref_mag: np.ndarray | None,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    bins = np.linspace(0, max(mag_total.max(), 3.05), 50)
    for ax, data, label, color in [
        (axes[0, 0], mag_total, "|F_total|", "C0"),
        (axes[0, 1], mag_contact, "|F_contact|", "C1"),
        (axes[1, 0], mag_hydro, "|F_hydro|", "C2"),
    ]:
        ax.hist(data, bins=bins, color=color, alpha=0.75, edgecolor="none")
        if ref_mag is not None and label == "|F_total|":
            ax.hist(ref_mag, bins=bins, color="gray", alpha=0.35, edgecolor="none", label=f"t={ref_t:.2f}s")
            ax.legend(fontsize=8)
        ax.axvline(3.0, color="r", ls="--", lw=1, label="clip 3N")
        ax.set_xlabel(f"{label} [N]")
        ax.set_ylabel("vertex count")
        ax.set_title(f"{label} histogram (n={len(data)})")
        if label != "|F_total|" or ref_mag is None:
            ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(tau_x, bins=60, color="C3", alpha=0.75)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"per-vertex $\tau_x$ contribution [N·m]")
    ax.set_ylabel("vertex count")
    ax.set_title(r"$\tau_x$ contribution per surface vertex")

    fig.suptitle(f"FSI force distribution at t={sim_t:.2f}s (surface)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_spatial_spikes(
    out_path: Path,
    sim_t: float,
    vertices: np.ndarray,
    faces: np.ndarray,
    surf: np.ndarray,
    mag: np.ndarray,
    tau_x: np.ndarray,
    spike_mask: np.ndarray,
    pivot: np.ndarray,
):
    """侧视 Y-Z + 展开图，标出尖峰顶点。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    p_surf = vertices[surf]
    z, y = p_surf[:, 2], p_surf[:, 1]
    x_c, y_c = p_surf[:, 0].mean(), p_surf[:, 1].mean()
    theta = np.arctan2(p_surf[:, 1] - y_c, p_surf[:, 0] - x_c)

    vmax = np.percentile(mag, 99.5)

    # 侧视 |F|
    ax = axes[0]
    sc = ax.scatter(z, y, c=mag, s=4, cmap="inferno", norm=Normalize(0, vmax), rasterized=True)
    ax.scatter(z[spike_mask], y[spike_mask], s=40, facecolors="none", edgecolors="cyan", linewidths=0.8, label="spike (top 1%)")
    ax.set_xlabel("Z [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("side view |F| + spike outline")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="|F| [N]")

    # 侧视 tau_x 贡献
    ax = axes[1]
    tmax = np.percentile(np.abs(tau_x), 99)
    norm = Normalize(vmin=-tmax, vmax=tmax)
    ax.scatter(z, y, c=tau_x, s=4, cmap="RdBu_r", norm=norm, rasterized=True)
    ax.scatter(z[spike_mask], y[spike_mask], s=40, facecolors="none", edgecolors="lime", linewidths=0.8)
    ax.set_xlabel("Z [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(r"side view $\tau_x$ contribution")
    ax.set_aspect("equal")

    # 展开图 + 尖峰
    ax = axes[2]
    ax.scatter(theta, z, c=mag, s=4, cmap="inferno", norm=Normalize(0, vmax), rasterized=True)
    ax.scatter(theta[spike_mask], z[spike_mask], s=50, facecolors="none", edgecolors="cyan", linewidths=0.9)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel("Z [m]")
    ax.set_title("unwrap Z×θ (spikes circled)")

    fig.suptitle(
        f"t={sim_t:.2f}s spatial spikes (cyan = top 1% |F|, pivot COM)\n"
        f"sum tau_x={tau_x.sum():.3f} N·m",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_fy_histograms(
    out_path: Path,
    sim_t: float,
    fy_total: np.ndarray,
    fy_contact: np.ndarray,
    fy_hydro: np.ndarray,
):
    """图 1：signed F_y 直方图（total / contact / hydro）。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    datasets = [
        (fy_total, r"$F_{y,\mathrm{total}}$", "C0"),
        (fy_contact, r"$F_{y,\mathrm{contact}}$", "C1"),
        (fy_hydro, r"$F_{y,\mathrm{hydro}}$", "C2"),
    ]
    all_fy = np.concatenate([fy_total, fy_contact, fy_hydro])
    lo, hi = np.percentile(all_fy, [0.5, 99.5])
    pad = max(0.01, (hi - lo) * 0.15)
    bins = np.linspace(lo - pad, hi + pad, 55)

    for ax, (data, label, color) in zip(axes, datasets):
        ax.hist(data, bins=bins, color=color, alpha=0.78, edgecolor="none")
        mean_v = float(np.mean(data))
        sum_v = float(np.sum(data))
        ax.axvline(0.0, color="k", lw=0.8)
        ax.axvline(mean_v, color="red", ls="--", lw=1.2, label=rf"$\bar{{F}}_y$={mean_v:.5f} N")
        ax.set_xlabel(f"{label} [N]")
        ax.set_ylabel("vertex count")
        ax.set_title(f"{label}  n={len(data)}  Σ={sum_v:.3f} N")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"t={sim_t:.2f}s signed $F_y$ histograms (surface)\n"
        rf"$\bar{{F}}_{{y,\mathrm{{total}}}}$={np.mean(fy_total):.5f} N, "
        rf"$\Sigma F_{{y,\mathrm{{total}}}}$={fy_total.sum():.3f} N",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_fy_contact_hydro_hist(
    out_path: Path,
    sim_t: float,
    fy_contact: np.ndarray,
    fy_hydro: np.ndarray,
    xlim: tuple[float, float] = (-1.0, 1.0),
    n_bins: int = 40,
):
    """signed F_y 直方图：接触斥力 / 水压力，固定 x 轴范围（默认 ±1 N）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(xlim[0], xlim[1], n_bins)

    for ax, data, label, color in [
        (axes[0], fy_contact, r"$F_{y,\mathrm{contact}}$", "C1"),
        (axes[1], fy_hydro, r"$F_{y,\mathrm{hydro}}$", "C2"),
    ]:
        mean_v = float(np.mean(data))
        sum_v = float(np.sum(data))
        ax.hist(data, bins=bins, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(0.0, color="k", lw=0.5)
        ax.axvline(mean_v, color="red", ls="--", lw=1.2, label=rf"$\bar{{F}}_y$={mean_v:.5f} N")
        ax.set_xlim(xlim)
        ax.set_xlabel(f"{label} [N]")
        ax.set_ylabel("vertex count")
        ax.set_title(f"{label} histogram (n={len(data)})  Σ={sum_v:.3f} N")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"FSI signed $F_y$ at t={sim_t:.2f}s (surface)\n"
        rf"contact $\bar{{F}}_y$={fy_contact.mean():.5f} N, "
        rf"hydro $\bar{{F}}_y$={fy_hydro.mean():.5f} N",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_upper_lower_split(
    out_path: Path,
    sim_t: float,
    positions: np.ndarray,
    surf: np.ndarray,
    fy_total: np.ndarray,
    tau_x: np.ndarray,
    y_com: float,
):
    """图 2：相对 COM 的上下半身 ΣF_y 与 τ_x 分解。"""
    y = positions[surf, 1]
    upper = y > y_com
    lower = y < y_com
    on_eq = ~(upper | lower)

    groups = {
        "upper (y > y_COM)": upper,
        "lower (y < y_COM)": lower,
        "on y_COM": on_eq,
    }
    stats = {}
    for name, mask in groups.items():
        stats[name] = {
            "n": int(mask.sum()),
            "sum_Fy": float(fy_total[mask].sum()),
            "mean_Fy": float(fy_total[mask].mean()) if mask.any() else 0.0,
            "sum_tau_x": float(tau_x[mask].sum()),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = ["upper (y > y_COM)", "lower (y < y_COM)"]
    x = np.arange(len(names))
    sum_fy = [stats[n]["sum_Fy"] for n in names]
    sum_tau = [stats[n]["sum_tau_x"] for n in names]
    ns = [stats[n]["n"] for n in names]

    ax = axes[0]
    bars = ax.bar(x, sum_fy, color=["C0", "C3"], alpha=0.85)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nn={ns[i]}" for i, n in enumerate(names)], fontsize=9)
    ax.set_ylabel(r"$\Sigma F_y$ [N]")
    ax.set_title(r"$\Sigma F_y$ by upper / lower half")
    for b, v in zip(bars, sum_fy):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    ax = axes[1]
    bars = ax.bar(x, sum_tau, color=["C2", "C1"], alpha=0.85)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nn={ns[i]}" for i, n in enumerate(names)], fontsize=9)
    ax.set_ylabel(r"$\Sigma \tau_x$ contrib [N·m]")
    ax.set_title(r"$\tau_x$ contribution by upper / lower half")
    for b, v in zip(bars, sum_tau):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    fig.suptitle(
        f"t={sim_t:.2f}s  y_COM={y_com:.4f} m  "
        f"total ΣF_y={fy_total.sum():.3f} N  total τ_x={tau_x.sum():.3f} N·m",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return stats


def plot_z_segment_forces(
    out_path: Path,
    sim_t: float,
    positions: np.ndarray,
    surf: np.ndarray,
    fy_total: np.ndarray,
    fz_total: np.ndarray,
    tau_x: np.ndarray,
    n_bins: int = 20,
):
    """图 3：沿体轴 z 分段的 ΣF_y、ΣF_z、τ_x。"""
    z = positions[surf, 2]
    z_edges = np.linspace(z.min(), z.max(), n_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    bin_idx = np.clip(np.digitize(z, z_edges) - 1, 0, n_bins - 1)

    sum_fy = np.zeros(n_bins)
    sum_fz = np.zeros(n_bins)
    sum_tau = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = bin_idx == b
        counts[b] = int(m.sum())
        sum_fy[b] = fy_total[m].sum()
        sum_fz[b] = fz_total[m].sum()
        sum_tau[b] = tau_x[m].sum()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, data, ylab, title in [
        (axes[0], sum_fy, r"$\Sigma F_y$ [N]", r"$\Sigma F_y(z)$"),
        (axes[1], sum_fz, r"$\Sigma F_z$ [N]", r"$\Sigma F_z(z)$"),
        (axes[2], sum_tau, r"$\Sigma \tau_x$ [N·m]", r"$\Sigma \tau_x(z)$"),
    ]:
        ax.bar(z_centers, data, width=(z_edges[1] - z_edges[0]) * 0.9, alpha=0.85)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].set_xlabel("Z [m] (body axis)")
    fig.suptitle(f"t={sim_t:.2f}s  force & torque by Z segment (n={n_bins})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return z_centers, sum_fy, sum_fz, sum_tau, counts


def render_signed_mesh(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    vtx_vals: np.ndarray,
    title: str,
    vmax: float,
):
    """3D 三角面片，面心值为顶点值平均，发散色标。"""
    face_vals = vtx_vals[faces].mean(axis=1)
    polys = vertices[faces]
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    colors = cm.RdBu_r(norm(face_vals))
    mesh = Poly3DCollection(polys, facecolors=colors, edgecolors=(0, 0, 0, 0.04), linewidths=0.08)
    ax.add_collection3d(mesh)
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=22, azim=-58)
    return norm


def plot_surface_heatmaps_3d(
    out_path: Path,
    sim_t: float,
    vertices: np.ndarray,
    faces: np.ndarray,
    fy_total: np.ndarray,
    fy_contact: np.ndarray,
    fy_hydro: np.ndarray,
    tau_x: np.ndarray,
    surf: np.ndarray,
):
    """图 4：3D 表面热力图（signed F_y / τ_x / contact / hydro）。"""
    n_vtx = len(vertices)
    vtx_fy = np.zeros(n_vtx)
    vtx_fc = np.zeros(n_vtx)
    vtx_fh = np.zeros(n_vtx)
    vtx_tau = np.zeros(n_vtx)
    vtx_fy[surf] = fy_total
    vtx_fc[surf] = fy_contact
    vtx_fh[surf] = fy_hydro
    vtx_tau[surf] = tau_x

    panels = [
        (vtx_fy, r"$F_y$ (total)"),
        (vtx_tau, r"$\tau_x$ contribution"),
        (vtx_fc, r"$F_{y,\mathrm{contact}}$"),
        (vtx_fh, r"$F_{y,\mathrm{hydro}}$"),
    ]
    vmax = max(np.percentile(np.abs(v), 99) for v, _ in panels)
    vmax = max(vmax, 1e-6)

    fig = plt.figure(figsize=(14, 11))
    for i, (vals, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        norm = render_signed_mesh(ax, vertices, faces, vals, title, vmax)

    fig.subplots_adjust(right=0.88, wspace=0.05, hspace=0.12)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    sm = cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), cmap="RdBu_r")
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="signed value (+ red / − blue)")
    fig.suptitle(
        f"t={sim_t:.2f}s 3D surface heatmaps (RdBu: + / −)\n"
        r"head $\leftarrow$ Z $\rightarrow$ tail",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def write_fy_analysis_report(
    path: Path,
    sim_t: float,
    fy_total: np.ndarray,
    fy_contact: np.ndarray,
    fy_hydro: np.ndarray,
    upper_lower_stats: dict,
    z_centers: np.ndarray,
    sum_fy_z: np.ndarray,
    sum_fz_z: np.ndarray,
    sum_tau_z: np.ndarray,
    y_com: float,
):
    with path.open("w") as f:
        f.write(f"# F_y / torque decomposition at t={sim_t:.2f}s\n\n")
        f.write("## Signed F_y means (per vertex)\n")
        f.write(f"  Fy_total:   mean={fy_total.mean():.6f}  sum={fy_total.sum():.4f} N\n")
        f.write(f"  Fy_contact: mean={fy_contact.mean():.6f}  sum={fy_contact.sum():.4f} N\n")
        f.write(f"  Fy_hydro:   mean={fy_hydro.mean():.6f}  sum={fy_hydro.sum():.4f} N\n\n")
        f.write(f"## Upper / lower split (y_COM={y_com:.6f})\n")
        for name, s in upper_lower_stats.items():
            f.write(f"  {name}: n={s['n']}  sum_Fy={s['sum_Fy']:.4f}  mean_Fy={s['mean_Fy']:.6f}  sum_tau_x={s['sum_tau_x']:.4f}\n")
        f.write("\n## Z segments (dominant |tau_x| bin)\n")
        dom = int(np.argmax(np.abs(sum_tau_z)))
        f.write(
            f"  peak |tau_x| at z={z_centers[dom]:.4f} m: "
            f"sum_tau_x={sum_tau_z[dom]:.4f}, sum_Fy={sum_fy_z[dom]:.4f}, sum_Fz={sum_fz_z[dom]:.4f}\n"
        )
        f.write("\n## All Z bins: z_center, n, sum_Fy, sum_Fz, sum_tau_x\n")
        for zc, sf, sz, st in zip(z_centers, sum_fy_z, sum_fz_z, sum_tau_z):
            f.write(f"  {zc:.5f}  {sf:+.4f}  {sz:+.4f}  {st:+.4f}\n")


def plot_top_vertices_bar(out_path: Path, sim_t: float, labels: list, mags: list, tau_vals: list):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, mags, 0.4, label="|F| [N]", color="C0")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, tau_vals, 0.4, label=r"$|\tau_x|$ contrib [N·m]", color="C3", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("|F| [N]")
    ax2.set_ylabel(r"$|\tau_x|$ [N·m]")
    ax1.set_title(f"Top spike vertices at t={sim_t:.2f}s")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("build/fsi_snapshots"))
    parser.add_argument("--time", type=float, default=3.0)
    parser.add_argument("--ref-time", type=float, default=2.75, help="reference time for comparison")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    snap_dir = args.dir
    out_dir = args.out or (snap_dir / f"analysis_t{args.time:.2f}")
    out_dir.mkdir(parents=True, exist_ok=True)

    faces = load_faces(snap_dir)
    surf = surface_indices(faces)

    vertices, contact, hydro, total, meta = load_snapshot(snap_dir, args.time)
    pivot = parse_vec3(meta.get("com", "0 0 0"))

    mag_c = np.linalg.norm(contact[surf], axis=1)
    mag_h = np.linalg.norm(hydro[surf], axis=1)
    mag_t = np.linalg.norm(total[surf], axis=1)
    fy_total = total[surf, 1]
    fy_contact = contact[surf, 1]
    fy_hydro = hydro[surf, 1]
    fz_total = total[surf, 2]
    tau_x = compute_tau_contributions(vertices, total, surf, pivot)
    y_com = float(pivot[1])

    stats = {
        "total": percentile_stats(mag_t),
        "contact": percentile_stats(mag_c),
        "hydro": percentile_stats(mag_h),
        "tau_x_contrib": percentile_stats(np.abs(tau_x)),
    }

    ref_mag = None
    if (snap_dir / f"t{args.ref_time:.2f}_forces.csv").exists():
        v2, _, _, t2, _ = load_snapshot(snap_dir, args.ref_time)
        ref_mag = np.linalg.norm(t2[surf], axis=1)

    # 尖峰：top 1% |F| 或 >= 2.5N
    p99 = np.percentile(mag_t, 99)
    spike_thr = max(p99, 2.5)
    spike_mask = mag_t >= spike_thr

    # Top lists
    order_f = np.argsort(-mag_t)[:15]
    top_spike_rows = []
    for k in order_f:
        i = surf[k]
        top_spike_rows.append(
            f"vtx={i} |F|={mag_t[k]:.4f} Fy={total[i,1]:.4f} tau_x={tau_x[k]:.4f} z={vertices[i,2]:.4f} y={vertices[i,1]:.4f}"
        )

    order_t = np.argsort(-np.abs(tau_x))[:15]
    top_tau_rows = []
    for k in order_t:
        i = surf[k]
        top_tau_rows.append(
            f"vtx={i} tau_x={tau_x[k]:.4f} |F|={mag_t[k]:.4f} z={vertices[i,2]:.4f} y={vertices[i,1]:.4f}"
        )

    write_stats_report(out_dir / "force_stats.txt", args.time, stats, top_spike_rows, top_tau_rows)

    plot_distribution_figure(
        out_dir / "force_histograms.png",
        args.time,
        mag_t,
        mag_c,
        mag_h,
        tau_x,
        args.ref_time,
        ref_mag,
    )
    plot_spatial_spikes(
        out_dir / "spatial_spikes.png",
        args.time,
        vertices,
        faces,
        surf,
        mag_t,
        tau_x,
        spike_mask,
        pivot,
    )

    labels = [f"v{i}" for i in surf[order_f[:10]]]
    plot_top_vertices_bar(
        out_dir / "top_spike_vertices.png",
        args.time,
        labels,
        mag_t[order_f[:10]].tolist(),
        np.abs(tau_x[order_f[:10]]).tolist(),
    )

    # --- 图 1–4：F_y 分解与空间诊断 ---
    plot_fy_histograms(
        out_dir / "fig1_fy_histograms.png",
        args.time,
        fy_total,
        fy_contact,
        fy_hydro,
    )
    plot_fy_contact_hydro_hist(
        out_dir / "fy_contact_hydro_hist.png",
        args.time,
        fy_contact,
        fy_hydro,
        xlim=(-1.0, 1.0),
        n_bins=40,
    )
    upper_lower_stats = plot_upper_lower_split(
        out_dir / "fig2_upper_lower_split.png",
        args.time,
        vertices,
        surf,
        fy_total,
        tau_x,
        y_com,
    )
    z_centers, sum_fy_z, sum_fz_z, sum_tau_z, _ = plot_z_segment_forces(
        out_dir / "fig3_z_segment_forces.png",
        args.time,
        vertices,
        surf,
        fy_total,
        fz_total,
        tau_x,
        n_bins=20,
    )
    plot_surface_heatmaps_3d(
        out_dir / "fig4_surface_heatmaps_3d.png",
        args.time,
        vertices,
        faces,
        fy_total,
        fy_contact,
        fy_hydro,
        tau_x,
        surf,
    )
    write_fy_analysis_report(
        out_dir / "fy_analysis_stats.txt",
        args.time,
        fy_total,
        fy_contact,
        fy_hydro,
        upper_lower_stats,
        z_centers,
        sum_fy_z,
        sum_fz_z,
        sum_tau_z,
        y_com,
    )

    # 打印摘要
    print(f"\n=== t={args.time:.2f}s surface force stats ===")
    for kind, s in stats.items():
        print(f"  {kind}: max={s['max']:.4f} p99={s['p99']:.4f} mean={s['mean']:.4f} "
              f"n@3N={s['n_at_clip_3N']} n>2N={s['n_gt_2N']}")
    print(f"  sum tau_x (surface, COM pivot) = {tau_x.sum():.4f} N·m  (meta: {meta.get('tau_x_surface_about_com')})")
    print(f"  spike threshold = {spike_thr:.4f} N, n_spikes = {spike_mask.sum()}")
    print(f"  Fy_total: mean={fy_total.mean():.6f} N  sum={fy_total.sum():.3f} N")
    print(f"  upper ΣFy={upper_lower_stats['upper (y > y_COM)']['sum_Fy']:.3f} N  "
          f"lower ΣFy={upper_lower_stats['lower (y < y_COM)']['sum_Fy']:.3f} N")
    print(f"  upper Στ_x={upper_lower_stats['upper (y > y_COM)']['sum_tau_x']:.3f} N·m  "
          f"lower Στ_x={upper_lower_stats['lower (y < y_COM)']['sum_tau_x']:.3f} N·m")
    print(f"  reports -> {out_dir}/")


if __name__ == "__main__":
    main()
