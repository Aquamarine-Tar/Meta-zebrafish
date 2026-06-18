#!/usr/bin/env python3
import csv
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("result/flow_sweep_480hz_1p5s_projection")
OUT = ROOT / "analysis_figures"


def parse_flow_tag(tag):
    if tag == "flow_0":
        return 0.0
    m = re.match(r"flow_m0p(\d+)$", tag)
    if not m:
        return None
    digits = m.group(1)
    if len(digits) == 1:
        return -float("0." + digits)
    return -float("0." + digits)


def parse_metrics(path):
    out = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    def f(name, default=np.nan):
        try:
            return float(out.get(name, default).split()[0])
        except Exception:
            return default
    def vec(name):
        text = out.get(name, "")
        vals = [float(x) for x in text.split()] if text else [np.nan, np.nan, np.nan]
        return vals[:3] if len(vals) >= 3 else [np.nan, np.nan, np.nan]
    flow_vals = vec("flow")
    ub_vals = vec("ub_disp")
    com_vals = vec("com_disp")
    return {
        "avg_step_ms": f("avg_step_ms"),
        "max_step_ms": f("max_step_ms"),
        "final_vol": f("final_vol"),
        "vol_shrink_pct": f("vol_shrink_pct"),
        "final_inverted": f("final_inverted"),
        "com_x": com_vals[0],
        "com_y": com_vals[1],
        "com_z": f("com_disp_z", com_vals[2]),
        "ub_x": ub_vals[0],
        "ub_y": ub_vals[1],
        "ub_z": f("ub_disp_z", ub_vals[2]),
        "hydro_nonzero_surf": f("hydro_nonzero_surf"),
        "hydro_surface_total": f("hydro_surface_total"),
        "hydro_nonzero_pct": f("hydro_nonzero_pct"),
        "flow_z": flow_vals[2] if len(flow_vals) >= 3 and not math.isnan(flow_vals[2]) else np.nan,
        "hydro_transverse_projection": f("hydro_transverse_projection"),
    }


def load_rows():
    rows = []
    for flow_dir in sorted(ROOT.glob("flow_*")):
        if not flow_dir.is_dir():
            continue
        flow_tag = flow_dir.name
        for run_dir in sorted(flow_dir.glob("run_*")):
            mpath = run_dir / "metrics.txt"
            if not mpath.exists():
                continue
            metrics = parse_metrics(mpath)
            if math.isnan(metrics["flow_z"]):
                fallback = parse_flow_tag(flow_tag)
                metrics["flow_z"] = fallback if fallback is not None else np.nan
            metrics["flow_tag"] = flow_tag
            metrics["run"] = int(run_dir.name.split("_")[-1])
            metrics["run_dir"] = run_dir
            rows.append(metrics)
    return rows


def group_by_flow(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[round(r["flow_z"], 6)].append(r)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def mean_std(vals):
    a = np.array(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def style_axes(ax):
    ax.grid(True, color="#e6e8ef", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#b9beca")
    ax.spines["bottom"].set_color("#b9beca")


def savefig(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_summary(groups):
    flows = np.array(list(groups.keys()), dtype=float)
    labels = [f"{x:g}" for x in flows]

    def stat(metric):
        m, s = [], []
        for f in flows:
            mean, std = mean_std([r[metric] for r in groups[f]])
            m.append(mean); s.append(std)
        return np.array(m), np.array(s)

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))
    fig.suptitle("480 Hz / 1.5 s Flow Sweep: Stability and Motion", fontsize=16, fontweight="bold")

    ub_m, ub_s = stat("ub_z")
    axs[0, 0].errorbar(flows, ub_m, yerr=ub_s, marker="o", linewidth=2.2, capsize=4, color="#2454C6")
    axs[0, 0].scatter([r["flow_z"] for g in groups.values() for r in g], [r["ub_z"] for g in groups.values() for r in g],
                      color="#5B8DEF", edgecolor="white", s=46, zorder=3)
    axs[0, 0].axhline(0, color="#444", linewidth=1)
    axs[0, 0].set_xlabel("background flow z velocity")
    axs[0, 0].set_ylabel("registered z displacement")
    axs[0, 0].set_title("Registered +z displacement remains positive")
    style_axes(axs[0, 0])

    vol_m, vol_s = stat("final_vol")
    inv_m, inv_s = stat("final_inverted")
    ax = axs[0, 1]
    ax.errorbar(flows, vol_m, yerr=vol_s, marker="s", linewidth=2.2, capsize=4, color="#159A72", label="volume ratio")
    ax.axhline(0.90, color="#C62828", linestyle="--", linewidth=1.2, label="10% shrink limit")
    ax.set_xlabel("background flow z velocity")
    ax.set_ylabel("final volume ratio")
    ax.set_ylim(0.88, max(1.0, np.nanmax(vol_m + vol_s) + 0.015))
    ax2 = ax.twinx()
    ax2.plot(flows, inv_m, marker="^", linewidth=2, color="#D68000", label="inverted tets")
    ax2.set_ylabel("mean inverted tets")
    ax.set_title("Stable volume; flipped elements remain low")
    style_axes(ax)
    lines, names = ax.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, names + names2, loc="lower left", frameon=False)

    ms_m, ms_s = stat("avg_step_ms")
    axs[1, 0].bar(labels, ms_m, yerr=ms_s, capsize=4, color="#6E5AA8", alpha=0.88)
    axs[1, 0].axhline(750, color="#C62828", linestyle="--", linewidth=1.2, label="750 ms target")
    axs[1, 0].set_xlabel("background flow z velocity")
    axs[1, 0].set_ylabel("average step time (ms)")
    axs[1, 0].set_title("Current runs exceed the 750 ms runtime target")
    axs[1, 0].legend(frameon=False)
    style_axes(axs[1, 0])

    cov_m, cov_s = stat("hydro_nonzero_pct")
    axs[1, 1].plot(flows, cov_m, marker="D", linewidth=2.2, color="#0097A7")
    axs[1, 1].fill_between(flows, cov_m - cov_s, cov_m + cov_s, color="#0097A7", alpha=0.12)
    axs[1, 1].set_xlabel("background flow z velocity")
    axs[1, 1].set_ylabel("nonzero hydro-force surface coverage (%)")
    axs[1, 1].set_ylim(98.5, 100.5)
    axs[1, 1].set_title("All snapshots maintain full vertex force coverage")
    style_axes(axs[1, 1])

    fig.tight_layout()
    return savefig(fig, "01_summary_dashboard.png")


def plot_displacement_components(groups):
    flows = np.array(list(groups.keys()), dtype=float)
    metrics = [("ub_x", "x", "#CC4E5C"), ("ub_y", "y", "#339966"), ("ub_z", "z", "#2454C6")]
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for metric, label, color in metrics:
        means, stds = [], []
        for f in flows:
            m, s = mean_std([r[metric] for r in groups[f]])
            means.append(m); stds.append(s)
        ax.errorbar(flows, means, yerr=stds, marker="o", linewidth=2.2, capsize=4, label=f"registered {label}", color=color)
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_xlabel("background flow z velocity")
    ax.set_ylabel("registered displacement component")
    ax.set_title("Registered displacement vector components")
    ax.legend(frameon=False, ncol=3)
    style_axes(ax)
    return savefig(fig, "02_registered_displacement_components.png")


def read_torque(run_dir):
    path = run_dir / "torque_summary.csv"
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_force_time_heatmaps(groups):
    times = [0.5, 1.0, 1.5]
    flows = np.array(list(groups.keys()), dtype=float)
    mean_fz = np.full((len(flows), len(times)), np.nan)
    std_fz = np.full_like(mean_fz, np.nan)
    for i, f in enumerate(flows):
        vals_by_t = defaultdict(list)
        for r in groups[f]:
            for tr in read_torque(r["run_dir"]):
                vals_by_t[round(tr["sim_time"], 2)].append(tr["force_sum_z"])
        for j, t in enumerate(times):
            mean_fz[i, j], std_fz[i, j] = mean_std(vals_by_t[t])

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    vmax = np.nanmax(np.abs(mean_fz))
    im = ax.imshow(mean_fz, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(times)), [f"{t:.1f}s" for t in times])
    ax.set_yticks(range(len(flows)), [f"{f:g}" for f in flows])
    ax.set_xlabel("snapshot time")
    ax.set_ylabel("background flow z velocity")
    ax.set_title("Net z hydro-force from torque summaries")
    for i in range(len(flows)):
        for j in range(len(times)):
            if np.isfinite(mean_fz[i, j]):
                ax.text(j, i, f"{mean_fz[i, j]:.1f}", ha="center", va="center", fontsize=9, color="#111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean force_sum_z")
    return savefig(fig, "03_net_z_force_heatmap.png")


def read_csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def load_positions_forces(run_dir, tag):
    pos_path = run_dir / f"{tag}_positions.csv"
    force_path = run_dir / f"{tag}_forces.csv"
    if not pos_path.exists() or not force_path.exists():
        return None
    pos = read_csv_rows(pos_path)
    forces = read_csv_rows(force_path)
    n = min(len(pos), len(forces))
    arr = np.zeros((n, 7), dtype=float)
    for i in range(n):
        arr[i, 0] = float(pos[i]["x"])
        arr[i, 1] = float(pos[i]["y"])
        arr[i, 2] = float(pos[i]["z"])
        arr[i, 3] = float(forces[i]["hx"])
        arr[i, 4] = float(forces[i]["hy"])
        arr[i, 5] = float(forces[i]["hz"])
        arr[i, 6] = math.sqrt(arr[i, 3] ** 2 + arr[i, 4] ** 2 + arr[i, 5] ** 2)
    return arr


def plot_force_clouds():
    cases = [
        ("flow_0", "run_1", "t1.50", "still water"),
        ("flow_m0p05", "run_1", "t1.50", "flow z=-0.05"),
        ("flow_m0p2", "run_1", "t1.50", "flow z=-0.20"),
    ]
    arrays = []
    for flow, run, tag, title in cases:
        arr = load_positions_forces(ROOT / flow / run, tag)
        if arr is not None:
            arrays.append((arr, title))
    if not arrays:
        return None
    vmax = np.percentile(np.abs(np.concatenate([a[:, 5] for a, _ in arrays])), 98)
    mag_max = np.percentile(np.concatenate([a[:, 6] for a, _ in arrays]), 98)
    fig, axs = plt.subplots(2, len(arrays), figsize=(5.0 * len(arrays), 8.3), sharex=False, sharey=False)
    if len(arrays) == 1:
        axs = np.array([[axs[0]], [axs[1]]])
    for j, (arr, title) in enumerate(arrays):
        sc = axs[0, j].scatter(arr[:, 2], arr[:, 1], c=arr[:, 5], s=5, cmap="RdBu_r", vmin=-vmax, vmax=vmax, linewidths=0)
        axs[0, j].set_title(f"{title}: signed z force")
        axs[0, j].set_xlabel("z")
        axs[0, j].set_ylabel("y")
        axs[0, j].set_aspect("equal", adjustable="box")
        style_axes(axs[0, j])
        sc2 = axs[1, j].scatter(arr[:, 2], arr[:, 1], c=arr[:, 6], s=5, cmap="magma", vmin=0, vmax=mag_max, linewidths=0)
        axs[1, j].set_title(f"{title}: force magnitude")
        axs[1, j].set_xlabel("z")
        axs[1, j].set_ylabel("y")
        axs[1, j].set_aspect("equal", adjustable="box")
        style_axes(axs[1, j])
    cb = fig.colorbar(sc, ax=axs[0, :].ravel().tolist(), fraction=0.025, pad=0.02)
    cb.set_label("hydro force z component")
    cb2 = fig.colorbar(sc2, ax=axs[1, :].ravel().tolist(), fraction=0.025, pad=0.02)
    cb2.set_label("hydro force magnitude")
    fig.suptitle("Surface hydro-force distributions at t=1.50 s", fontsize=15, fontweight="bold")
    return savefig(fig, "04_surface_force_clouds_t1p50.png")


def plot_side_balance(groups):
    records = []
    for f, runs in groups.items():
        for r in runs:
            for tag in ["t0.50", "t1.00", "t1.50"]:
                arr = load_positions_forces(r["run_dir"], tag)
                if arr is None:
                    continue
                y_mid = np.median(arr[:, 1])
                left = arr[arr[:, 1] >= y_mid]
                right = arr[arr[:, 1] < y_mid]
                records.append({
                    "flow": f,
                    "time": float(tag[1:]),
                    "side_delta_hz": float(np.sum(left[:, 5]) - np.sum(right[:, 5])),
                    "side_abs_delta": float(np.sum(np.abs(left[:, 5])) - np.sum(np.abs(right[:, 5]))),
                    "net_hz": float(np.sum(arr[:, 5])),
                })
    if not records:
        return None
    flows = np.array(list(groups.keys()), dtype=float)
    times = [0.5, 1.0, 1.5]
    grid = np.full((len(flows), len(times)), np.nan)
    for i, f in enumerate(flows):
        for j, t in enumerate(times):
            vals = [r["side_delta_hz"] for r in records if abs(r["flow"] - f) < 1e-6 and abs(r["time"] - t) < 1e-6]
            grid[i, j], _ = mean_std(vals)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, aspect="auto", cmap="PiYG", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(times)), [f"{t:.1f}s" for t in times])
    ax.set_yticks(range(len(flows)), [f"{f:g}" for f in flows])
    ax.set_xlabel("snapshot time")
    ax.set_ylabel("background flow z velocity")
    ax.set_title("Left/right split of z-force over fish surface")
    for i in range(len(flows)):
        for j in range(len(times)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("sum(hz, y>=median) - sum(hz, y<median)")
    return savefig(fig, "05_left_right_z_force_balance.png")


def make_pressure_montage():
    entries = [
        ("flow_0/run_1/t1.50_pressure_zy.png", "flow 0, t=1.50, z-y"),
        ("flow_m0p05/run_1/t1.50_pressure_zy.png", "flow -0.05, t=1.50, z-y"),
        ("flow_m0p08/run_1/t1.50_pressure_zy.png", "flow -0.08, t=1.50, z-y"),
        ("flow_m0p2/run_1/t1.50_pressure_zy.png", "flow -0.20, t=1.50, z-y"),
        ("flow_0/run_1/t1.50_pressure_zx.png", "flow 0, t=1.50, z-x"),
        ("flow_m0p05/run_1/t1.50_pressure_zx.png", "flow -0.05, t=1.50, z-x"),
        ("flow_m0p08/run_1/t1.50_pressure_zx.png", "flow -0.08, t=1.50, z-x"),
        ("flow_m0p2/run_1/t1.50_pressure_zx.png", "flow -0.20, t=1.50, z-x"),
    ]
    imgs = []
    labels = []
    for rel, label in entries:
        p = ROOT / rel
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        imgs.append(img)
        labels.append(label)
    if not imgs:
        return None
    thumb_w = 560
    pad = 24
    label_h = 38
    scaled = []
    for img in imgs:
        scale = thumb_w / img.width
        scaled.append(img.resize((thumb_w, int(img.height * scale)), Image.LANCZOS))
    row_h = max(im.height for im in scaled[:4]) + label_h
    rows = int(math.ceil(len(scaled) / 4))
    canvas = Image.new("RGB", (4 * thumb_w + 5 * pad, rows * row_h + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    for idx, img in enumerate(scaled):
        row = idx // 4
        col = idx % 4
        x = pad + col * (thumb_w + pad)
        y = pad + row * (row_h + pad)
        draw.text((x, y), labels[idx], fill=(30, 35, 45), font=font)
        canvas.paste(img, (x, y + label_h))
    path = OUT / "06_pressure_existing_png_montage.png"
    canvas.save(path)
    return path


def write_augmented_csv(rows):
    path = OUT / "augmented_metrics.csv"
    fields = ["flow_z", "flow_tag", "run", "avg_step_ms", "max_step_ms", "final_vol",
              "vol_shrink_pct", "final_inverted", "com_x", "com_y", "com_z",
              "ub_x", "ub_y", "ub_z", "hydro_nonzero_surf", "hydro_surface_total",
              "hydro_nonzero_pct", "hydro_transverse_projection"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path


def write_report(rows, groups, figures):
    report = OUT / "analysis_report.md"
    lines = []
    lines.append("# Flow Sweep Analysis\n")
    lines.append("Data source: `result/flow_sweep_480hz_1p5s_projection`.\n")
    lines.append("This analysis includes all completed runs found under the directory, including the later `flow_m0p2` runs that are not present in the original `summary.csv`.\n")
    lines.append("## Figures\n")
    for p in figures:
        if p is None:
            continue
        lines.append(f"- [{p.name}]({p.name})")
    lines.append("\n## Grouped Metrics\n")
    lines.append("| flow_z | runs | avg step ms | max step ms | volume | shrink % | inverted | ub_z mean ± sd | ub_z range | coverage |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for f, rs in groups.items():
        avg_ms, avg_ms_s = mean_std([r["avg_step_ms"] for r in rs])
        max_ms, _ = mean_std([r["max_step_ms"] for r in rs])
        vol, vol_s = mean_std([r["final_vol"] for r in rs])
        shrink, _ = mean_std([r["vol_shrink_pct"] for r in rs])
        inv, _ = mean_std([r["final_inverted"] for r in rs])
        ubz_vals = [r["ub_z"] for r in rs]
        ubz, ubz_s = mean_std(ubz_vals)
        cov, _ = mean_std([r["hydro_nonzero_pct"] for r in rs])
        lines.append(
            f"| {f:g} | {len(rs)} | {avg_ms:.1f} ± {avg_ms_s:.1f} | {max_ms:.1f} | "
            f"{vol:.4f} ± {vol_s:.4f} | {shrink:.2f} | {inv:.1f} | "
            f"{ubz:.4f} ± {ubz_s:.4f} | {min(ubz_vals):.4f}..{max(ubz_vals):.4f} | {cov:.1f}% |"
        )
    lines.append("\n## Conclusions\n")
    lines.append("- Stability is acceptable in these logs: all final volume ratios are above 0.95, so shrinkage stays below 5%, and inverted tet counts remain low.")
    lines.append("- Vertex force coverage is excellent: all completed runs report 100% nonzero hydro-force coverage at the final snapshot, and the snapshot coverage table reports 100% over the original 36 snapshots.")
    lines.append("- Runtime misses the earlier 750 ms target by a wide margin: most 480 Hz runs are around 1.56-1.61 s per step, and `flow_z=-0.2` run 1 is much slower.")
    lines.append("- The main physical concern is reproducibility of flow response. Increasing negative background flow from 0 to -0.08 does not monotonically reduce registered +z displacement, and the later -0.2 runs are inconsistent (`ub_z=0.4160` and `0.1735`).")
    lines.append("- Therefore this result set supports the diagnosis that the projection/low ghost-coupling parameter state preserved coverage and stability, but did not produce a reliable background-flow drag response.")
    lines.append("- The existing pressure images are visually useful, but the aggregate displacement/force statistics do not yet demonstrate the expected left-right pressure difference translating into a reproducible z-direction swimming response.")
    report.write_text("\n".join(lines) + "\n")
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if not rows:
        raise SystemExit(f"no metrics found under {ROOT}")
    groups = group_by_flow(rows)
    figures = [
        plot_summary(groups),
        plot_displacement_components(groups),
        plot_force_time_heatmaps(groups),
        plot_force_clouds(),
        plot_side_balance(groups),
        make_pressure_montage(),
    ]
    csv_path = write_augmented_csv(rows)
    report = write_report(rows, groups, figures)
    print(f"loaded_runs={len(rows)}")
    print(f"flows={','.join(f'{f:g}' for f in groups)}")
    print(f"out={OUT}")
    print(f"metrics={csv_path}")
    print(f"report={report}")
    for p in figures:
        if p is not None:
            print(f"figure={p}")


if __name__ == "__main__":
    main()
