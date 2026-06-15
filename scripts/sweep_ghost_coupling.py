#!/usr/bin/env python3
"""扫参 VAP ghost 散度/速度惩罚缩放，1.5s hydro_30frame。"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
BIN = BUILD / "hydro_30frame"
OUT_DIR = BUILD / "ghost_coupling_sweep_postfix"
LD = "/usr/local/cuda-13.1/lib64"

# ghost 过强 → 从 1.0 向下扫；也测单通道缩放
GRID = [
    (1.0, 1.0),
    (0.5, 1.0),
    (1.0, 0.5),
    (0.5, 0.5),
    (0.25, 0.5),
    (0.5, 0.25),
    (0.25, 0.25),
    (0.1, 0.25),
    (0.25, 0.1),
    (0.1, 0.1),
]

PAT_SUMMARY = re.compile(
    r"SUMMARY .*?final_vol=([\d.]+).*?final_inverted=(\d+).*?"
    r"ub_disp=\(([^)]+)\).*?hydro_nonzero_pct=([\d.]+)"
)
PAT_BAND = re.compile(
    r"BAND_DIAG n_band=(\d+).*?fluid_influence_fraction=([\d.]+).*?"
    r"ghost_influence_fraction=([\d.]+)"
)


def run_case(div: float, vel: float) -> dict:
    tag = f"div{div:g}_vel{vel:g}".replace(".", "p")
    case_dir = OUT_DIR / tag
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "run.log"
    diag_prefix = str(case_dir / "band")

    cmd = [
        str(BIN),
        "--seconds", "1.5",
        "--sim-hz", "960",
        "--substeps", "18",
        "--ramp", "0.1",
        "--max-force", "5",
        "--ghost-stride", "1",
        "--ghost-div-scale", str(div),
        "--ghost-vel-scale", str(vel),
        "--diag-at", "1.5",
        "--diag-out", diag_prefix,
    ]
    env = {**os.environ, "LD_LIBRARY_PATH": LD}
    proc = subprocess.run(
        cmd,
        cwd=str(BUILD),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout)

    row = {
        "div": div,
        "vel": vel,
        "tag": tag,
        "ok": proc.returncode == 0,
        "vol": None,
        "inv": None,
        "ub_disp": None,
        "hydro_pct": None,
        "n_band": None,
        "fluid_inf_frac": None,
        "ghost_inf_frac": None,
        "stable": False,
        "score": -1e9,
    }
    sm = PAT_SUMMARY.search(proc.stdout)
    bm = PAT_BAND.search(proc.stdout)
    if sm:
        row["vol"] = float(sm.group(1))
        row["inv"] = int(sm.group(2))
        row["ub_disp"] = sm.group(3)
        row["hydro_pct"] = float(sm.group(4))
    if bm:
        row["n_band"] = int(bm.group(1))
        row["fluid_inf_frac"] = float(bm.group(2))
        row["ghost_inf_frac"] = float(bm.group(3))

    row["stable"] = (
        row["vol"] is not None
        and row["inv"] is not None
        and row["vol"] >= 0.94
        and row["inv"] <= 30
    )
    if row["stable"] and row["hydro_pct"] is not None:
        fluid_bonus = (row["fluid_inf_frac"] or 0.0) * 30.0
        row["score"] = row["hydro_pct"] + fluid_bonus - row["inv"] * 0.5
    return row


def main() -> int:
    if not BIN.is_file():
        print(f"missing {BIN}, build hydro_30frame first", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for div, vel in GRID:
        print(f"=== run ghost_div={div} ghost_vel={vel} ===", flush=True)
        rows.append(run_case(div, vel))

    stable_rows = [r for r in rows if r["stable"]]
    ranked = sorted(
        [r for r in rows if r["hydro_pct"] is not None],
        key=lambda r: (
            int(r["stable"]),
            r["hydro_pct"] or 0.0,
            r["fluid_inf_frac"] or 0.0,
            -(r["inv"] or 9999),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None

    summary_path = OUT_DIR / "sweep_summary.tsv"
    with summary_path.open("w") as f:
        f.write(
            "div\tvel\tstable\tvol\tinv\thydro_pct\tfluid_inf_frac\tghost_inf_frac\t"
            "n_band\tub_disp\tscore\ttag\n"
        )
        for r in rows:
            f.write(
                f"{r['div']}\t{r['vel']}\t{int(r['stable'])}\t{r['vol']}\t{r['inv']}\t"
                f"{r['hydro_pct']}\t{r['fluid_inf_frac']}\t{r['ghost_inf_frac']}\t"
                f"{r['n_band']}\t{r['ub_disp']}\t{r['score']:.3f}\t{r['tag']}\n"
            )

    print("\n=== SWEEP SUMMARY (post VAP/face-force fixes) ===")
    print(f"stable cases (vol>=0.94, inv<=30): {len(stable_rows)}/{len(rows)}")
    print(f"table: {summary_path}")
    print("\nRanked by stable > hydro_pct > fluid_inf_frac > -inv:")
    for r in ranked:
        mark = "OK" if r["stable"] else "--"
        print(
            f"  [{mark}] div={r['div']:4g} vel={r['vel']:4g}  "
            f"hydro={r['hydro_pct']:6.2f}%  inv={r['inv']:4d}  vol={r['vol']:.4f}  "
            f"fluid_inf={100*(r['fluid_inf_frac'] or 0):5.1f}%  "
            f"ub=({r['ub_disp']})"
        )
    if best:
        print(
            f"\nbest: div={best['div']} vel={best['vel']} "
            f"hydro_pct={best['hydro_pct']} inv={best['inv']} vol={best['vol']} "
            f"fluid_inf_frac={best['fluid_inf_frac']} ub_disp=({best['ub_disp']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
