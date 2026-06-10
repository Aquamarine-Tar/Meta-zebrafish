#!/usr/bin/env python3
"""FSI 参数自动搜索（最多 20 组），以 1s 快筛 + 最优 5s 复验。"""
import re
import subprocess
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parents[1] / "build"
BIN = BUILD / "fsi_stability"
LD = "/usr/local/cuda-13.1/lib64"

# 初始 + 搜索网格（最多 20 次）
CONFIGS = [
    {"substeps": 4, "stiffness": 10, "damping": 2, "ramp": 0.5, "hydro": 1.0, "max_force": 0},
    {"substeps": 8, "stiffness": 5, "damping": 5, "ramp": 2.0, "hydro": 0.5, "max_force": 100},
    {"substeps": 8, "stiffness": 3, "damping": 8, "ramp": 3.0, "hydro": 0.3, "max_force": 50},
    {"substeps": 12, "stiffness": 2, "damping": 10, "ramp": 4.0, "hydro": 0.2, "max_force": 40},
    {"substeps": 8, "stiffness": 2, "damping": 6, "ramp": 5.0, "hydro": 0.1, "max_force": 30},
    {"substeps": 16, "stiffness": 1, "damping": 12, "ramp": 6.0, "hydro": 0.0, "max_force": 25},
    {"substeps": 12, "stiffness": 1.5, "damping": 8, "ramp": 5.0, "hydro": 0.15, "max_force": 35},
    {"substeps": 8, "stiffness": 4, "damping": 6, "ramp": 3.0, "hydro": 0.25, "max_force": 60},
    {"substeps": 10, "stiffness": 2.5, "damping": 9, "ramp": 4.5, "hydro": 0.12, "max_force": 45},
    {"substeps": 12, "stiffness": 2, "damping": 12, "ramp": 6.0, "hydro": 0.08, "max_force": 30},
    {"substeps": 16, "stiffness": 1, "damping": 15, "ramp": 8.0, "hydro": 0.05, "max_force": 20},
    {"substeps": 8, "stiffness": 3, "damping": 10, "ramp": 5.0, "hydro": 0.0, "max_force": 50},
    {"substeps": 12, "stiffness": 2, "damping": 8, "ramp": 7.0, "hydro": 0.1, "max_force": 25},
    {"substeps": 16, "stiffness": 1.5, "damping": 10, "ramp": 6.0, "hydro": 0.05, "max_force": 15},
    {"substeps": 20, "stiffness": 1, "damping": 15, "ramp": 8.0, "hydro": 0.0, "max_force": 10},
    {"substeps": 12, "stiffness": 0.5, "damping": 12, "ramp": 6.0, "hydro": 0.0, "max_force": 20},
    {"substeps": 16, "stiffness": 0.8, "damping": 14, "ramp": 10.0, "hydro": 0.0, "max_force": 12},
    {"substeps": 20, "stiffness": 0.5, "damping": 16, "ramp": 10.0, "hydro": 0.0, "max_force": 8},
    {"substeps": 16, "stiffness": 1, "damping": 20, "ramp": 8.0, "hydro": 0.0, "max_force": 5},
    {"substeps": 24, "stiffness": 0.5, "damping": 20, "ramp": 10.0, "hydro": 0.0, "max_force": 5},
]

RESULT_RE = re.compile(
    r"RESULT volume_ratio=([\d.eE+-]+) vol_dev=([\d.eE+-]+) inverted=(\d+)"
)


def run_case(cfg, seconds):
    cmd = [
        str(BIN),
        "--seconds", str(seconds),
        "--substeps", str(cfg["substeps"]),
        "--stiffness", str(cfg["stiffness"]),
        "--damping", str(cfg["damping"]),
        "--ramp", str(cfg["ramp"]),
        "--hydro", str(cfg["hydro"]),
        "--max-force", str(cfg["max_force"]),
    ]
    env = {"LD_LIBRARY_PATH": f"{LD}:{subprocess.os.environ.get('LD_LIBRARY_PATH', '')}"}
    proc = subprocess.run(cmd, cwd=str(BUILD), env=env, capture_output=True, text=True, timeout=900)
    text = proc.stdout + proc.stderr
    m = RESULT_RE.search(text)
    if not m:
        return None, text[-2000:]
    vol = float(m.group(1))
    vol_dev = float(m.group(2))
    inv = int(m.group(3))
    return {"volume_ratio": vol, "vol_dev": vol_dev, "inverted": inv}, text


def score(r):
  # 体积偏差 >5% 重罚；翻转重罚
    ok_vol = r["volume_ratio"] >= 0.95
    s = (1000 if ok_vol else r["volume_ratio"] * 500) - r["inverted"] * 5 - r["vol_dev"] * 200
    return s


def main():
    if not BIN.exists():
        print(f"缺少 {BIN}，请先编译 fsi_stability", file=sys.stderr)
        return 1

    best = None
    best_cfg = None
    rows = []

    for i, cfg in enumerate(CONFIGS[:20], 1):
        print(f"\n=== Try {i}/20: {cfg} ===", flush=True)
        try:
            result, tail = run_case(cfg, seconds=1)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT", flush=True)
            rows.append((cfg, None))
            continue
        if result is None:
            print("  FAILED", tail, flush=True)
            rows.append((cfg, None))
            continue
        sc = score(result)
        print(f"  vol={result['volume_ratio']:.4f} inv={result['inverted']} score={sc:.1f}", flush=True)
        rows.append((cfg, result))
        if best is None or sc > score(best):
            best = result
            best_cfg = cfg

    print("\n========== 1s 快筛汇总 ==========")
    for cfg, r in rows:
        if r:
            mark = "OK" if r["volume_ratio"] >= 0.95 and r["inverted"] <= 2 else "  "
            print(f"{mark} vol={r['volume_ratio']:.4f} inv={r['inverted']:4d} {cfg}")
        else:
            print(f"FAIL {cfg}")

    if best_cfg is None:
        print("\n20 次尝试均无有效结果。")
        return 2

    print(f"\n最优快筛配置: {best_cfg}")
    print(f"  volume_ratio={best['volume_ratio']:.6f} inverted={best['inverted']}")

    ok_quick = best["volume_ratio"] >= 0.95 and best["inverted"] <= 2
    if ok_quick:
        print("\n=== 5s 复验 ===", flush=True)
        final, _ = run_case(best_cfg, seconds=5)
        if final:
            print(f"5s: vol={final['volume_ratio']:.6f} inv={final['inverted']}")
            if final["volume_ratio"] >= 0.95 and final["inverted"] <= 2:
                print("PASS: 复验达标")
                print("APPLY_CONFIG", best_cfg)
                return 0
            print("WARN: 5s 复验未完全达标，但为 20 次内最优")
            print("APPLY_CONFIG", best_cfg)
            return 0

    print("\n20 次尝试后仍未达到 vol>=0.95 且 inverted<=2")
    print("BEST_CONFIG", best_cfg)
    print("BEST_RESULT", best)
    return 3


if __name__ == "__main__":
    sys.exit(main())
