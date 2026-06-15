#!/usr/bin/env bash
# FSI 扫参：substeps × particle_spacing
# 默认 ramp=0.1、flow 沿用 Environment (0,0,-0.05)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
OUT="$BUILD/fsi_sweep_ramp010"
BIN="$BUILD/hydro_30frame"
SUMMARY="$OUT/summary.tsv"
LD="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:${LD}"

RAMP="${RAMP:-0.1}"
T_SEC="${T_SEC:-0.5}"
SIM_HZ="${SIM_HZ:-960}"
MAX_FORCE="${MAX_FORCE:-5}"
LOG_EVERY="${LOG_EVERY:-480}"
PHASE="${PHASE:-all}"  # baseline | substeps | spacing | combo | all

mkdir -p "$OUT"

run_case() {
  local run_id="$1"
  local substeps="$2"
  local dx="$3"
  local log="$OUT/${run_id}.log"

  echo ">>> [$run_id] substeps=$substeps dx=$dx ramp=$RAMP"
  unset METAWORM_PARTICLE_SPACING
  if [[ "$dx" != "0.010" ]]; then
    export METAWORM_PARTICLE_SPACING="$dx"
  fi

  stdbuf -oL "$BIN" \
    --sim-hz "$SIM_HZ" \
    --seconds "$T_SEC" \
    --ramp "$RAMP" \
    --max-force "$MAX_FORCE" \
    --substeps "$substeps" \
    --log-every "$LOG_EVERY" \
    2>&1 | tee "$log"

  local summary result
  summary=$(grep '^SUMMARY' "$log" | tail -1 || true)
  result=$(grep '^RESULT' "$log" | tail -1 || true)
  if [[ -z "$summary" || -z "$result" ]]; then
    echo -e "${run_id}\t${substeps}\t${dx}\tFAIL\t-\t-\t-\t-\t-\terror" >> "$SUMMARY"
    return 1
  fi

  python3 - "$run_id" "$substeps" "$dx" "$summary" "$result" <<'PY' >> "$SUMMARY"
import re, sys
run_id, sub, dx, summary, result = sys.argv[1:6]
def grab(pat, text, default="?"):
    m = re.search(pat, text)
    return m.group(1) if m else default
avg = grab(r"avg_step_ms=([0-9.]+)", summary)
vol = grab(r"final_vol=([0-9.]+)", summary)
if vol == "?":
    vol = grab(r"volume_ratio=([0-9.]+)", result)
inv = grab(r"final_inverted=([0-9]+)", summary)
if inv == "?":
    inv = grab(r"inverted=([0-9]+)", result)
dz = grab(r"com_disp_z=([-0-9.eE+]+)", summary)
if dz == "?":
    dz = grab(r"com_disp_z=([-0-9.eE+]+)", result)
status = grab(r"status=([a-z]+)", result, "ok")
print(f"{run_id}\t{sub}\t{dx}\t{status}\t{avg}\t{vol}\t{inv}\t{dz}\t-")
PY
}

judge_runs() {
  python3 - "$SUMMARY" <<'PY'
import sys
path = sys.argv[1]
rows = []
with open(path) as f:
    for line in f:
        if line.startswith("run_id") or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 8:
            continue
        rows.append(parts)

baseline = next((r for r in rows if r[0] == "baseline"), None)
if not baseline:
    sys.exit(0)
try:
    b_avg = float(baseline[4])
    b_inv = int(float(baseline[6]))
except ValueError:
    sys.exit(0)

updated = []
for r in rows:
    if len(r) < 9:
        r += ["?"] * (9 - len(r))
    verdict = r[8]
    if r[0] == "baseline":
        r[8] = "baseline"
    elif r[3] != "ok" or r[4] == "FAIL":
        r[8] = "fail"
    else:
        vol = float(r[5])
        inv = int(float(r[6]))
        avg = float(r[4])
        speedup = (b_avg - avg) / b_avg * 100.0
        if vol < 0.94 or inv > 20:
            r[8] = "reject"
        elif vol < 0.950 or inv > b_inv + 5:
            r[8] = "marginal"
        elif speedup >= 33.0:
            r[8] = f"pass(+{speedup:.1f}%)"
        elif speedup >= 20.0:
            r[8] = f"ok(+{speedup:.1f}%)"
        else:
            r[8] = f"slow(+{speedup:.1f}%)"
    updated.append("\t".join(r))

with open(path, "w") as f:
    f.write("run_id\tsubsteps\tdx\tstatus\tavg_step_ms\tvol_ratio\tinverted\tcom_disp_z\tverdict\n")
    for line in updated:
        f.write(line + "\n")
PY
}

echo "# FSI sweep started $(date) ramp=$RAMP t_sec=$T_SEC" > "$SUMMARY"
echo -e "run_id\tsubsteps\tdx\tstatus\tavg_step_ms\tvol_ratio\tinverted\tcom_disp_z\tverdict" >> "$SUMMARY"

if [[ "$PHASE" == "all" || "$PHASE" == "baseline" ]]; then
  run_case "baseline" 24 0.010 || true
  judge_runs
fi

if [[ "$PHASE" == "all" || "$PHASE" == "substeps" ]]; then
  for s in 18 16 12; do
    run_case "sub${s}" "$s" 0.010 || true
  done
  judge_runs
fi

if [[ "$PHASE" == "all" || "$PHASE" == "spacing" ]]; then
  for dx in 0.011 0.012; do
    run_case "dx${dx#0.}" 24 "$dx" || true
  done
  judge_runs
fi

if [[ "$PHASE" == "all" || "$PHASE" == "combo" ]]; then
  run_case "combo_sub18_dx011" 18 0.011 || true
  run_case "combo_sub16_dx011" 16 0.011 || true
  run_case "combo_sub16_dx010" 16 0.010 || true
  judge_runs
fi

echo "=== sweep summary: $SUMMARY ==="
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "FSI sweep done $(date)"
