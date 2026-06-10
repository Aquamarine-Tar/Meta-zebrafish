#!/bin/bash
# 渐进恢复 hydro_force_scale，5s 仿真，记录体积/翻转指标
set -euo pipefail
BUILD="$(cd "$(dirname "$0")/../build" && pwd)"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}"
BIN="$BUILD/fsi_stability"
LOG="$BUILD/hydro_sweep_results.txt"

BASE="--seconds 5 --substeps 24 --stiffness 0.3 --damping 25 --ramp 12 --max-force 3"
HYDROS=(0.00 0.02 0.05 0.08 0.10 0.12 0.15 0.20 0.25 0.30 0.40 0.50 0.75 1.00)

echo "hydro_sweep started $(date)" | tee "$LOG"
printf "%-8s %-14s %-12s %-10s %-8s\n" "hydro" "volume_ratio" "vol_shrink%" "inverted" "status" | tee -a "$LOG"
printf "%-8s %-14s %-12s %-10s %-8s\n" "----" "------------" "----------" "--------" "------" | tee -a "$LOG"

for h in "${HYDROS[@]}"; do
  echo ">>> Running hydro=$h ..." | tee -a "$LOG"
  OUT=$(stdbuf -oL "$BIN" $BASE --hydro "$h" 2>&1) || true
  RESULT=$(echo "$OUT" | grep "^RESULT" | tail -1)
  if [[ -z "$RESULT" ]]; then
    printf "%-8s %-14s %-12s %-10s %-8s\n" "$h" "FAIL" "-" "-" "error" | tee -a "$LOG"
    continue
  fi
  VOL=$(echo "$RESULT" | sed -n 's/.*volume_ratio=\([^ ]*\).*/\1/p')
  INV=$(echo "$RESULT" | sed -n 's/.*inverted=\([^ ]*\).*/\1/p')
  SHRINK=$(python3 -c "print(f'{(1-float('$VOL'))*100:.2f}')")
  STATUS="OK"
  python3 -c "import sys; v=float('$VOL'); i=int('$INV'); sys.exit(0 if v>=0.95 and i<=20 else 1)" || STATUS="BAD"
  printf "%-8s %-14s %-12s %-10s %-8s\n" "$h" "$VOL" "$SHRINK" "$INV" "$STATUS" | tee -a "$LOG"
  if [[ "$STATUS" == "BAD" ]]; then
    echo "Stop: hydro=$h unacceptable (vol<0.95 or inverted>20)" | tee -a "$LOG"
    break
  fi
done
echo "hydro_sweep done $(date)" | tee -a "$LOG"
