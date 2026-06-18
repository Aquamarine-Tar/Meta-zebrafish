# 480 Hz flow sweep status

Command used for the valid run:

```bash
build/fsi_force_snapshot --out result/flow_sweep_480hz_1p5s_valid/flow_0/run_1 --seconds 1.5 --t-start 0.5 --t-step 0.5 --sim-hz 480 --ramp 0.5 --max-force 5 --substeps 18 --flow 0,0,0
```

The run used the real GPU path:

- Initial fluid particles: 99758
- Initial fish-interior particles removed: 1535
- Initial boundary-layer particles added: 3909

Result:

| flow_z | run | avg_step_ms | final_vol | shrink_pct | inverted | ub_disp | final force coverage |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | 1 | 958.881649 | 0.966841831 | 3.315817 | 6 | (-0.996574, -0.267800, 0.606114) | 0/7144 |

Snapshot coverage:

| time_s | surface force coverage | force_sum |
|---:|---:|---|
| 0.5 | 7144/7144 | (-9.06884, -4.5361, 7.75333) |
| 1.0 | 0/7144 | (0, 0, 0) |
| 1.5 | 0/7144 | (0, 0, 0) |

Pressure visualizations were generated at the last full-coverage snapshot:

- `flow_0/run_1/t0.50_pressure_zy.png`
- `flow_0/run_1/t0.50_pressure_zx.png`

Conclusion:

This parameter set is not suitable for the requested 12-run sweep at 480 Hz and 1.5 s. It remains volumetrically stable enough by the stated volume criterion, but the fish loses hydrodynamic force coverage after 0.5 s and has zero surface hydro force by 1.0 s.

Notes on invalid batch attempts:

- Shell redirection or compound shell loops triggered `CUDA driver version is insufficient for CUDA runtime version` in this sandbox path.
- Direct invocation of `build/fsi_force_snapshot ...` uses the real GPU and gives valid fluid initialization.
