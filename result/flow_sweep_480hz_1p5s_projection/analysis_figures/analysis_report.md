# Flow Sweep Analysis

Data source: `result/flow_sweep_480hz_1p5s_projection`.

This analysis includes all completed runs found under the directory, including the later `flow_m0p2` runs that are not present in the original `summary.csv`.

## Figures

- [01_summary_dashboard.png](01_summary_dashboard.png)
- [02_registered_displacement_components.png](02_registered_displacement_components.png)
- [03_net_z_force_heatmap.png](03_net_z_force_heatmap.png)
- [04_surface_force_clouds_t1p50.png](04_surface_force_clouds_t1p50.png)
- [05_left_right_z_force_balance.png](05_left_right_z_force_balance.png)
- [06_pressure_existing_png_montage.png](06_pressure_existing_png_montage.png)

## Grouped Metrics

| flow_z | runs | avg step ms | max step ms | volume | shrink % | inverted | ub_z mean ± sd | ub_z range | coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| -0.2 | 2 | 2441.4 ± 1144.2 | 3175.7 | 0.9697 ± 0.0072 | 3.03 | 4.5 | 0.2947 ± 0.1715 | 0.1735..0.4160 | 100.0% |
| -0.08 | 3 | 1599.4 ± 9.1 | 1925.7 | 0.9643 ± 0.0038 | 3.57 | 4.0 | 0.3739 ± 0.0954 | 0.2672..0.4509 | 100.0% |
| -0.05 | 3 | 1590.2 ± 15.0 | 1847.0 | 0.9711 ± 0.0101 | 2.89 | 4.0 | 0.3609 ± 0.0309 | 0.3289..0.3905 | 100.0% |
| -0.025 | 3 | 1591.7 ± 5.4 | 1822.8 | 0.9678 ± 0.0068 | 3.22 | 5.3 | 0.3970 ± 0.1166 | 0.2627..0.4728 | 100.0% |
| 0 | 3 | 1571.6 ± 10.4 | 1829.5 | 0.9660 ± 0.0038 | 3.40 | 8.0 | 0.4137 ± 0.0735 | 0.3504..0.4944 | 100.0% |

## Conclusions

- Stability is acceptable in these logs: all final volume ratios are above 0.95, so shrinkage stays below 5%, and inverted tet counts remain low.
- Vertex force coverage is excellent: all completed runs report 100% nonzero hydro-force coverage at the final snapshot, and the snapshot coverage table reports 100% over the original 36 snapshots.
- Runtime misses the earlier 750 ms target by a wide margin: most 480 Hz runs are around 1.56-1.61 s per step, and `flow_z=-0.2` run 1 is much slower.
- The main physical concern is reproducibility of flow response. Increasing negative background flow from 0 to -0.08 does not monotonically reduce registered +z displacement, and the later -0.2 runs are inconsistent (`ub_z=0.4160` and `0.1735`).
- Therefore this result set supports the diagnosis that the projection/low ghost-coupling parameter state preserved coverage and stability, but did not produce a reliable background-flow drag response.
- The existing pressure images are visually useful, but the aggregate displacement/force statistics do not yet demonstrate the expected left-right pressure difference translating into a reproducible z-direction swimming response.
