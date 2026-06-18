# flow_sweep_480hz_1p5s_projection 数据分析

数据源：`result/flow_sweep_480hz_1p5s_projection`。

本报告纳入该目录下所有已完成的 run：原始 4 个流速各 3 次，共 12 次，以及后续补跑的 `flow_z=-0.2` 两次。因此总计 14 次 run。注意：这批数据对应的是当时的 projection/低 ghost coupling 参数状态，即 `hydro_transverse_projection=1`、`ghost_div_scale=0.02`、`ghost_vel_scale=0.05`、`hydro_scale=0.1`。

## 图像索引

- [01_summary_dashboard.png](01_summary_dashboard.png)：稳定性、z 位移、耗时、受力覆盖总览。
- [02_registered_displacement_components.png](02_registered_displacement_components.png)：配准位移的 x/y/z 分量随流速变化。
- [03_net_z_force_heatmap.png](03_net_z_force_heatmap.png)：`torque_summary.csv` 中每个时间点的净 z 向水动力。
- [04_surface_force_clouds_t1p50.png](04_surface_force_clouds_t1p50.png)：t=1.50 s 表面顶点水动力云图。
- [05_left_right_z_force_balance.png](05_left_right_z_force_balance.png)：按 y 中位面切分左右两侧后的 z 向力差。
- [06_pressure_existing_png_montage.png](06_pressure_existing_png_montage.png)：已有压力分布 PNG 的对比拼图。
- [augmented_metrics.csv](augmented_metrics.csv)：重新汇总后的完整 metrics 表，包含后续的 `flow_z=-0.2` run。

## 分组统计

| flow_z | runs | avg step ms | volume | shrink % | inverted | ub_z mean ± sd | ub_z range | force coverage |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| -0.2 | 2 | 2441.4 ± 1144.2 | 0.9697 ± 0.0072 | 3.03 | 4.5 | 0.2947 ± 0.1715 | 0.1735..0.4160 | 100.0% |
| -0.08 | 3 | 1599.4 ± 9.1 | 0.9643 ± 0.0038 | 3.57 | 4.0 | 0.3739 ± 0.0954 | 0.2672..0.4509 | 100.0% |
| -0.05 | 3 | 1590.2 ± 15.0 | 0.9711 ± 0.0101 | 2.89 | 4.0 | 0.3609 ± 0.0309 | 0.3289..0.3905 | 100.0% |
| -0.025 | 3 | 1591.7 ± 5.4 | 0.9678 ± 0.0068 | 3.22 | 5.3 | 0.3970 ± 0.1166 | 0.2627..0.4728 | 100.0% |
| 0 | 3 | 1571.6 ± 10.4 | 0.9660 ± 0.0038 | 3.40 | 8.0 | 0.4137 ± 0.0735 | 0.3504..0.4944 | 100.0% |

## 结论

1. 稳定性是合格的：所有 run 的最终体积比都大于 0.95，体积缩减约 2.0% 到 4.0%，明显小于 10%；翻转单元数量也维持在较低水平。

2. 顶点受力覆盖很好：所有已完成 run 的最终 `hydro_nonzero_pct` 都是 100%。原始 36 个快照的 `snapshot_coverage.csv` 也显示 100% 覆盖。

3. 运行时间没有达到 750 ms 目标：原始 0 到 -0.08 流速组平均每步约 1.57 到 1.61 秒；`flow_z=-0.2` 的 run 1 明显更慢，平均约 3.25 秒。

4. 最主要异常仍然是流速响应不可信：从 `flow_z=0` 增大负向来流到 `-0.08`，配准 z 位移没有单调减小；`flow_z=-0.2` 两次 run 的差异也很大，分别为 `ub_z=0.4160` 和 `0.1735`。这说明该参数状态下，负 z 来流并没有稳定、可复现地削弱正 z 游动位移。

5. 因此，这批最新日志支持一个明确判断：projection/低 ghost coupling 参数能够保持稳定性和表面受力覆盖，但它没有给出合理的背景流阻滞响应。这个结论也解释了为什么用户观察到“这么大的负 z 流速，仍然有很多正向位移”。

6. 压力图和表面力云图可以看到局部红蓝压力/力区，但这些局部分布没有在统计上转化为期望的、可复现的左右压力差推动 z 向游动结果。后续应优先使用已回退的 `hydro_scale=1.0`、`ghost_div_scale=0.25`、`ghost_vel_scale=0.25`、`hydro_transverse_projection=0` 参数重新跑同样扫参，再比较本报告中的图。
