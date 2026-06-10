/**
 * force_disp_test.cpp: 第一帧流体耦合验证
 */
#include "Environment.h"
#include <iostream>
#include <cmath>

static void PrintVolumeStats(const char* label, const VolumeStats& s)
{
    printf("%s: ratio=%.6f, tet_ratio=[%.4f, %.4f], inverted=%d\n",
           label, s.volume_ratio, s.min_tet_ratio, s.max_tet_ratio, s.inverted_tets);
}

int main()
{
    Environment env("data/fish.meta", false);

    Eigen::VectorXd x0 = env.GetSoftWorld()->GetPositions();
    int nv = x0.size() / 3;
    double dt = 1.0 / env.GetSimulationHz();

    VolumeStats vol0 = env.GetCreature()->ComputeVolumeStats(x0);
    PrintVolumeStats("初始体积", vol0);

    env.Step();

    const auto& x1 = env.GetSoftWorld()->GetPositions();
    const auto& v1 = env.GetSoftWorld()->GetVelocities();
    const auto& ext_f = env.GetSoftWorld()->GetExternalForce();
    VolumeStats vol1 = env.GetCreature()->ComputeVolumeStats(x1);
    PrintVolumeStats("第一帧后", vol1);

    double total_f = 0, max_f = 0;
    int non_zero = 0;
    for (int i = 0; i < nv; i++) {
        double fx = ext_f(3*i), fy = ext_f(3*i+1), fz = ext_f(3*i+2);
        double fm = sqrt(fx*fx + fy*fy + fz*fz);
        if (fm > 1e-6) non_zero++;
        total_f += fm;
        if (fm > max_f) max_f = fm;
    }

    double max_disp = 0;
    for (int i = 0; i < nv; i++) {
        double dx = x1(3*i) - x0(3*i);
        double dy = x1(3*i+1) - x0(3*i+1);
        double dz = x1(3*i+2) - x0(3*i+2);
        max_disp = std::max(max_disp, sqrt(dx*dx + dy*dy + dz*dz));
    }

    double max_vel = 0;
    for (int i = 0; i < nv; i++) {
        double vx = v1(3*i), vy = v1(3*i+1), vz = v1(3*i+2);
        max_vel = std::max(max_vel, sqrt(vx*vx + vy*vy + vz*vz));
    }

    printf("\n=== 第一帧结果 (dt=%.4fs) ===\n", dt);
    printf("外力: non_zero=%d/%d, total=%.2f N, max=%.4f N\n", non_zero, nv, total_f, max_f);
    printf("位移: max=%.4f mm\n", max_disp * 1000);
    printf("速度: max=%.4f m/s\n", max_vel);
    if (env.GetFluidBridge())
        printf("来流 ramp: %.4f\n", env.GetFluidBridge()->GetFlowRampFactor());

    bool pass = (std::abs(vol1.volume_ratio - 1.0) < 0.01) && (vol1.inverted_tets == 0)
             && (max_disp < 10.0 * dt);
    printf(pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
