/**
 * 快速力反馈验证
 */
#include "Environment.h"
#include <iostream>
#include <cmath>

int main()
{
    Environment env("data/fish.meta", false);
    std::cout << "=== Force feedback check ===" << std::endl;

    const auto& x = env.GetSoftWorld()->GetPositions();
    const auto& v = env.GetSoftWorld()->GetVelocities();
    const auto& contours = env.GetCreature()->GetContours();

    // 手动调用 SPH 看返回的力
    auto* bridge = env.GetFluidBridge();
    bridge->SetFishBoundary(x, v, contours);
    double dt = 1.0 / 240.0;
    Eigen::VectorXd f = bridge->ComputeFluidForces((float)dt);

    // 检查力的统计
    double max_f = 0, sum_f = 0;
    for (int i = 0; i < f.size(); i++) {
        double fi = std::abs(f(i));
        max_f = std::max(max_f, fi);
        sum_f += fi;
    }
    std::cout << "Forces size: " << f.size() << std::endl;
    std::cout << "Max |force|: " << max_f << std::endl;
    std::cout << "Sum |force|: " << sum_f << std::endl;
    std::cout << "Non-zero count: ";
    int nz = 0;
    for (int i = 0; i < f.size(); i++) if (f(i) != 0) nz++;
    std::cout << nz << std::endl;

    // 运行几步看力是否持续变化
    for (int step = 0; step < 3; step++) {
        const auto& x2 = env.GetSoftWorld()->GetPositions();
        const auto& v2 = env.GetSoftWorld()->GetVelocities();
        bridge->SetFishBoundary(x2, v2, contours);
        f = bridge->ComputeFluidForces((float)dt);
        double step_max = 0;
        for (int i = 0; i < f.size(); i++) step_max = std::max(step_max, std::abs(f(i)));
        std::cout << "Step " << step << " max force: " << step_max << std::endl;
        env.Step();
    }
    return 0;
}
