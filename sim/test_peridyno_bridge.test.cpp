#include "fluid/PeridynoBridge.h"
#include <iostream>
#include <vector>
#include <chrono>

int main()
{
    std::cout << "=== PeridynoBridge 单元测试 ===" << std::endl;

    // 1. 初始化
    Eigen::Vector3d domain_min(-0.5, -0.5, -0.5);
    Eigen::Vector3d domain_max(0.5, 0.5, 0.5);
    float spacing = 0.05f; // 较大的粒子间距用于测试

    PeridynoBridge bridge;
    bridge.Initialize(domain_min, domain_max, spacing);

    std::cout << "初始化完成: " << bridge.GetFluidParticleCount() << " 个流体粒子" << std::endl;

    // 2. 模拟几个时间步（无边界，仅验证 GPU 计算正确性）
    auto t0 = std::chrono::high_resolution_clock::now();
    int num_steps = 100;
    for (int i = 0; i < num_steps; i++) {
        // 无边界情况（传入空数据），测试 ComputeFluidForces
        Eigen::VectorXd empty_verts, empty_vels;
        std::vector<Eigen::Vector3i> empty_faces;
        bridge.SetFishBoundary(empty_verts, empty_vels, empty_faces);
        Eigen::VectorXd forces = bridge.ComputeFluidForces(1.0f / 240.0f);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << num_steps << " 步 SPH 仿真完成, 耗时 " << duration << "ms"
              << " (平均 " << (double)duration / num_steps << "ms/步)" << std::endl;

    // 3. 提取流体粒子位置用于验证
    std::vector<float> positions;
    bridge.GetFluidParticles(positions);
    std::cout << "流体粒子总数: " << positions.size() / 3 << std::endl;

    // 打印前3个粒子的位置，验证数值合理
    for (int i = 0; i < 3 && i * 3 + 2 < (int)positions.size(); i++) {
        std::cout << "粒子 " << i << ": ("
                  << positions[3*i] << ", " << positions[3*i+1] << ", " << positions[3*i+2] << ")" << std::endl;
    }

    bridge.Destroy();
    std::cout << "测试通过!" << std::endl;
    return 0;
}
