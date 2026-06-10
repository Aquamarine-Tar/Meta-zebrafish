/**
 * headless_test.test.cpp: 无图形界面的集成测试
 * 验证 Environment 初始化 + PeridynoBridge SPH 仿真流程
 */
#include "Environment.h"
#include <iostream>
#include <chrono>

int main()
{
    std::cout << "=== Headless 集成测试 ===" << std::endl;

    auto tinit0 = std::chrono::high_resolution_clock::now();
    Environment env("data/fish.meta", false);
    auto tinit1 = std::chrono::high_resolution_clock::now();
    auto init_ms = std::chrono::duration<double, std::milli>(tinit1 - tinit0).count();

    std::cout << "Init time: " << init_ms << "ms" << std::endl;
    std::cout << "  Muscles: " << env.GetCreature()->GetNumMuscleSegments() << std::endl;
    std::cout << "  FEM vertices: " << env.GetSoftWorld()->GetPositions().size() / 3 << std::endl;

    // 运行几步测试，每步输出耗时
    const int N = 5;
    double total_step_ms = 0;
    for (int i = 0; i < N; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        env.Step();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_step_ms += ms;
        std::cout << "  Step " << i << ": " << ms << "ms" << std::endl;
    }
    std::cout << "Avg step: " << total_step_ms / N << "ms" << std::endl;
    std::cout << "SUCCESS: " << N << " steps completed" << std::endl;
    return 0;
}
