/**
 * FSI 稳定性测试：模拟 sin 肌肉模式，输出体积/翻转指标
 * 用法:
 *   ./fsi_stability [--seconds 5] [--substeps N] [--stiffness F] [--damping F]
 *                   [--flow F] [--ramp F] [--hydro F] [--max-force F]
 */
#include "Environment.h"
#include "Worm.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <cstdlib>

static double g_pi = 3.14159265358979323846;

static Eigen::VectorXd MakeSineActivations(Environment* env, double time_sec)
{
    const double amplitude = 0.25;
    const double offset = 0.25;
    const double temporalFrequency = 1.0;
    const double spatialCycles = 1.0;
    const auto& muscles = env->GetCreature()->GetMuscles();
    Eigen::VectorXd activations = Eigen::VectorXd::Zero(env->GetMuscleActivationSize());

    int idx = 0;
    for (int muscleIdx = 0; muscleIdx < (int)muscles.size(); ++muscleIdx)
    {
        int sampleCount = muscles[muscleIdx]->GetNumSampling();
        double musclePhase = (muscleIdx % 2 == 0) ? 0.0 : g_pi;
        for (int sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx)
        {
            double bodyPhase = 2.0 * g_pi * spatialCycles * sampleIdx / sampleCount;
            double signal = offset + amplitude * std::sin(2.0 * g_pi * temporalFrequency * time_sec - bodyPhase + musclePhase);
            activations[idx + sampleIdx] = std::min(std::max(signal, 0.0), 1.0);
        }
        idx += sampleCount;
    }
    return activations;
}

static float ArgFloat(int argc, char** argv, const char* key, float def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == key)
            return std::atof(argv[i + 1]);
    return def;
}

static int ArgInt(int argc, char** argv, const char* key, int def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == key)
            return std::atoi(argv[i + 1]);
    return def;
}

int main(int argc, char** argv)
{
    const double seconds = ArgFloat(argc, argv, "--seconds", 5.0f);
    Environment env("data/fish.meta", false);
    PeridynoBridge* bridge = env.GetFluidBridge();
    if (!bridge || !bridge->IsInitialized())
    {
        std::cerr << "RESULT status=fail reason=no_fluid_bridge" << std::endl;
        return 1;
    }

    // 默认沿用 Environment 内已配置的 FSI 参数；仅当命令行显式传入时才覆盖
    int substeps = bridge->GetSubsteps();
    float stiffness = 0.3f, damping = 25.0f, ramp = 12.0f, hydro = 0.035f, max_force = 3.0f;
    for (int i = 1; i + 1 < argc; ++i)
    {
        std::string k = argv[i];
        if (k == "--substeps") substeps = std::atoi(argv[i + 1]);
        else if (k == "--stiffness") stiffness = std::atof(argv[i + 1]);
        else if (k == "--damping") damping = std::atof(argv[i + 1]);
        else if (k == "--ramp") ramp = std::atof(argv[i + 1]);
        else if (k == "--hydro") hydro = std::atof(argv[i + 1]);
        else if (k == "--max-force") max_force = std::atof(argv[i + 1]);
    }
    bridge->SetSubsteps(substeps);
    bridge->SetContactParams(stiffness, damping);
    bridge->SetFlowRampTime(ramp);
    bridge->SetHydroForceScale(hydro);
    bridge->SetMaxVertexForce(max_force);

    const int sim_hz = (int)env.GetSimulationHz();
    const int ctrl_hz = (int)env.GetControlHz();
    const int ratio = sim_hz / ctrl_hz;
    const int total_steps = (int)(seconds * sim_hz);

    std::cout << "=== FSI stability test ===" << std::endl;
    std::cout << "substeps=" << substeps << " stiffness=" << stiffness
              << " damping=" << damping << " ramp=" << ramp
              << " hydro=" << hydro << " max_force=" << max_force << std::endl;

    for (int step = 0; step < total_steps; ++step)
    {
        if (step % ratio == 0)
        {
            double t = step / (double)sim_hz;
            env.GetCreature()->SetActivationLevelsDirectly(MakeSineActivations(&env, t));
            env.SetPhase(step / ratio);
        }
        env.Step();
    }

    const Eigen::VectorXd& x = env.GetSoftWorld()->GetPositions();
    VolumeStats stats = env.GetCreature()->ComputeVolumeStats(x);
    const double vol_dev = std::abs(stats.volume_ratio - 1.0);

    std::cout << "RESULT volume_ratio=" << stats.volume_ratio
              << " vol_dev=" << vol_dev
              << " inverted=" << stats.inverted_tets
              << " min_tet=" << stats.min_tet_ratio
              << " max_tet=" << stats.max_tet_ratio
              << " substeps=" << substeps
              << " stiffness=" << stiffness
              << " damping=" << damping
              << " ramp=" << ramp
              << " hydro=" << hydro
              << " max_force=" << max_force
              << std::endl;
    return 0;
}
