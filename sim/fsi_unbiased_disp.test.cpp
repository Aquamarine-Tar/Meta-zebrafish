/**
 * 记录 FEM 质心与无偏位移（EvalBodyTransform 平移 t，同可视化绿色轨迹），
 * 对比 contact+hydro / hydro-only 两种工况。
 *
 * 用法:
 *   ./fsi_unbiased_disp --seconds 3 --mode both --out build/fsi_disp_compare/both.csv
 *   ./fsi_unbiased_disp --seconds 3 --mode hydro --out build/fsi_disp_compare/hydro.csv
 *   ./fsi_unbiased_disp --seconds 3 --out-dir build/fsi_disp_compare   # 依次跑 both + hydro
 */
#include "Environment.h"
#include "Worm.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

static std::string ArgString(int argc, char** argv, const char* key, const std::string& def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == key)
            return argv[i + 1];
    return def;
}

static float ArgFloat(int argc, char** argv, const char* key, float def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == key)
            return std::atof(argv[i + 1]);
    return def;
}

static bool ArgFlag(int argc, char** argv, const char* key)
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == key)
            return true;
    return false;
}

static Eigen::Vector3d ComputeCom(const Eigen::VectorXd& positions)
{
    const int n = positions.size() / 3;
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    for (int i = 0; i < n; ++i)
        com += positions.segment<3>(3 * i);
    return com / std::max(n, 1);
}

/** 无偏位移：参考构形 → 当前构形的刚体配准平移（可视化绿色轨迹） */
static Eigen::Vector3d ComputeUnbiasedTranslation(Worm* worm, const Eigen::VectorXd& x)
{
    const Eigen::VectorXd& ref = worm->GetVerticesReference();
    return worm->EvalBodyTransform(ref, x).translation();
}

static void ConfigureBridge(PeridynoBridge* bridge, const std::string& mode)
{
    if (mode == "hydro")
    {
        bridge->SetEnableContactRepulsion(false);
        bridge->SetEnableHydroPressure(true);
    }
    else if (mode == "both")
    {
        bridge->SetEnableContactRepulsion(true);
        bridge->SetEnableHydroPressure(true);
    }
    else
    {
        std::cerr << "bad mode: " << mode << " (use both|hydro)\n";
        std::exit(1);
    }
    // 与 Environment.cpp 默认 FSI 参数一致
    bridge->SetSubsteps(24);
    bridge->SetContactParams(0.3f, 15.0f);
    bridge->SetFlowRampTime(12.0f);
    bridge->SetHydroForceScale(1.0f);
    bridge->SetMaxVertexForce(3.0f);
}

static bool RunOneMode(int argc, char** argv, const std::string& mode, const std::string& out_csv)
{
    const double seconds = ArgFloat(argc, argv, "--seconds", 3.0f);
    const double t_start = ArgFloat(argc, argv, "--t-start", 2.0f);
    const double t_end = ArgFloat(argc, argv, "--t-end", 3.0f);

    Environment env("data/fish.meta", false);
    env.SetVolumeLogIntervalSteps(0);
    PeridynoBridge* bridge = env.GetFluidBridge();
    if (!bridge || !bridge->IsInitialized())
    {
        std::cerr << "RESULT status=fail reason=no_fluid_bridge mode=" << mode << std::endl;
        return false;
    }
    ConfigureBridge(bridge, mode);

    const int sim_hz = (int)env.GetSimulationHz();
    const int ctrl_hz = (int)env.GetControlHz();
    const int ratio = sim_hz / ctrl_hz;
    const int total_steps = (int)(seconds * sim_hz);

    std::ofstream out(out_csv);
    if (!out)
    {
        std::cerr << "cannot write " << out_csv << std::endl;
        return false;
    }
    out << "sim_time,com_x,com_y,com_z,ub_x,ub_y,ub_z\n";

    std::cout << "=== unbiased disp record mode=" << mode
              << " seconds=" << seconds
              << " window=[" << t_start << "," << t_end << "]"
              << " contact=" << (bridge->GetEnableContactRepulsion() ? 1 : 0)
              << " hydro=" << (bridge->GetEnableHydroPressure() ? 1 : 0)
              << " -> " << out_csv << " ===\n";

    int window_samples = 0;
    for (int step = 0; step < total_steps; ++step)
    {
        if (step % ratio == 0)
        {
            const double t = step / (double)sim_hz;
            env.GetCreature()->SetActivationLevelsDirectly(MakeSineActivations(&env, t));
            env.SetPhase(step / ratio);
        }
        env.Step();

        const double sim_t = (step + 1) / (double)sim_hz;
        if (sim_t < t_start || sim_t > t_end)
            continue;

        const Eigen::VectorXd& x = env.GetSoftWorld()->GetPositions();
        const Eigen::Vector3d com = ComputeCom(x);
        const Eigen::Vector3d ub = ComputeUnbiasedTranslation(env.GetCreature(), x);
        out << sim_t << ','
            << com.x() << ',' << com.y() << ',' << com.z() << ','
            << ub.x() << ',' << ub.y() << ',' << ub.z() << '\n';
        ++window_samples;
    }

    out.close();
    std::cout << "RESULT mode=" << mode
              << " status=ok samples=" << window_samples
              << " csv=" << out_csv << std::endl;
    return true;
}

int main(int argc, char** argv)
{
    const std::string out_dir = ArgString(argc, argv, "--out-dir", "");
    if (!out_dir.empty() || ArgFlag(argc, argv, "--run-both"))
    {
        const std::string dir = out_dir.empty() ? "build/fsi_disp_compare" : out_dir;
        const std::string both_csv = dir + "/both.csv";
        const std::string hydro_csv = dir + "/hydro.csv";
        const bool ok1 = RunOneMode(argc, argv, "both", both_csv);
        const bool ok2 = RunOneMode(argc, argv, "hydro", hydro_csv);
        return (ok1 && ok2) ? 0 : 1;
    }

    const std::string mode = ArgString(argc, argv, "--mode", "both");
    const std::string out_csv = ArgString(argc, argv, "--out", "unbiased_disp.csv");
    return RunOneMode(argc, argv, mode, out_csv) ? 0 : 1;
}
