/**
 * 30 帧水压力 / 稳定性 / 耗时 / 质心监控（headless）
 * 用法: ./hydro_30frame [--frames N] [--ramp SEC]
 */
#include "Environment.h"
#include "Worm.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
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
    for (int muscleIdx = 0; muscleIdx < (int)muscles.size(); ++muscleIdx) {
        int sampleCount = muscles[muscleIdx]->GetNumSampling();
        double musclePhase = (muscleIdx % 2 == 0) ? 0.0 : g_pi;
        for (int sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
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

static Eigen::Vector3d ComputeCom(const Eigen::VectorXd& positions)
{
    const int n = (int)(positions.size() / 3);
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    for (int i = 0; i < n; ++i)
        com += positions.segment<3>(3 * i);
    return com / std::max(n, 1);
}

static int CountZeroHydroSurface(
    const FsiVertexForceSnapshot& snap,
    const std::set<int>& surface_vertices)
{
    int cnt = 0;
    const int nv = (int)(snap.hydro.size() / 3);
    for (int i = 0; i < nv; ++i) {
        if (!surface_vertices.count(i)) continue;
        const double hm = std::sqrt(
            snap.hydro(3 * i) * snap.hydro(3 * i) +
            snap.hydro(3 * i + 1) * snap.hydro(3 * i + 1) +
            snap.hydro(3 * i + 2) * snap.hydro(3 * i + 2));
        if (hm < 1e-12) cnt++;
    }
    return cnt;
}

int main(int argc, char** argv)
{
    const int frames = ArgInt(argc, argv, "--frames", 0);
    const float seconds = ArgFloat(argc, argv, "--seconds", 0.125f);
    const int sim_hz_arg = ArgInt(argc, argv, "--sim-hz", 960);
    const float ramp_sec = ArgFloat(argc, argv, "--ramp", 0.01f);
    const float max_force = ArgFloat(argc, argv, "--max-force", 3.0f);
    const int log_every = ArgInt(argc, argv, "--log-every", 0);  // 0=自动

    Environment env("data/fish.meta", false);
    env.SetSimulationHz(sim_hz_arg);
    env.SetVolumeLogIntervalSteps(0);

    const int sim_hz = (int)env.GetSimulationHz();
    const int total_frames = (frames > 0) ? frames : (int)(seconds * sim_hz);

    PeridynoBridge* bridge = env.GetFluidBridge();
    if (!bridge || !bridge->IsInitialized()) {
        std::cerr << "RESULT status=fail reason=no_fluid_bridge\n";
        return 1;
    }

    bridge->SetSubsteps(24);
    bridge->SetContactParams(0.3f, 15.0f);
    bridge->SetFlowRampTime(ramp_sec);
    bridge->SetHydroForceScale(1.0f);
    bridge->SetMaxVertexForce(max_force);
    bridge->SetEnableContactRepulsion(false);
    bridge->SetEnableHydroPressure(true);

    const int ctrl_hz = (int)env.GetControlHz();
    const int ratio = sim_hz / ctrl_hz;

    std::set<int> surface_vertices;
    for (const auto& tri : env.GetCreature()->GetContours()) {
        surface_vertices.insert(tri[0]);
        surface_vertices.insert(tri[1]);
        surface_vertices.insert(tri[2]);
    }

    const int log_stride = (log_every > 0) ? log_every : std::max(sim_hz / 4, 1);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== hydro_30frame sim_hz=" << sim_hz
              << " frames=" << total_frames
              << " t_sec=" << (total_frames / (double)sim_hz)
              << " ramp_sec=" << ramp_sec
              << " max_force=" << max_force
              << " hydro_scale=" << bridge->GetHydroForceScale()
              << " ===\n";

    Eigen::Vector3d com0 = ComputeCom(env.GetSoftWorld()->GetPositions());
    double sum_step_ms = 0.0;
    VolumeStats last_vol{};
    int last_zero_surface = 0;

    for (int step = 0; step < total_frames; ++step) {
        if (step % ratio == 0) {
            const double t = step / (double)sim_hz;
            env.GetCreature()->SetActivationLevelsDirectly(MakeSineActivations(&env, t));
            env.SetPhase(step / ratio);
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        env.Step();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum_step_ms += step_ms;

        const double sim_t = (step + 1) / (double)sim_hz;
        const auto& x = env.GetSoftWorld()->GetPositions();
        last_vol = env.GetCreature()->ComputeVolumeStats(x);
        const Eigen::Vector3d com = ComputeCom(x);
        const Eigen::Vector3d disp = com - com0;
        last_zero_surface = CountZeroHydroSurface(bridge->GetLastFsiVertexForceSnapshot(), surface_vertices);

        if ((step + 1) % log_stride == 0 || step + 1 == total_frames) {
            std::cout << "[step " << (step + 1) << " t=" << sim_t << "s]"
                      << " step_ms=" << step_ms
                      << " vol=" << last_vol.volume_ratio
                      << " inv=" << last_vol.inverted_tets
                      << " com=(" << com.x() << "," << com.y() << "," << com.z() << ")"
                      << " disp=(" << disp.x() << "," << disp.y() << "," << disp.z() << ")"
                      << " disp_z=" << disp.z()
                      << " hydro_zero_surf=" << last_zero_surface << "\n";
        }
    }

    const FsiForceDiagnostics fsi = bridge->ConsumeSecondFsiForceDiagnostics();
    const Eigen::Vector3d com_final = ComputeCom(env.GetSoftWorld()->GetPositions());
    const Eigen::Vector3d disp_final = com_final - com0;

    std::cout << "SUMMARY sim_hz=" << sim_hz
              << " max_force=" << max_force
              << " frames=" << total_frames
              << " t_sec=" << (total_frames / (double)sim_hz)
              << " avg_step_ms=" << (sum_step_ms / std::max(total_frames, 1))
              << " final_vol=" << last_vol.volume_ratio
              << " vol_shrink_pct=" << ((1.0 - last_vol.volume_ratio) * 100.0)
              << " final_inverted=" << last_vol.inverted_tets
              << " com_disp_z=" << disp_final.z()
              << " com_disp_mag=" << disp_final.norm()
              << " hydro_surface_zero=" << last_zero_surface
              << " roll_hydro_avg=" << fsi.hydro_total_avg_n
              << " roll_hydro_peak=" << fsi.hydro_peak_n
              << "\n";
    std::cout << "RESULT status=ok"
              << " sim_hz=" << sim_hz
              << " max_force=" << max_force
              << " frames=" << total_frames
              << " volume_ratio=" << last_vol.volume_ratio
              << " inverted=" << last_vol.inverted_tets
              << " com_disp_z=" << disp_final.z()
              << " com_disp_mag=" << disp_final.norm()
              << "\n";
    return 0;
}
