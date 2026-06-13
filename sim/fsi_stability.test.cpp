/**
 * FSI 稳定性 / 对比实验：sin 肌肉模式，输出体积、力、耗时
 * 用法:
 *   ./fsi_stability --seconds 3 --mode contact|hydro|both [--hydro F] ...
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

static float ArgFloat(int argc, char** argv, const char* key, float def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == key)
            return std::atof(argv[i + 1]);
    return def;
}

static std::string ArgString(int argc, char** argv, const char* key, const std::string& def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == key)
            return argv[i + 1];
    return def;
}

struct RunSummary {
    std::string mode;
    double seconds = 0.0;
    float hydro_scale = 0.0f;
    bool contact_on = true;
    bool hydro_on = true;
    VolumeStats final_stats{};
    double avg_step_ms = 0.0;
    double avg_sph_ms = 0.0;
    double avg_fem_ms = 0.0;
    double contact_total_avg_n = 0.0;
    double hydro_total_avg_n = 0.0;
    double contact_peak_n = 0.0;
    double hydro_peak_n = 0.0;
    int fsi_windows = 0;
};

static RunSummary RunExperiment(int argc, char** argv)
{
    const double seconds = ArgFloat(argc, argv, "--seconds", 3.0f);
    const std::string mode = ArgString(argc, argv, "--mode", "both");

    Environment env("data/fish.meta", false);
    env.SetVolumeLogIntervalSteps(0);  // 由本测试自行采集 FSI 统计

    PeridynoBridge* bridge = env.GetFluidBridge();
    RunSummary summary;
    summary.mode = mode;
    summary.seconds = seconds;

    if (!bridge || !bridge->IsInitialized())
    {
        std::cerr << "RESULT status=fail reason=no_fluid_bridge mode=" << mode << std::endl;
        return summary;
    }

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

    if (mode == "contact")
    {
        bridge->SetEnableContactRepulsion(true);
        bridge->SetEnableHydroPressure(false);
        hydro = bridge->GetHydroForceScale();
    }
    else if (mode == "hydro")
    {
        bridge->SetEnableContactRepulsion(false);
        bridge->SetEnableHydroPressure(true);
        hydro = 1.0f;
    }
    else if (mode == "both")
    {
        bridge->SetEnableContactRepulsion(true);
        bridge->SetEnableHydroPressure(true);
        hydro = 0.035f;
    }
    else
    {
        std::cerr << "RESULT status=fail reason=bad_mode mode=" << mode << std::endl;
        return summary;
    }

    bridge->SetSubsteps(substeps);
    bridge->SetContactParams(stiffness, damping);
    bridge->SetFlowRampTime(ramp);
    bridge->SetHydroForceScale(hydro);
    bridge->SetMaxVertexForce(max_force);

    summary.hydro_scale = hydro;
    summary.contact_on = bridge->GetEnableContactRepulsion();
    summary.hydro_on = bridge->GetEnableHydroPressure();

    const int sim_hz = (int)env.GetSimulationHz();
    const int ctrl_hz = (int)env.GetControlHz();
    const int ratio = sim_hz / ctrl_hz;
    const int total_steps = (int)(seconds * sim_hz);

    std::cout << "=== FSI experiment mode=" << mode
              << " seconds=" << seconds
              << " contact=" << (summary.contact_on ? 1 : 0)
              << " hydro=" << (summary.hydro_on ? 1 : 0)
              << " hydro_scale=" << hydro
              << " substeps=" << substeps
              << " stiffness=" << stiffness
              << " damping=" << damping
              << " ramp=" << ramp
              << " max_force=" << max_force
              << " ===" << std::endl;

    double sum_step_ms = 0.0;
    double sum_sph_ms = 0.0;
    double sum_fem_ms = 0.0;
    double sum_contact_total = 0.0;
    double sum_hydro_total = 0.0;
    double max_contact_peak = 0.0;
    double max_hydro_peak = 0.0;
    int fsi_windows = 0;

    // 记录质心轨迹（每整数秒采样）
    std::vector<Eigen::Vector3d> com_trajectory;
    {
        const auto& x = env.GetSoftWorld()->GetPositions();
        Eigen::Vector3d com0 = Eigen::Vector3d::Zero();
        int nv = (int)(x.size() / 3);
        for (int i = 0; i < nv; ++i)
            com0 += Eigen::Vector3d(x(3*i), x(3*i+1), x(3*i+2));
        com0 /= nv;
        com_trajectory.push_back(com0);
        printf("[COM t=0.0s] x=%.6f y=%.6f z=%.6f\n", com0.x(), com0.y(), com0.z());
    }

    for (int step = 0; step < total_steps; ++step)
    {
        if (step % ratio == 0)
        {
            const double t = step / (double)sim_hz;
            env.GetCreature()->SetActivationLevelsDirectly(MakeSineActivations(&env, t));
            env.SetPhase(step / ratio);
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        env.Step();
        const auto t1 = std::chrono::high_resolution_clock::now();
        sum_step_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 每整仿真秒采集：FSI 力统计 + COM + 体积诊断 + t=2s 顶点力快照
        if ((step + 1) % sim_hz == 0)
        {
            const double sim_t = (step + 1) / (double)sim_hz;

            // t=2s 时刻：保存逐顶点受力快照到 CSV
            if (fabs(sim_t - 2.0) < 1e-6) {
                const auto& fv = bridge->GetLastFsiVertexForceSnapshot();
                const auto& pos = env.GetSoftWorld()->GetPositions();
                int nv = (int)(fv.total.size() / 3);
                std::ofstream csv("vertex_forces_t2s.csv");
                csv << "vid,pos_x,pos_y,pos_z,contact_fx,contact_fy,contact_fz,"
                    << "hydro_fx,hydro_fy,hydro_fz,total_fx,total_fy,total_fz,f_mag\n";
                for (int i = 0; i < nv; ++i) {
                    double cx = fv.contact(3*i), cy = fv.contact(3*i+1), cz = fv.contact(3*i+2);
                    double hx = fv.hydro(3*i),   hy = fv.hydro(3*i+1),   hz = fv.hydro(3*i+2);
                    double fm = sqrt((cx+hx)*(cx+hx) + (cy+hy)*(cy+hy) + (cz+hz)*(cz+hz));
                    csv << i << ","
                        << pos(3*i) << "," << pos(3*i+1) << "," << pos(3*i+2) << ","
                        << cx << "," << cy << "," << cz << ","
                        << hx << "," << hy << "," << hz << ","
                        << cx+hx << "," << cy+hy << "," << cz+hz << ","
                        << fm << "\n";
                }
                csv.close();
                printf("[DUMP] Wrote %d vertex forces to vertex_forces_t2s.csv\n", nv);
            }

            // COM
            {
                const auto& x = env.GetSoftWorld()->GetPositions();
                Eigen::Vector3d com = Eigen::Vector3d::Zero();
                int nv = (int)(x.size() / 3);
                for (int i = 0; i < nv; ++i)
                    com += Eigen::Vector3d(x(3*i), x(3*i+1), x(3*i+2));
                com /= nv;
                com_trajectory.push_back(com);
                Eigen::Vector3d disp = com - com_trajectory[0];
                printf("[COM t=%.1fs] x=%.6f y=%.6f z=%.6f disp_xyz=[%.6f, %.6f, %.6f] disp_mag=%.6f\n",
                       sim_t, com.x(), com.y(), com.z(),
                       disp.x(), disp.y(), disp.z(), disp.norm());
            }

            // FSI force
            const FsiForceDiagnostics fsi = bridge->ConsumeSecondFsiForceDiagnostics();
            if (fsi.frames > 0)
            {
                sum_contact_total += fsi.contact_total_avg_n;
                sum_hydro_total += fsi.hydro_total_avg_n;
                max_contact_peak = std::max(max_contact_peak, fsi.contact_peak_n);
                max_hydro_peak = std::max(max_hydro_peak, fsi.hydro_peak_n);
                ++fsi_windows;

                std::cout << "[t=" << sim_t << "s] contact_avg=" << fsi.contact_total_avg_n
                          << " contact_peak=" << fsi.contact_peak_n
                          << " hydro_avg=" << fsi.hydro_total_avg_n
                          << " hydro_peak=" << fsi.hydro_peak_n
                          << std::endl;
            }

            // 体积诊断（每仿真秒）
            VolumeStats vs = env.GetCreature()->ComputeVolumeStats(env.GetSoftWorld()->GetPositions());
            printf("[VOL t=%.1fs] ratio=%.6f inverted=%d\n", sim_t, vs.volume_ratio, vs.inverted_tets);
        }
    }

    summary.final_stats = env.GetCreature()->ComputeVolumeStats(env.GetSoftWorld()->GetPositions());
    summary.avg_step_ms = sum_step_ms / std::max(total_steps, 1);
    summary.contact_total_avg_n = (fsi_windows > 0) ? sum_contact_total / fsi_windows : 0.0;
    summary.hydro_total_avg_n = (fsi_windows > 0) ? sum_hydro_total / fsi_windows : 0.0;
    summary.contact_peak_n = max_contact_peak;
    summary.hydro_peak_n = max_hydro_peak;
    summary.fsi_windows = fsi_windows;

    const double vol_shrink_pct = (1.0 - summary.final_stats.volume_ratio) * 100.0;
    Eigen::Vector3d final_com_disp = com_trajectory.back() - com_trajectory.front();
    std::cout << "RESULT mode=" << mode
              << " status=ok"
              << " seconds=" << seconds
              << " contact_on=" << (summary.contact_on ? 1 : 0)
              << " hydro_on=" << (summary.hydro_on ? 1 : 0)
              << " hydro_scale=" << hydro
              << " volume_ratio=" << summary.final_stats.volume_ratio
              << " vol_shrink_pct=" << vol_shrink_pct
              << " inverted=" << summary.final_stats.inverted_tets
              << " min_tet=" << summary.final_stats.min_tet_ratio
              << " max_tet=" << summary.final_stats.max_tet_ratio
              << " contact_total_avg_n=" << summary.contact_total_avg_n
              << " contact_peak_n=" << summary.contact_peak_n
              << " hydro_total_avg_n=" << summary.hydro_total_avg_n
              << " hydro_peak_n=" << summary.hydro_peak_n
              << " avg_step_ms=" << summary.avg_step_ms
              << " com_dispx=" << final_com_disp.x()
              << " com_dispy=" << final_com_disp.y()
              << " com_dispz=" << final_com_disp.z()
              << " com_disp_mag=" << final_com_disp.norm()
              << std::endl;
    return summary;
}

int main(int argc, char** argv)
{
    RunExperiment(argc, argv);
    return 0;
}
