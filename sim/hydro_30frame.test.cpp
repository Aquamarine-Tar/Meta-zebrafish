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

static Eigen::VectorXd MakeSineActivations(
    Environment* env,
    double time_sec,
    double amplitude,
    double offset,
    double temporalFrequency,
    double spatialCycles,
    double waveSign)
{
    const auto& muscles = env->GetCreature()->GetMuscles();
    Eigen::VectorXd activations = Eigen::VectorXd::Zero(env->GetMuscleActivationSize());

    int idx = 0;
    for (int muscleIdx = 0; muscleIdx < (int)muscles.size(); ++muscleIdx) {
        int sampleCount = muscles[muscleIdx]->GetNumSampling();
        double musclePhase = (muscleIdx % 2 == 0) ? 0.0 : g_pi;
        for (int sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
            double bodyPhase = 2.0 * g_pi * spatialCycles * sampleIdx / sampleCount;
            double signal = offset + amplitude * std::sin(
                2.0 * g_pi * temporalFrequency * time_sec + waveSign * bodyPhase + musclePhase);
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

static std::string ArgString(int argc, char** argv, const char* key, const std::string& def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == key)
            return std::string(argv[i + 1]);
    return def;
}

static bool ParseFlowTriple(const std::string& text, float& vx, float& vy, float& vz)
{
    const size_t c1 = text.find(',');
    const size_t c2 = (c1 == std::string::npos) ? std::string::npos : text.find(',', c1 + 1);
    if (c1 == std::string::npos || c2 == std::string::npos)
        return false;
    vx = std::atof(text.substr(0, c1).c_str());
    vy = std::atof(text.substr(c1 + 1, c2 - c1 - 1).c_str());
    vz = std::atof(text.substr(c2 + 1).c_str());
    return true;
}

static Eigen::Vector3d ComputeCom(const Eigen::VectorXd& positions)
{
    const int n = (int)(positions.size() / 3);
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    for (int i = 0; i < n; ++i)
        com += positions.segment<3>(3 * i);
    return com / std::max(n, 1);
}

/** 无偏位移：参考构形 → 当前构形的刚体配准平移（同可视化绿色轨迹） */
static Eigen::Vector3d ComputeUnbiasedTranslation(Worm* worm, const Eigen::VectorXd& x)
{
    const Eigen::VectorXd& ref = worm->GetVerticesReference();
    return worm->EvalBodyTransform(ref, x).translation();
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
    const float ramp_sec = ArgFloat(argc, argv, "--ramp", 0.1f);
    const float max_force = ArgFloat(argc, argv, "--max-force", 3.0f);
    const float hydro_scale = ArgFloat(argc, argv, "--hydro-scale", 1.0f);
    const int substeps = ArgInt(argc, argv, "--substeps", 24);
    const int ghost_stride = ArgInt(argc, argv, "--ghost-stride", 1);
    const int log_every = ArgInt(argc, argv, "--log-every", 0);  // 0=自动
    const float diag_at = ArgFloat(argc, argv, "--diag-at", -1.0f);
    const std::string diag_out = ArgString(argc, argv, "--diag-out", "vap_band_diag_t3/vap_band");
    const std::string meta_path = ArgString(argc, argv, "--meta", "data/fish.meta");
    const float act_amp = ArgFloat(argc, argv, "--act-amp", 0.25f);
    const float act_offset = ArgFloat(argc, argv, "--act-offset", 0.25f);
    const float act_freq = ArgFloat(argc, argv, "--act-freq", 1.0f);
    const float act_cycles = ArgFloat(argc, argv, "--act-cycles", 1.0f);
    const float wave_sign = ArgFloat(argc, argv, "--wave-sign", -1.0f);
    const int diag_step = (diag_at > 0.0f) ? (int)(diag_at * sim_hz_arg + 0.5f) : -1;
    bool flow_override = false;
    float flow_vx = 0.0f, flow_vy = 0.0f, flow_vz = -0.05f;
    bool vap_full_domain = false;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--vap-full-domain") {
            vap_full_domain = true;
        } else if (i + 1 < argc && k == "--flow") {
            if (!ParseFlowTriple(argv[++i], flow_vx, flow_vy, flow_vz)) {
                std::cerr << "bad --flow, expected VX,VY,VZ\n";
                return 1;
            }
            flow_override = true;
        }
    }

    Environment env(meta_path, false);
    env.SetSimulationHz(sim_hz_arg);
    env.SetVolumeLogIntervalSteps(0);

    const int sim_hz = (int)env.GetSimulationHz();
    const int total_frames = (frames > 0) ? frames : (int)(seconds * sim_hz);

    PeridynoBridge* bridge = env.GetFluidBridge();
    if (!bridge || !bridge->IsInitialized()) {
        std::cerr << "RESULT status=fail reason=no_fluid_bridge\n";
        return 1;
    }

    if (flow_override)
        bridge->SetFlowVelocity(Eigen::Vector3d(flow_vx, flow_vy, flow_vz));
    bridge->SetSubsteps(substeps);
    bridge->SetContactParams(0.3f, 15.0f);
    bridge->SetFlowRampTime(ramp_sec);
    bridge->SetHydroForceScale(hydro_scale);
    bridge->SetMaxVertexForce(max_force);
    bridge->SetEnableContactRepulsion(false);
    bridge->SetEnableHydroPressure(true);
    bridge->SetGhostSampleStride(ghost_stride);
    const float ghost_div_scale = ArgFloat(argc, argv, "--ghost-div-scale", 0.25f);
    const float ghost_vel_scale = ArgFloat(argc, argv, "--ghost-vel-scale", 0.25f);
    bridge->SetVapGhostCouplingScales(ghost_div_scale, ghost_vel_scale);
    bridge->SetVapFullDomain(vap_full_domain);

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
              << " meta=" << meta_path
              << " frames=" << total_frames
              << " t_sec=" << (total_frames / (double)sim_hz)
              << " ramp_sec=" << ramp_sec
              << " max_force=" << max_force
              << " substeps=" << substeps
              << " ghost_stride=" << ghost_stride
              << " dx=" << bridge->GetParticleSpacing()
              << " hydro_scale=" << bridge->GetHydroForceScale()
              << " ghost_div_scale=" << ghost_div_scale
              << " ghost_vel_scale=" << ghost_vel_scale
              << " act_amp=" << act_amp
              << " act_offset=" << act_offset
              << " act_freq=" << act_freq
              << " act_cycles=" << act_cycles
              << " wave_sign=" << wave_sign
              << " flow=(" << bridge->GetFlowVelocity().x()
              << "," << bridge->GetFlowVelocity().y()
              << "," << bridge->GetFlowVelocity().z() << ")"
              << " vap_full_domain=" << (vap_full_domain ? 1 : 0)
              << " ===\n";

    Eigen::Vector3d com0 = ComputeCom(env.GetSoftWorld()->GetPositions());
    const Eigen::Vector3d ub0 = ComputeUnbiasedTranslation(env.GetCreature(), env.GetSoftWorld()->GetPositions());
    double sum_step_ms = 0.0;
    VolumeStats last_vol{};
    int last_zero_surface = 0;
    VapBandDiagnosticsSummary band_diag{};

    for (int step = 0; step < total_frames; ++step) {
        if (step % ratio == 0) {
            const double t = step / (double)sim_hz;
            env.GetCreature()->SetActivationLevelsDirectly(
                MakeSineActivations(&env, t, act_amp, act_offset, act_freq, act_cycles, wave_sign));
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
        const Eigen::Vector3d ub = ComputeUnbiasedTranslation(env.GetCreature(), x);
        const Eigen::Vector3d ub_disp = ub - ub0;
        last_zero_surface = CountZeroHydroSurface(bridge->GetLastFsiVertexForceSnapshot(), surface_vertices);

        if (diag_step > 0 && step + 1 == diag_step) {
            band_diag = bridge->ExportVapBandDiagnostics(diag_out);
            std::cout << "[band_diag t=" << sim_t << "s]"
                      << " n_band=" << band_diag.n_band_fluid
                      << " n_mask=" << band_diag.n_band_mask_fluid
                      << " avg_ghost/fluid_nbr_ratio=" << band_diag.avg_ghost_fluid_neighbor_ratio
                      << " avg_surf_nearest_fluid_dist=" << band_diag.avg_surface_nearest_fluid_dist
                      << " avg_fluid_inf=" << band_diag.avg_fluid_influence
                      << " avg_ghost_inf=" << band_diag.avg_ghost_influence
                      << " ghost_inf_frac=" << band_diag.ghost_influence_fraction
                      << " out=" << diag_out << "\n";
        }

        if ((step + 1) % log_stride == 0 || step + 1 == total_frames) {
            std::cout << "[step " << (step + 1) << " t=" << sim_t << "s]"
                      << " step_ms=" << step_ms
                      << " vol=" << last_vol.volume_ratio
                      << " inv=" << last_vol.inverted_tets
                      << " com=(" << com.x() << "," << com.y() << "," << com.z() << ")"
                      << " disp=(" << disp.x() << "," << disp.y() << "," << disp.z() << ")"
                      << " ub_disp=(" << ub_disp.x() << "," << ub_disp.y() << "," << ub_disp.z() << ")"
                      << " disp_z=" << disp.z()
                      << " ub_disp_z=" << ub_disp.z()
                      << " hydro_zero_surf=" << last_zero_surface << "\n";
        }
    }

    const FsiForceDiagnostics fsi = bridge->ConsumeSecondFsiForceDiagnostics();
    const Eigen::Vector3d com_final = ComputeCom(env.GetSoftWorld()->GetPositions());
    const Eigen::Vector3d disp_final = com_final - com0;
    const Eigen::Vector3d ub_final = ComputeUnbiasedTranslation(env.GetCreature(), env.GetSoftWorld()->GetPositions());
    const Eigen::Vector3d ub_disp_final = ub_final - ub0;
    const int surf_n = (int)surface_vertices.size();
    const int hydro_nonzero_surf = std::max(0, surf_n - last_zero_surface);
    const double hydro_nonzero_pct = 100.0 * hydro_nonzero_surf / std::max(surf_n, 1);

    std::cout << "SUMMARY sim_hz=" << sim_hz
              << " substeps=" << substeps
              << " dx=" << bridge->GetParticleSpacing()
              << " max_force=" << max_force
              << " frames=" << total_frames
              << " t_sec=" << (total_frames / (double)sim_hz)
              << " avg_step_ms=" << (sum_step_ms / std::max(total_frames, 1))
              << " final_vol=" << last_vol.volume_ratio
              << " vol_shrink_pct=" << ((1.0 - last_vol.volume_ratio) * 100.0)
              << " final_inverted=" << last_vol.inverted_tets
              << " com_disp=(" << disp_final.x() << "," << disp_final.y() << "," << disp_final.z() << ")"
              << " ub_disp=(" << ub_disp_final.x() << "," << ub_disp_final.y() << "," << ub_disp_final.z() << ")"
              << " com_disp_z=" << disp_final.z()
              << " ub_disp_z=" << ub_disp_final.z()
              << " com_disp_mag=" << disp_final.norm()
              << " ub_disp_mag=" << ub_disp_final.norm()
              << " hydro_surface_zero=" << last_zero_surface
              << " hydro_nonzero_surf=" << hydro_nonzero_surf
              << " hydro_nonzero_pct=" << hydro_nonzero_pct
              << " ghost_stride=" << ghost_stride
              << " ghost_count=" << bridge->GetGhostVertexCount()
              << " ghost_div_scale=" << ghost_div_scale
              << " ghost_vel_scale=" << ghost_vel_scale
              << " vap_full_domain=" << (vap_full_domain ? 1 : 0)
              << " roll_hydro_avg=" << fsi.hydro_total_avg_n
              << " roll_hydro_peak=" << fsi.hydro_peak_n
              << "\n";
    if (band_diag.valid) {
        std::cout << "BAND_DIAG n_band=" << band_diag.n_band_fluid
                  << " avg_ghost_fluid_ratio=" << band_diag.avg_ghost_fluid_neighbor_ratio
                  << " avg_surf_nearest_fluid_dist=" << band_diag.avg_surface_nearest_fluid_dist
                  << " avg_fluid_influence=" << band_diag.avg_fluid_influence
                  << " avg_ghost_influence=" << band_diag.avg_ghost_influence
                  << " fluid_influence_fraction=" << (1.0 - band_diag.ghost_influence_fraction)
                  << " ghost_influence_fraction=" << band_diag.ghost_influence_fraction
                  << " csv=" << diag_out << "_band_particles.csv"
                  << "\n";
    }
    std::cout << "RESULT status=ok"
              << " sim_hz=" << sim_hz
              << " max_force=" << max_force
              << " frames=" << total_frames
              << " volume_ratio=" << last_vol.volume_ratio
              << " inverted=" << last_vol.inverted_tets
              << " com_disp_z=" << disp_final.z()
              << " ub_disp_z=" << ub_disp_final.z()
              << " com_disp_mag=" << disp_final.norm()
              << " ub_disp_mag=" << ub_disp_final.norm()
              << " hydro_nonzero_pct=" << hydro_nonzero_pct
              << " ghost_stride=" << ghost_stride
              << " ghost_count=" << bridge->GetGhostVertexCount()
              << "\n";
    return 0;
}
