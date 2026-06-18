/**
 * Trace a small set of near-fish VAP-band fluid particles for gap diagnosis.
 *
 * Outputs:
 *   <out-dir>/tracked_particles.csv  per tracked particle, per FEM step
 *   <out-dir>/frame_summary.csv      per FEM step stability/coverage summary
 *   <out-dir>/selected_particles.txt chosen fluid ids
 */
#include "Environment.h"
#include "Worm.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

static const double kPi = 3.14159265358979323846;

static void EnsureDir(const std::string& dir)
{
    std::string path;
    for (char ch : dir) {
        path.push_back(ch);
        if (ch == '/')
            mkdir(path.c_str(), 0755);
    }
    if (!path.empty())
        mkdir(path.c_str(), 0755);
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
        const int sampleCount = muscles[muscleIdx]->GetNumSampling();
        const double musclePhase = (muscleIdx % 2 == 0) ? 0.0 : kPi;
        for (int sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
            const double bodyPhase = 2.0 * kPi * spatialCycles * sampleIdx / sampleCount;
            const double signal = offset + amplitude * std::sin(
                2.0 * kPi * temporalFrequency * time_sec + waveSign * bodyPhase + musclePhase);
            activations[idx + sampleIdx] = std::min(std::max(signal, 0.0), 1.0);
        }
        idx += sampleCount;
    }
    return activations;
}

static Eigen::Vector3d ComputeCom(const Eigen::VectorXd& positions)
{
    const int n = (int)(positions.size() / 3);
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    for (int i = 0; i < n; ++i)
        com += positions.segment<3>(3 * i);
    return com / std::max(n, 1);
}

static Eigen::Vector3d ComputeUnbiasedTranslation(Worm* worm, const Eigen::VectorXd& x)
{
    return worm->EvalBodyTransform(worm->GetVerticesReference(), x).translation();
}

static int CountZeroHydroSurface(
    const FsiVertexForceSnapshot& snap,
    const std::set<int>& surface_vertices)
{
    int cnt = 0;
    const int nv = (int)(snap.hydro.size() / 3);
    for (int i = 0; i < nv; ++i) {
        if (!surface_vertices.count(i)) continue;
        const double hx = snap.hydro(3 * i);
        const double hy = snap.hydro(3 * i + 1);
        const double hz = snap.hydro(3 * i + 2);
        if (hx * hx + hy * hy + hz * hz < 1e-24)
            ++cnt;
    }
    return cnt;
}

int main(int argc, char** argv)
{
    const std::string meta_path = ArgString(argc, argv, "--meta", "data/fish.meta");
    const std::string out_dir = ArgString(argc, argv, "--out-dir", "build/vap_particle_trace_still_0p5s");
    const float seconds = ArgFloat(argc, argv, "--seconds", 0.5f);
    const int sim_hz_arg = ArgInt(argc, argv, "--sim-hz", 960);
    const int substeps = ArgInt(argc, argv, "--substeps", 18);
    const float ramp_sec = ArgFloat(argc, argv, "--ramp", 0.1f);
    const float max_force = ArgFloat(argc, argv, "--max-force", 5.0f);
    const float hydro_scale = ArgFloat(argc, argv, "--hydro-scale", 1.0f);
    const int trace_count = ArgInt(argc, argv, "--trace-count", 8);
    const float act_amp = ArgFloat(argc, argv, "--act-amp", 0.25f);
    const float act_offset = ArgFloat(argc, argv, "--act-offset", 0.25f);
    const float act_freq = ArgFloat(argc, argv, "--act-freq", 1.0f);
    const float act_cycles = ArgFloat(argc, argv, "--act-cycles", 1.0f);
    const float wave_sign = ArgFloat(argc, argv, "--wave-sign", -1.0f);

    float flow_vx = 0.0f, flow_vy = 0.0f, flow_vz = 0.0f;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 < argc && std::string(argv[i]) == "--flow") {
            if (!ParseFlowTriple(argv[++i], flow_vx, flow_vy, flow_vz)) {
                std::cerr << "bad --flow, expected VX,VY,VZ\n";
                return 1;
            }
        }
    }

    EnsureDir(out_dir);
    const std::string trace_csv = out_dir + "/tracked_particles.csv";
    const std::string frame_csv = out_dir + "/frame_summary.csv";
    const std::string selected_path = out_dir + "/selected_particles.txt";

    Environment env(meta_path, false);
    env.SetSimulationHz(sim_hz_arg);
    env.SetVolumeLogIntervalSteps(0);

    PeridynoBridge* bridge = env.GetFluidBridge();
    if (!bridge || !bridge->IsInitialized()) {
        std::cerr << "RESULT status=fail reason=no_fluid_bridge\n";
        return 1;
    }

    bridge->SetFlowVelocity(Eigen::Vector3d(flow_vx, flow_vy, flow_vz));
    bridge->SetSubsteps(substeps);
    bridge->SetContactParams(0.3f, 15.0f);
    bridge->SetFlowRampTime(ramp_sec);
    bridge->SetHydroForceScale(hydro_scale);
    bridge->SetMaxVertexForce(max_force);
    bridge->SetEnableContactRepulsion(false);
    bridge->SetEnableHydroPressure(true);
    bridge->SetGhostSampleStride(1);
    bridge->SetVapGhostCouplingScales(0.25f, 0.25f);

    const int sim_hz = (int)env.GetSimulationHz();
    const int total_frames = (int)(seconds * sim_hz + 0.5f);
    const int ctrl_hz = (int)env.GetControlHz();
    const int ratio = sim_hz / ctrl_hz;

    std::set<int> surface_vertices;
    for (const auto& tri : env.GetCreature()->GetContours()) {
        surface_vertices.insert(tri[0]);
        surface_vertices.insert(tri[1]);
        surface_vertices.insert(tri[2]);
    }

    std::ofstream frame_out(frame_csv);
    frame_out << std::fixed << std::setprecision(9);
    frame_out << "step,sim_t,step_ms,volume_ratio,inverted,com_x,com_y,com_z,"
                 "com_disp_x,com_disp_y,com_disp_z,ub_disp_x,ub_disp_y,ub_disp_z,"
                 "hydro_zero_surface,hydro_nonzero_pct,tracked_count,tracked_in_band\n";

    std::vector<int> tracked_ids;
    const Eigen::Vector3d com0 = ComputeCom(env.GetSoftWorld()->GetPositions());
    const Eigen::Vector3d ub0 = ComputeUnbiasedTranslation(env.GetCreature(), env.GetSoftWorld()->GetPositions());

    std::cout << "=== vap_particle_trace"
              << " meta=" << meta_path
              << " frames=" << total_frames
              << " flow=(" << flow_vx << "," << flow_vy << "," << flow_vz << ")"
              << " out_dir=" << out_dir
              << " trace_count=" << trace_count
              << " ===\n";

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
        const int frame = step + 1;
        const double sim_t = frame / (double)sim_hz;

        if (tracked_ids.empty()) {
            tracked_ids = bridge->SelectVapBandTraceParticles(trace_count);
            std::ofstream sel(selected_path);
            sel << "trace_slot,fluid_id\n";
            for (int i = 0; i < (int)tracked_ids.size(); ++i)
                sel << i << ',' << tracked_ids[i] << '\n';
            bridge->ExportVapTraceParticlesCsv(trace_csv, tracked_ids, frame, sim_t, false);
        } else {
            bridge->ExportVapTraceParticlesCsv(trace_csv, tracked_ids, frame, sim_t, true);
        }

        int tracked_in_band = 0;
        {
            const std::string tmp_path = out_dir + "/_last_trace_tmp.csv";
            bridge->ExportVapTraceParticlesCsv(tmp_path, tracked_ids, frame, sim_t, false);
            std::ifstream in(tmp_path);
            std::string line;
            std::getline(in, line);
            while (std::getline(in, line)) {
                size_t pos = 0;
                int comma_count = 0;
                int in_band = 0;
                while (comma_count < 4 && pos != std::string::npos) {
                    pos = line.find(',', pos + (comma_count == 0 ? 0 : 1));
                    ++comma_count;
                }
                if (pos != std::string::npos)
                    in_band = std::atoi(line.substr(pos + 1).c_str());
                tracked_in_band += in_band ? 1 : 0;
            }
        }

        const Eigen::VectorXd& x = env.GetSoftWorld()->GetPositions();
        const VolumeStats vol = env.GetCreature()->ComputeVolumeStats(x);
        const Eigen::Vector3d com = ComputeCom(x);
        const Eigen::Vector3d disp = com - com0;
        const Eigen::Vector3d ub = ComputeUnbiasedTranslation(env.GetCreature(), x);
        const Eigen::Vector3d ub_disp = ub - ub0;
        const int zero_surf = CountZeroHydroSurface(bridge->GetLastFsiVertexForceSnapshot(), surface_vertices);
        const int surf_n = (int)surface_vertices.size();
        const double nonzero_pct = 100.0 * (surf_n - zero_surf) / std::max(surf_n, 1);

        frame_out << frame << ',' << sim_t << ',' << step_ms << ','
                  << vol.volume_ratio << ',' << vol.inverted_tets << ','
                  << com.x() << ',' << com.y() << ',' << com.z() << ','
                  << disp.x() << ',' << disp.y() << ',' << disp.z() << ','
                  << ub_disp.x() << ',' << ub_disp.y() << ',' << ub_disp.z() << ','
                  << zero_surf << ',' << nonzero_pct << ','
                  << tracked_ids.size() << ',' << tracked_in_band << '\n';

        if (frame % 120 == 0 || frame == total_frames) {
            std::cout << "[trace step " << frame << " t=" << sim_t
                      << "] ub_z=" << ub_disp.z()
                      << " vol=" << vol.volume_ratio
                      << " inv=" << vol.inverted_tets
                      << " hydro_nonzero_pct=" << nonzero_pct
                      << " tracked_in_band=" << tracked_in_band << "/" << tracked_ids.size()
                      << "\n";
        }
    }

    std::cout << "TRACE_OUT particles=" << trace_csv
              << " frames=" << frame_csv
              << " selected=" << selected_path << "\n";
    std::cout << "RESULT status=ok frames=" << total_frames
              << " tracked=" << tracked_ids.size() << "\n";
    return 0;
}
