/**
 * 在指定仿真时刻导出 FSI 顶点力快照（接触 + 水压 + 合力）及绕 x 轴力矩
 * 用法:
 *   ./fsi_force_snapshot [--out DIR] [--seconds 5] [--substeps N]
 *   ./fsi_force_snapshot --torque-only --seconds 5 --t-start 2 --t-step 0.25
 */
#include "Environment.h"
#include "Worm.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

static void EnsureDir(const std::string& path)
{
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

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

static void WriteVectorCsv(std::ofstream& out, const Eigen::VectorXd& v)
{
    const int n = (int)(v.size() / 3);
    out << "vtx,x,y,z\n";
    for (int i = 0; i < n; ++i)
        out << i << ',' << v(3 * i) << ',' << v(3 * i + 1) << ',' << v(3 * i + 2) << '\n';
}

static void WriteForcesCsv(const std::string& path, const FsiVertexForceSnapshot& snap)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    const int n = (int)(snap.total.size() / 3);
    out << "vtx,cx,cy,cz,hx,hy,hz,fx,fy,fz\n";
    for (int i = 0; i < n; ++i)
    {
        out << i << ','
            << snap.contact(3 * i) << ',' << snap.contact(3 * i + 1) << ',' << snap.contact(3 * i + 2) << ','
            << snap.hydro(3 * i) << ',' << snap.hydro(3 * i + 1) << ',' << snap.hydro(3 * i + 2) << ','
            << snap.total(3 * i) << ',' << snap.total(3 * i + 1) << ',' << snap.total(3 * i + 2) << '\n';
    }
}

static void WriteFacesCsv(const std::string& path, const std::vector<Eigen::Vector3i>& faces)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "f,i0,i1,i2\n";
    for (int f = 0; f < (int)faces.size(); ++f)
        out << f << ',' << faces[f][0] << ',' << faces[f][1] << ',' << faces[f][2] << '\n';
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

struct TorqueReport {
    double tau_x_surface_com = 0.0;
    double tau_x_surface_center = 0.0;
    double tau_x_all_com = 0.0;
    Eigen::Vector3d torque_surface_com = Eigen::Vector3d::Zero();
    Eigen::Vector3d force_sum_surface = Eigen::Vector3d::Zero();
};

static TorqueReport ComputeTorques(
    const Eigen::VectorXd& positions,
    const Eigen::VectorXd& total_forces,
    const std::vector<Eigen::Vector3i>& contours,
    int center_index)
{
    TorqueReport rep;
    const Eigen::Vector3d com = ComputeCom(positions);
    const Eigen::Vector3d center = positions.segment<3>(3 * center_index);

    std::set<int> surface;
    for (const auto& tri : contours)
    {
        surface.insert(tri[0]);
        surface.insert(tri[1]);
        surface.insert(tri[2]);
    }

    auto accumulate = [&](const Eigen::Vector3d& pivot, bool surface_only, Eigen::Vector3d& torque_out, double& tau_x_out, Eigen::Vector3d* force_sum) {
        torque_out.setZero();
        if (force_sum) force_sum->setZero();
        const int n = (int)(positions.size() / 3);
        for (int i = 0; i < n; ++i)
        {
            if (surface_only && surface.count(i) == 0) continue;
            const Eigen::Vector3d r = positions.segment<3>(3 * i) - pivot;
            const Eigen::Vector3d f = total_forces.segment<3>(3 * i);
            if (force_sum) *force_sum += f;
            torque_out += r.cross(f);
        }
        tau_x_out = torque_out.x();
    };

    accumulate(com, true, rep.torque_surface_com, rep.tau_x_surface_com, &rep.force_sum_surface);

    Eigen::Vector3d t_center = Eigen::Vector3d::Zero();
    for (int i : surface)
    {
        const Eigen::Vector3d r = positions.segment<3>(3 * i) - center;
        const Eigen::Vector3d f = total_forces.segment<3>(3 * i);
        t_center += r.cross(f);
    }
    rep.tau_x_surface_center = t_center.x();

    Eigen::Vector3d t_all = Eigen::Vector3d::Zero();
    const int n = (int)(positions.size() / 3);
    for (int i = 0; i < n; ++i)
    {
        const Eigen::Vector3d r = positions.segment<3>(3 * i) - com;
        const Eigen::Vector3d f = total_forces.segment<3>(3 * i);
        t_all += r.cross(f);
    }
    rep.tau_x_all_com = t_all.x();

    return rep;
}

static void WriteMeta(const std::string& path, double sim_t, int iter, const FsiVertexForceSnapshot& snap, const TorqueReport& rep,
                      const Eigen::Vector3d& com, const Eigen::Vector3d& center)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << std::setprecision(10);
    out << "sim_time=" << sim_t << '\n';
    out << "iter=" << iter << '\n';
    out << "flow_ramp=" << snap.flow_ramp << '\n';
    out << "com=" << com.transpose() << '\n';
    out << "center=" << center.transpose() << '\n';
    out << "tau_x_surface_about_com=" << rep.tau_x_surface_com << '\n';
    out << "tau_x_surface_about_center=" << rep.tau_x_surface_center << '\n';
    out << "tau_x_all_vertices_about_com=" << rep.tau_x_all_com << '\n';
    out << "torque_surface_com=" << rep.torque_surface_com.transpose() << '\n';
    out << "force_sum_surface=" << rep.force_sum_surface.transpose() << '\n';
}

static std::vector<int> BuildSnapshotIters(int sim_hz, double t_start, double t_end, double t_step)
{
    std::vector<int> iters;
    for (double t = t_start; t <= t_end + 1e-9; t += t_step)
        iters.push_back((int)std::lround(t * sim_hz));
    return iters;
}

static bool IsSnapshotIter(int iter, const std::vector<int>& snapshot_iters)
{
    for (int target : snapshot_iters)
        if (iter == target) return true;
    return false;
}

static double IterToTime(int iter, int sim_hz)
{
    return iter / static_cast<double>(sim_hz);
}

/** t 是否为 0.5 s 的整数倍时刻（0.5, 1.0, 1.5, …） */
static bool IsHalfSecondMark(double sim_t, int sim_hz)
{
    if (sim_t < 0.5 - 1e-9)
        return false;
    const int half_step = (int)std::lround(sim_t * 2.0);
    const double target_t = half_step * 0.5;
    return std::abs(sim_t - target_t) < 0.5 / sim_hz;
}

/** 表面顶点水压力非零占比 + 质心（用于半秒监控日志） */
static void LogHalfSecondMonitor(
    double sim_t,
    int iter,
    const FsiVertexForceSnapshot& snap,
    const Eigen::VectorXd& positions,
    const std::set<int>& surface)
{
    int nonzero = 0;
    const int total = (int)surface.size();
    for (int i : surface)
    {
        const double hx = snap.hydro(3 * i);
        const double hy = snap.hydro(3 * i + 1);
        const double hz = snap.hydro(3 * i + 2);
        if (hx * hx + hy * hy + hz * hz > 1e-24)
            ++nonzero;
    }
    const Eigen::Vector3d com = ComputeCom(positions);
    const double pct = 100.0 * nonzero / std::max(total, 1);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[monitor t=" << sim_t << "s iter=" << iter << "] "
              << "surface_nonzero_hydro=" << nonzero << "/" << total
              << " (" << std::setprecision(2) << pct << std::setprecision(6) << "%)"
              << " com=(" << com.x() << "," << com.y() << "," << com.z() << ")\n";
}

int main(int argc, char** argv)
{
    std::string out_dir = "fsi_snapshots";
    double seconds = 3.0;
    double t_start = 2.0;
    double t_step = 0.25;
    float max_force = 5.0f;
    float ramp_sec = 0.1f;
    int sim_hz_arg = 960;
    int substeps_override = -1;
    int ghost_stride = 1;
    bool torque_only = false;
    bool flow_override = false;
    float flow_vx = 0.0f;
    float flow_vy = 0.0f;
    float flow_vz = -0.05f;
    for (int i = 1; i < argc; ++i)
    {
        std::string k = argv[i];
        if (k == "--torque-only") torque_only = true;
        else if (i + 1 < argc && k == "--out") { out_dir = argv[++i]; }
        else if (i + 1 < argc && k == "--seconds") { seconds = std::atof(argv[++i]); }
        else if (i + 1 < argc && k == "--t-start") { t_start = std::atof(argv[++i]); }
        else if (i + 1 < argc && k == "--t-step") { t_step = std::atof(argv[++i]); }
        else if (i + 1 < argc && k == "--max-force") { max_force = std::atof(argv[++i]); }
        else if (i + 1 < argc && k == "--ramp") { ramp_sec = std::atof(argv[++i]); }
        else if (i + 1 < argc && k == "--sim-hz") { sim_hz_arg = std::atoi(argv[++i]); }
        else if (i + 1 < argc && k == "--substeps") { substeps_override = std::atoi(argv[++i]); }
        else if (i + 1 < argc && k == "--ghost-stride") { ghost_stride = std::atoi(argv[++i]); }
        else if (i + 1 < argc && k == "--flow")
        {
            if (!ParseFlowTriple(argv[++i], flow_vx, flow_vy, flow_vz))
            {
                std::cerr << "bad --flow, expected VX,VY,VZ\n";
                return 1;
            }
            flow_override = true;
        }
    }

    EnsureDir(out_dir);

    Environment env("data/fish.meta", false);
    env.SetSimulationHz(sim_hz_arg);
    env.SetVolumeLogIntervalSteps(0);

    PeridynoBridge* bridge = env.GetFluidBridge();
    if (!bridge || !bridge->IsInitialized())
    {
        std::cerr << "fail: no fluid bridge\n";
        return 1;
    }
    if (substeps_override > 0)
        bridge->SetSubsteps(substeps_override);
    if (flow_override)
        bridge->SetFlowVelocity(Eigen::Vector3d(flow_vx, flow_vy, flow_vz));
    bridge->SetGhostSampleStride(ghost_stride);
    bridge->SetContactParams(0.3f, 15.0f);
    bridge->SetFlowRampTime(ramp_sec);
    bridge->SetHydroForceScale(1.0f);
    bridge->SetMaxVertexForce(max_force);
    bridge->SetEnableContactRepulsion(false);
    bridge->SetEnableHydroPressure(true);

    const int sim_hz = (int)env.GetSimulationHz();
    const int ctrl_hz = (int)env.GetControlHz();
    const int ratio = sim_hz / ctrl_hz;
    const int total_steps = (int)(seconds * sim_hz);
    const std::vector<int> snapshot_iters = BuildSnapshotIters(sim_hz, t_start, seconds, t_step);

    const auto& contours = env.GetCreature()->GetContours();
    std::set<int> surface_vertices;
    for (const auto& tri : contours)
    {
        surface_vertices.insert(tri[0]);
        surface_vertices.insert(tri[1]);
        surface_vertices.insert(tri[2]);
    }
    if (!torque_only)
        WriteFacesCsv(out_dir + "/surface_faces.csv", contours);

    const std::string summary_path = out_dir + "/torque_summary.csv";
    std::ofstream summary(summary_path);
    summary << "sim_time,iter,tau_x_surface_com,tau_x_surface_center,tau_x_all_com,"
               "force_sum_x,force_sum_y,force_sum_z,ramp\n";

    const Eigen::Vector3d flow_vel = bridge->GetFlowVelocity();
    const Eigen::Vector3d com0 = ComputeCom(env.GetSoftWorld()->GetPositions());
    const Eigen::Vector3d ub0 = ComputeUnbiasedTranslation(env.GetCreature(), env.GetSoftWorld()->GetPositions());

    std::cout << "=== FSI snapshot ===\n"
              << "seconds=" << seconds << " sim_hz=" << sim_hz
              << " t=[" << t_start << "," << seconds << "] step=" << t_step
              << " max_force=" << max_force << " ramp_sec=" << ramp_sec
              << " flow=(" << flow_vel.x() << "," << flow_vel.y() << "," << flow_vel.z() << ")"
              << " ghost_stride=" << ghost_stride
              << " substeps=" << bridge->GetSubsteps()
              << " dx=" << bridge->GetParticleSpacing()
              << " torque_only=" << (torque_only ? 1 : 0) << '\n';

    for (int step = 0; step < total_steps; ++step)
    {
        if (step % ratio == 0)
        {
            const double t = step / static_cast<double>(sim_hz);
            env.GetCreature()->SetActivationLevelsDirectly(MakeSineActivations(&env, t));
            env.SetPhase(step / ratio);
        }

        env.Step();
        const int iter = step + 1;
        const double sim_t = IterToTime(iter, sim_hz);

        if (IsHalfSecondMark(sim_t, sim_hz))
        {
            const FsiVertexForceSnapshot& monitor_snap = bridge->GetLastFsiVertexForceSnapshot();
            if (monitor_snap.valid)
            {
                const Eigen::VectorXd& monitor_pos =
                    monitor_snap.positions.size() > 0 ? monitor_snap.positions : env.GetSoftWorld()->GetPositions();
                LogHalfSecondMonitor(sim_t, iter, monitor_snap, monitor_pos, surface_vertices);
            }
            else
            {
                std::cerr << "warn: half-second monitor skipped (invalid snapshot) t=" << sim_t << " iter=" << iter << '\n';
            }
        }

        if (!IsSnapshotIter(iter, snapshot_iters))
            continue;

        const FsiVertexForceSnapshot& snap = bridge->GetLastFsiVertexForceSnapshot();
        if (!snap.valid)
        {
            std::cerr << "warn: snapshot invalid at iter " << iter << '\n';
            continue;
        }

        const Eigen::VectorXd& pos = snap.positions.size() > 0 ? snap.positions : env.GetSoftWorld()->GetPositions();
        const int center_idx = env.GetCreature()->GetCenterIndex();
        const Eigen::Vector3d com = ComputeCom(pos);
        const Eigen::Vector3d center = pos.segment<3>(3 * center_idx);
        const TorqueReport rep = ComputeTorques(pos, snap.total, contours, center_idx);

        std::ostringstream tag;
        tag << "t" << std::fixed << std::setprecision(2) << sim_t;
        if (!torque_only)
        {
            const std::string prefix = out_dir + "/" + tag.str();
            WriteForcesCsv(prefix + "_forces.csv", snap);
            {
                std::ofstream pout(prefix + "_positions.csv");
                WriteVectorCsv(pout, pos);
            }
            WriteMeta(prefix + "_meta.txt", sim_t, iter, snap, rep, com, center);
        }

        summary << sim_t << ',' << iter << ','
                << rep.tau_x_surface_com << ',' << rep.tau_x_surface_center << ','
                << rep.tau_x_all_com << ','
                << rep.force_sum_surface.x() << ',' << rep.force_sum_surface.y() << ','
                << rep.force_sum_surface.z() << ',' << snap.flow_ramp << '\n';

        std::cout << "[snapshot t=" << sim_t << "s iter=" << iter << "] "
                  << "tau_x(com,surface)=" << rep.tau_x_surface_com << " N·m, "
                  << "tau_x(center,surface)=" << rep.tau_x_surface_center << " N·m\n";
    }

    summary.close();

    const Eigen::VectorXd& x_final = env.GetSoftWorld()->GetPositions();
    const Eigen::Vector3d com_final = ComputeCom(x_final);
    const Eigen::Vector3d ub_final = ComputeUnbiasedTranslation(env.GetCreature(), x_final);
    const Eigen::Vector3d com_disp = com_final - com0;
    const Eigen::Vector3d ub_disp = ub_final - ub0;
    const double sim_t_final = total_steps / static_cast<double>(sim_hz);

    const std::string disp_path = out_dir + "/displacement_summary.txt";
    {
        std::ofstream disp_out(disp_path);
        disp_out << std::setprecision(10);
        disp_out << "sim_time=" << sim_t_final << '\n';
        disp_out << "flow_velocity=" << flow_vel.transpose() << '\n';
        disp_out << "com0=" << com0.transpose() << '\n';
        disp_out << "com_final=" << com_final.transpose() << '\n';
        disp_out << "com_disp=" << com_disp.transpose() << '\n';
        disp_out << "com_disp_z=" << com_disp.z() << '\n';
        disp_out << "ub0=" << ub0.transpose() << '\n';
        disp_out << "ub_final=" << ub_final.transpose() << '\n';
        disp_out << "ub_disp=" << ub_disp.transpose() << '\n';
        disp_out << "ub_disp_z=" << ub_disp.z() << '\n';
    }

    const VolumeStats vol = env.GetCreature()->ComputeVolumeStats(x_final);
    int hydro_nonzero = 0;
    const FsiVertexForceSnapshot& snap_final = bridge->GetLastFsiVertexForceSnapshot();
    if (snap_final.valid)
    {
        for (int i : surface_vertices)
        {
            const double hx = snap_final.hydro(3 * i);
            const double hy = snap_final.hydro(3 * i + 1);
            const double hz = snap_final.hydro(3 * i + 2);
            if (hx * hx + hy * hy + hz * hz > 1e-24)
                ++hydro_nonzero;
        }
    }
    const int surf_n = (int)surface_vertices.size();
    const double hydro_nonzero_pct = 100.0 * hydro_nonzero / std::max(surf_n, 1);

    std::cout << std::setprecision(6);
    std::cout << "SUMMARY sim_time=" << sim_t_final
              << " final_vol=" << vol.volume_ratio
              << " final_inverted=" << vol.inverted_tets
              << " com_disp=(" << com_disp.x() << "," << com_disp.y() << "," << com_disp.z() << ")"
              << " ub_disp=(" << ub_disp.x() << "," << ub_disp.y() << "," << ub_disp.z() << ")"
              << " com_disp_z=" << com_disp.z()
              << " ub_disp_z=" << ub_disp.z()
              << " hydro_nonzero_surf=" << hydro_nonzero << "/" << surf_n
              << " hydro_nonzero_pct=" << hydro_nonzero_pct
              << " ghost_stride=" << ghost_stride
              << " flow=(" << flow_vel.x() << "," << flow_vel.y() << "," << flow_vel.z() << ")\n";
    std::cout << "done. summary -> " << summary_path << '\n';
    std::cout << "displacement -> " << disp_path << '\n';
    return 0;
}
