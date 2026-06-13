#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

/**
 * PeridynoBridge: 轻量级 GPU SPH 流体求解器桥接类
 *
 * 在 GPU 上执行简化的 WCSPH（弱可压缩 SPH）流体仿真，
 * 将鱼体表面三角 mesh 作为移动边界条件，计算并返回各表面顶点受到
 * 的流体力（接触排斥力 + 流体压力）。
 *
 * 主流程（cpp侧）:
 *   1. 提取鱼体表面 mesh (FEM World → 表面顶点+面拓扑)
 *   2. 调用 SetFishBoundary(...)  → 上传边界数据到 GPU
 *   3. 调用 ComputeFluidForces(dt) → GPU SPH 子步进 → 返回时间平均力
 *   4. FEM World 施加外力 → Projective Dynamics 时间推进
 *
 * 边界处理（v2）:
 *   - 三角形基元：流体粒子 ↔ 最近三角面，力按重心坐标分配
 *   - 分离非穿透接触力（弹簧阻尼）与流体压力（SPH 压力积分）
 *   - 子步进 + 时间平均冲量，减少显式弱耦合冲击
 */
/** 过去约 1 仿真秒内 FSI 力的滚动统计（由 ConsumeSecondFsiForceDiagnostics 输出）
 *  total：每帧全顶点力模长之和 Σ|F_i| 再对帧求平均 [N]
 *  peak：窗口内单顶点力模长最大值 max|F_i| [N] */
struct FsiForceDiagnostics {
    double contact_total_avg_n = 0.0;
    double hydro_total_avg_n   = 0.0;
    double contact_peak_n      = 0.0;
    double hydro_peak_n        = 0.0;
    int    frames              = 0;
};

/** 上一帧 ComputeFluidForces 后各 FEM 顶点的分轨/合力（与传给 FEM 的处理一致） */
struct FsiVertexForceSnapshot {
    Eigen::VectorXd contact;   // 3×N，接触排斥 [N]
    Eigen::VectorXd hydro;     // 3×N，水压力 [N]
    Eigen::VectorXd total;     // 3×N，合力（含顶点截断）[N]
    Eigen::VectorXd positions; // 3×N，计算力时顶点位置 [m]
    float flow_ramp = 0.0f;
    bool valid = false;
};

class PeridynoBridge
{
public:
    PeridynoBridge();
    ~PeridynoBridge();

    // ---- 初始化 ----
    void Initialize(
        const Eigen::Vector3d& domain_min,
        const Eigen::Vector3d& domain_max,
        float particle_spacing,
        const Eigen::Vector3d& flow_velocity = Eigen::Vector3d(0, 0, 0),
        float rest_density = 1000.0f,
        float viscosity = 0.01f,
        float sound_speed = 20.0f,
        float boundary_thickness = 0.02f,
        const Eigen::VectorXd& surface_vertices = Eigen::VectorXd(),
        const std::vector<Eigen::Vector3i>& surface_faces = {});

    // 当前来流 ramp 系数 [0,1]
    float GetFlowRampFactor() const;

    // ---- 子步进与接触参数（初始化后可调）----
    // 每个 FEM 帧内 SPH 子步进次数（默认 4）
    void SetSubsteps(int n);
    int  GetSubsteps() const;

    // 非穿透接触力参数（三角形基元排斥）
    // stiffness: 法向刚度 (N/m), 默认 50
    // damping:   法向阻尼系数, 默认 1.0
    void SetContactParams(float stiffness, float damping);

    // 来流 ramp 时间（秒），力缩放 ~ ramp
    void SetFlowRampTime(float seconds);
    float GetFlowRampTime() const { return m_flow_ramp_time; }

    // SPH 压力积分水动力全局缩放 [0,1]（运行时；编译期 ENABLE_HYDRO_PRESSURE 可完全剔除代码）
    void SetHydroForceScale(float scale);
    float GetHydroForceScale() const { return m_hydro_force_scale; }

    // 运行时开关：接触排斥 / 水压力积分（需编译期对应 ENABLE_* 为 1 才生效）
    void SetEnableContactRepulsion(bool enable);
    bool GetEnableContactRepulsion() const { return m_enable_contact_repulsion; }
    void SetEnableHydroPressure(bool enable);
    bool GetEnableHydroPressure() const { return m_enable_hydro_pressure; }

    // 取出并重置过去累积的 FSI 力统计（Environment 每秒日志调用一次）
    FsiForceDiagnostics ConsumeSecondFsiForceDiagnostics();

    // 边界顶点力模长上限 (N)，0 表示不截断
    void SetMaxVertexForce(float max_force);
    float GetMaxVertexForce() const { return m_max_vertex_force; }

    // 重置仿真到初始状态
    void Reset();
    void Destroy();

    // ---- 核心接口 ----
    void SetFishBoundary(
        const Eigen::VectorXd& surface_vertices,
        const Eigen::VectorXd& surface_velocities,
        const std::vector<Eigen::Vector3i>& surface_faces);

    // 返回各表面顶点的流体力 (N*3, 索引顺序与 surface_vertices 一致)
    Eigen::VectorXd ComputeFluidForces(float dt);

    // 最近一次 ComputeFluidForces 的分轨快照（positions 为 SetFishBoundary 上传时的顶点位置）
    const FsiVertexForceSnapshot& GetLastFsiVertexForceSnapshot() const { return m_last_fsi_snapshot; }

    bool IsInitialized() const { return m_initialized; }
    int  GetFluidParticleCount() const;
    void GetFluidParticles(std::vector<float>& positions) const;
    void GetFluidVelocities(std::vector<float>& velocities) const;
    float GetParticleSpacing() const { return m_particle_spacing; }

private:
    void InitFluidParticles();
    void FreeVapGpuBuffers();
    void EnsureVapGpuCapacity(int num_fluid, int num_ghost);
    void RunVapSubstepImpl(float dt_sub, float ghost_time_offset, int bs);

    struct Impl;
    Impl* m_impl;

    Eigen::Vector3d m_domain_min, m_domain_max;
    float m_particle_spacing;
    Eigen::Vector3d m_flow_velocity;
    float m_rest_density;
    float m_viscosity;
    float m_sound_speed;
    float m_boundary_thickness;
    Eigen::VectorXd m_init_surface_vertices;
    std::vector<Eigen::Vector3i> m_init_surface_faces;
    bool m_initialized;

    int m_n_substeps = 4;
    float m_contact_stiffness = 10.0f;
    float m_contact_damping = 2.0f;
    float m_flow_ramp_time = 0.5f;
    float m_hydro_force_scale = 1.0f;
    float m_max_vertex_force = 0.0f;
    bool m_enable_contact_repulsion = true;
    bool m_enable_hydro_pressure = true;

    double m_roll_sum_contact_total = 0.0;
    double m_roll_sum_hydro_total = 0.0;
    double m_roll_peak_contact = 0.0;
    double m_roll_peak_hydro = 0.0;
    int m_roll_frame_count = 0;

    FsiVertexForceSnapshot m_last_fsi_snapshot;

    // 表面拓扑：用于零水压顶点的邻域插值
    std::vector<uint8_t> m_on_surface;
    std::vector<std::vector<int>> m_surface_adj;
    void RebuildSurfaceAdjacency(int nv, const std::vector<Eigen::Vector3i>& faces);
    void FillZeroHydroFromNeighbors(
        std::vector<float>& hx, std::vector<float>& hy, std::vector<float>& hz, int nv) const;
};
