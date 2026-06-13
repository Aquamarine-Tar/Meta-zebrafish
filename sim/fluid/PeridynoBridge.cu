/**
 * PeridynoBridge.cu — GPU 弱可压缩 SPH 流体 + 三角面 FSI 耦合
 *
 * 【整体模型】
 *   流体：WCSPH（Weakly Compressible SPH），Tait 状态方程求压，Poly6/Spiky 核。
 *   固体：FEM 鱼体表面三角 mesh 作为移动边界；力回传到各 FEM 顶点。
 *   耦合：显式子步弱耦合（每 FEM 帧内 n_substeps 次 SPH 子步，帧末时间平均力）。
 *
 * 【每帧主循环】（ComputeFluidForces）
 *   1. 更新鱼体三角面心/法向 → 重建面网格
 *   2. 重复 n_substeps 次：
 *      a. 重建流体均匀网格 → Tait 密度/压力
 *      b. VAP 模式：band 外 WCSPH 压力+粘性，band 内仅粘性；VAP 投影 → 面力
 *         纯 WCSPH 模式：全域压力+粘性
 *      c. 三角面排斥（可选）+ 面基水压力积分 → 边界力
 *      d. 域边界约束 + 来流 ramp → 显式积分
 *   3. 子步力求平均 → 可选顶点力截断 → 返回 Eigen::VectorXd 给 FEM
 *
 * 【单位】SI：m, kg, s, N, Pa
 */
#include "PeridynoBridge.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <chrono>

// ===== 编译期开关（编译时可彻底剔除对应核函数；运行时仍可用 SetEnable* 关闭）=====
#ifndef ENABLE_CONTACT_REPULSION
#define ENABLE_CONTACT_REPULSION 1  // 1=编译接触排斥核，0=完全不生成接触力代码
#endif
#ifndef ENABLE_HYDRO_PRESSURE
#define ENABLE_HYDRO_PRESSURE 1       // 1=编译水压力积分核，0=完全不生成水压力代码
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ---- float3 补充运算符 ----
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
__host__ __device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    if (len > 1e-12f) return v / len;
    return make_float3(0, 0, 0);
}

// int3 由 CUDA vector_types.h 提供

// 均匀网格加速：每格最多存放的粒子/三角面数量（超出则丢弃，影响精度）
#define MAX_PARTICLES_PER_CELL  64
#define MAX_FACES_PER_CELL      32
// 光滑长度 h = particle_spacing × 该比值（默认 h = 2Δx）
#define SPH_KERNEL_RADIUS_RATIO 2.0f

#define CUDA_SAFE_FREE(ptr) do { if (ptr) { cudaFree(ptr); ptr = nullptr; } } while(0)

/** 均匀网格参数，用于 O(N) 邻域搜索 */
struct GridParams {
    float3 grid_min;       // 网格原点（含 ghost 层）
    float  cell_size;      // 单元边长，通常取 h
    float  inv_cell_size;  // 1/cell_size
    int3   grid_dim;       // 各轴格数
    int    total_cells;    // 总格数 = dx×dy×dz
};

/** GPU 端全部仿真状态（Pimpl，与 PeridynoBridge 公共接口分离） */
struct PeridynoBridge::Impl {
    // ---- 流体 SPH 粒子 (GPU) ----
    float3* d_positions  = nullptr;
    float3* d_velocities = nullptr;
    float*  d_densities  = nullptr;
    float*  d_pressures  = nullptr;
    float3* d_forces     = nullptr;
    int num_particles = 0;

    // ---- FEM 鱼体边界顶点 (GPU)：每帧由 SetFishBoundary H2D 上传 ----
    float3* d_boundary_vertices   = nullptr;
    float3* d_boundary_velocities = nullptr;
    int     num_boundary_vertices = 0;

    // ---- 鱼体表面三角面 (GPU) ----
    int3*   d_face_indices  = nullptr;  // [num_faces] 三个顶点索引 (i0,i1,i2)
    float3* d_face_centers  = nullptr;  // [num_faces] 面心，每帧 update_face_data_kernel 更新
    float3* d_face_normals  = nullptr;  // [num_faces] 单位法向，每帧更新
    float*  d_face_areas    = nullptr;  // [num_faces] 面积，仅在拓扑首次上传时计算（当前未随变形更新）
    int     num_faces = 0;

    // ---- 三角面均匀网格 (GPU, 每帧重建，用于粒子→面 邻域搜索) ----
    GridParams face_grid_params;
    int* d_face_grid_counts   = nullptr;
    int* d_face_grid_particles = nullptr;  // 存储 face 索引
    bool face_grid_allocated = false;

    // ---- FSI 边界力缓冲 (GPU/CPU)：接触 / 水压力分轨统计 ----
    float3* d_boundary_forces_contact_sub   = nullptr;
    float3* d_boundary_forces_hydro_sub     = nullptr;
    float3* d_boundary_forces_contact_frame = nullptr;
    float3* d_boundary_forces_hydro_frame   = nullptr;
    float3* h_boundary_forces_contact_frame = nullptr;
    float3* h_boundary_forces_hydro_frame   = nullptr;

    // ---- 流体 SPH 均匀网格 (GPU, 每子步重建) ----
    int* d_grid_counts    = nullptr;
    int* d_grid_cell_start = nullptr;
    int* d_grid_cell_end  = nullptr;
    int* d_grid_particles = nullptr;
    GridParams grid_params;

    // ---- SPH 与 FSI 物理参数（Initialize 时由外部传入覆盖）----
    float particle_spacing = 0.04f;   // Δx：粒子初始间距 [m]
    float h     = 0.08f;              // 核支撑半径 h = 2Δx [m]
    float h2    = 0.0064f;            // h²，Poly6 核用
    float mass  = 0.064f;             // 单粒子质量 m = ρ₀ Δx³ [kg]
    float rest_density = 1000.0f;     // 参考密度 ρ₀ [kg/m³]（近似水）
    float viscosity    = 0.01f;       // 人工粘性系数 μ（非真实动力粘度）[Pa·s 量级]
    float sound_speed  = 20.0f;       // 状态方程声速 c_s [m/s]，控制可压缩性
    float gravity_y    = 0.0f;        // 重力加速度 g_y [m/s²]（当前为 0）
    float3 flow_velocity   = make_float3(0, 0, 0);  // 目标来流速度 [m/s]
    float  flow_ramp_time  = 0.5f;    // 来流线性 ramp 时间 [s]，ramp = min(t/T, 1)
    float  hydro_force_scale = 1.0f;  // 水压力积分全局缩放（Environment 中常设为 0.035）
    float  max_vertex_force  = 0.0f;  // 单顶点力模长上限 [N]，0=不截断
    float  current_time    = 0.0f;    // 流体仿真累计时间 [s]
    float  last_ramp       = 0.0f;    // 上一帧 ramp 系数，供诊断/二次缩放
    int    frame_count     = 0;       // 已计算帧数

    // 子步进：每个 FEM 时间步内 SPH 积分次数（Environment 默认 24）
    int n_substeps = 4;

    // 三角面非穿透接触（弹簧-阻尼，与流体压力分离）
    float contact_stiffness = 10.0f;   // 法向刚度 k [N/m]
    float contact_damping   = 2.0f;    // 法向阻尼系数 β（无量纲，乘以接近速度）

    float3 domain_min_f3, domain_max_f3;
    Eigen::Vector3d domain_min, domain_max;
    bool initialized = false;

    // ---- VAP 窄带投影（ghost 顶点 + band 内流体）----
    float3* d_boundary_normals = nullptr;
    unsigned char* d_vap_band_mask = nullptr;
    int* d_vap_prefix = nullptr;
    int* d_vap_fluid_to_merged = nullptr;
    float3* d_vap_merged_pos = nullptr;
    float3* d_vap_merged_vel = nullptr;
    float3* d_vap_merged_nrm = nullptr;
    unsigned char* d_vap_merged_attr = nullptr;
    float* d_vap_merged_density = nullptr;
    int* d_vap_neighbor_count = nullptr;
    int* d_vap_neighbors = nullptr;
    float* d_vap_alpha = nullptr;
    float* d_vap_aii = nullptr;
    float* d_vap_aii_fluid = nullptr;
    float* d_vap_aii_total = nullptr;
    float* d_vap_pressure = nullptr;
    float* d_vap_divergence = nullptr;
    unsigned char* d_vap_is_surface = nullptr;
    float* d_vap_ax = nullptr;
    float* d_vap_r = nullptr;
    float* d_vap_p = nullptr;
    int* d_vap_merged_grid_counts = nullptr;
    int* d_vap_merged_grid_particles = nullptr;
    float* d_vap_dot_partial = nullptr;
    int*   d_vap_band_counter = nullptr;
    int vap_merged_capacity = 0;
    int vap_dot_partial_cap = 0;
    int vap_last_n_band = 0;
    float vap_alpha_max = 0.0f;
    float vap_a_max = 0.0f;
};

// ==================== SPH 核函数（标准 WCSPH 形式）====================

/**
 * Poly6 核 W(r,h)：用于密度求和
 *   W(r) = (315 / 64πh⁹) · (h² - r²)³,  r < h
 * 密度：ρ_i = Σ_j m_j W(|x_i - x_j|, h)
 */
__device__ float poly6_kernel(float r2, float h, float h2) {
    if (r2 >= h2) return 0.0f;
    float diff = h2 - r2;
    return (315.0f / (64.0f * M_PI * powf(h, 9))) * diff * diff * diff;
}

/**
 * Spiky 核梯度 ∇W：用于压力梯度力
 *   ∇W = -(45 / πh⁶) · (h - r)² · (r_vec / r)
 * 压力力：f_p = -m · (p_i+p_j)/(2ρ_j) · ∇W
 */
__device__ float3 spiky_grad(const float3& r_vec, float r, float h) {
    if (r < 1e-8f || r >= h) return make_float3(0, 0, 0);
    float diff = h - r;
    float coeff = -45.0f / (M_PI * h * h * h * h * h * h) * diff * diff;
    return r_vec * (coeff / r);
}

/**
 * 粘性核拉普拉斯 ∇²W：用于人工粘性
 *   ∇²W = (45 / πh⁶) · (h - r)
 * 粘性力：f_v = m · μ · (v_j - v_i)/ρ_j · ∇²W
 */
__device__ float viscosity_laplacian(float r, float h) {
    if (r >= h) return 0.0f;
    return 45.0f / (M_PI * h * h * h * h * h * h) * (h - r);
}

#include "VapHydro.inl"

// ---- 将每个流体粒子插入均匀网格（并行 atomic 计数）----
__global__ void build_grid_kernel(
    const float3* positions, int* grid_counts, int* grid_particles,
    GridParams gp, int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    float3 pos = positions[idx];
    int cx = max(0, min((int)((pos.x - gp.grid_min.x) * gp.inv_cell_size), gp.grid_dim.x - 1));
    int cy = max(0, min((int)((pos.y - gp.grid_min.y) * gp.inv_cell_size), gp.grid_dim.y - 1));
    int cz = max(0, min((int)((pos.z - gp.grid_min.z) * gp.inv_cell_size), gp.grid_dim.z - 1));
    int cell = cz * gp.grid_dim.x * gp.grid_dim.y + cy * gp.grid_dim.x + cx;
    int slot = atomicAdd(&grid_counts[cell], 1);
    if (slot < MAX_PARTICLES_PER_CELL)
        grid_particles[cell * MAX_PARTICLES_PER_CELL + slot] = idx;
}

/**
 * 计算每个粒子的 SPH 密度与 Tait 状态方程压力
 *
 * 密度：ρ_i = max(Σ_j m_j W_ij, ρ₀)
 * 压力（Tait EOS, γ=7）：
 *   p_i = max( (ρ₀ c_s² / 7) · [ (ρ_i/ρ₀)^7 - 1 ], 0 )
 * 仅保留非负压力（弱可压缩近似）
 */
__global__ void compute_density_pressure_kernel(
    const float3* positions, float* densities, float* pressures,
    const int* grid_particles, GridParams gp,
    float h, float h2, float mass, float rest_density, float sound_speed,
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    float3 pos_i = positions[idx];
    float density = 0.0f;
    int cx = max(0, min((int)((pos_i.x - gp.grid_min.x) * gp.inv_cell_size), gp.grid_dim.x - 1));
    int cy = max(0, min((int)((pos_i.y - gp.grid_min.y) * gp.inv_cell_size), gp.grid_dim.y - 1));
    int cz = max(0, min((int)((pos_i.z - gp.grid_min.z) * gp.inv_cell_size), gp.grid_dim.z - 1));
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || nx >= gp.grid_dim.x || ny < 0 || ny >= gp.grid_dim.y || nz < 0 || nz >= gp.grid_dim.z) continue;
        int cell = nz * gp.grid_dim.x * gp.grid_dim.y + ny * gp.grid_dim.x + nx;
        for (int c = 0; c < MAX_PARTICLES_PER_CELL; c++) {
            int j = grid_particles[cell * MAX_PARTICLES_PER_CELL + c];
            if (j < 0) break;
            float3 r = pos_i - positions[j];
            float r2 = dot(r, r);
            density += mass * poly6_kernel(r2, h, h2);
        }
    }
    densities[idx] = fmaxf(density, rest_density);
    float rho_ratio = densities[idx] / rest_density;
    pressures[idx] = fmaxf((rest_density * sound_speed * sound_speed / 7.0f) * (powf(rho_ratio, 7.0f) - 1.0f), 0.0f);
}

/**
 * 计算 SPH 粒子间压力力 + 人工粘性 + 重力
 *
 * 对称压力项（Monaghan 形式，j>idx 避免重复，力反加到 j）：
 *   f_p = -m (p_i+p_j)/(2ρ_j) ∇W_ij
 * 粘性项：
 *   f_v = m μ (v_j-v_i)/ρ_j ∇²W_ij
 * 重力：f_g = (0, m·g_y, 0)
 */
__global__ void compute_sph_forces_kernel(
    const float3* positions, const float3* velocities,
    const float* densities, const float* pressures,
    float3* forces, const int* grid_particles, GridParams gp,
    float h, float mass, float rest_density, float viscosity,
    float gravity_y, int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    float3 pos_i = positions[idx], vel_i = velocities[idx];
    float  p_i   = pressures[idx];
    (void)densities;
    float3 force = make_float3(0, mass * gravity_y, 0);
    int cx = max(0, min((int)((pos_i.x - gp.grid_min.x) * gp.inv_cell_size), gp.grid_dim.x - 1));
    int cy = max(0, min((int)((pos_i.y - gp.grid_min.y) * gp.inv_cell_size), gp.grid_dim.y - 1));
    int cz = max(0, min((int)((pos_i.z - gp.grid_min.z) * gp.inv_cell_size), gp.grid_dim.z - 1));
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || nx >= gp.grid_dim.x || ny < 0 || ny >= gp.grid_dim.y || nz < 0 || nz >= gp.grid_dim.z) continue;
        int cell = nz * gp.grid_dim.x * gp.grid_dim.y + ny * gp.grid_dim.x + nx;
        for (int c = 0; c < MAX_PARTICLES_PER_CELL; c++) {
            int j = grid_particles[cell * MAX_PARTICLES_PER_CELL + c];
            if (j < 0) break;
            if (j <= idx) continue;
            float3 r = pos_i - positions[j];
            float dist = length(r);
            if (dist >= h || dist < 1e-8f) continue;
            float rho_j = densities[j], p_j = pressures[j];
            float3 f_p = -mass * (p_i + p_j) / (2.0f * rho_j) * spiky_grad(r, dist, h);
            float lapl  = viscosity_laplacian(dist, h);
            float3 f_v  = mass * viscosity * (velocities[j] - vel_i) / rho_j * lapl;
            force = force + f_p + f_v;
            forces[j] = forces[j] - f_p - f_v;
        }
    }
    forces[idx] = forces[idx] + force;
}

/**
 * VAP 混合：band 外粒子用 WCSPH 压力+粘性（抗 bulk 淤积）；band 内仅粘性，压力由 VAP→面力。
 * 对 (i,j) 成对作用：仅当 i 在 band 外时施加压力；band 内粒子只接收粘性项。
 */
__global__ void compute_sph_forces_hybrid_kernel(
    const float3* positions, const float3* velocities,
    const float* densities, const float* pressures,
    const unsigned char* band_mask,
    float3* forces, const int* grid_particles, GridParams gp,
    float h, float mass, float rest_density, float viscosity,
    float gravity_y, int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    const bool in_band_i = band_mask[idx] != 0;
    float3 pos_i = positions[idx], vel_i = velocities[idx];
    float  p_i   = pressures[idx];
    (void)rest_density;
    float3 force = make_float3(0, mass * gravity_y, 0);
    int cx = max(0, min((int)((pos_i.x - gp.grid_min.x) * gp.inv_cell_size), gp.grid_dim.x - 1));
    int cy = max(0, min((int)((pos_i.y - gp.grid_min.y) * gp.inv_cell_size), gp.grid_dim.y - 1));
    int cz = max(0, min((int)((pos_i.z - gp.grid_min.z) * gp.inv_cell_size), gp.grid_dim.z - 1));
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || nx >= gp.grid_dim.x || ny < 0 || ny >= gp.grid_dim.y || nz < 0 || nz >= gp.grid_dim.z) continue;
        int cell = nz * gp.grid_dim.x * gp.grid_dim.y + ny * gp.grid_dim.x + nx;
        for (int c = 0; c < MAX_PARTICLES_PER_CELL; c++) {
            int j = grid_particles[cell * MAX_PARTICLES_PER_CELL + c];
            if (j < 0) break;
            if (j <= idx) continue;
            float3 r = pos_i - positions[j];
            float dist = length(r);
            if (dist >= h || dist < 1e-8f) continue;
            float rho_j = densities[j], p_j = pressures[j];
            float3 f_p = -mass * (p_i + p_j) / (2.0f * rho_j) * spiky_grad(r, dist, h);
            float lapl  = viscosity_laplacian(dist, h);
            float3 f_v  = mass * viscosity * (velocities[j] - vel_i) / rho_j * lapl;
            const bool in_band_j = band_mask[j] != 0;
            if (!in_band_i && !in_band_j) {
                force = force + f_p + f_v;
                forces[j] = forces[j] - f_p - f_v;
            } else if (!in_band_i && in_band_j) {
                force = force + f_p + f_v;
                forces[j] = forces[j] - f_v;
            } else {
                force = force + f_v;
                forces[j] = forces[j] - f_v;
            }
        }
    }
    forces[idx] = forces[idx] + force;
}

// ==================== 三角形面数据更新 ====================

/**
 * 根据当前 FEM 顶点位置更新每个三角面的面心与单位法向
 *   面心 c = (v0+v1+v2)/3
 *   法向 n = normalize((v1-v0) × (v2-v0))  （未做一致 outward 定向）
 */
__global__ void update_face_data_kernel(
    const float3* vertices,
    const int3*   face_indices,
    float3*       face_centers,
    float3*       face_normals,
    int num_faces)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    int3 fi   = face_indices[f];
    float3 v0 = vertices[fi.x];
    float3 v1 = vertices[fi.y];
    float3 v2 = vertices[fi.z];

    float3 e0 = v1 - v0;
    float3 e1 = v2 - v0;
    float3 n  = make_float3(e0.y * e1.z - e0.z * e1.y,
                             e0.z * e1.x - e0.x * e1.z,
                             e0.x * e1.y - e0.y * e1.x);
    float len = length(n);
    if (len > 1e-12f) n = n / len;

    face_centers[f] = make_float3((v0.x + v1.x + v2.x) / 3.0f,
                                  (v0.y + v1.y + v2.y) / 3.0f,
                                  (v0.z + v1.z + v2.z) / 3.0f);
    face_normals[f] = n;
}

// ---- 将三角面心插入面网格，供 repulsion 核做粒子→面邻域搜索 ----
__global__ void build_face_grid_kernel(
    const float3* face_centers,
    int*  grid_counts,
    int*  grid_particles,
    GridParams gp,
    int num_faces)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;
    float3 pos = face_centers[f];
    int cx = max(0, min((int)((pos.x - gp.grid_min.x) * gp.inv_cell_size), gp.grid_dim.x - 1));
    int cy = max(0, min((int)((pos.y - gp.grid_min.y) * gp.inv_cell_size), gp.grid_dim.y - 1));
    int cz = max(0, min((int)((pos.z - gp.grid_min.z) * gp.inv_cell_size), gp.grid_dim.z - 1));
    int cell = cz * gp.grid_dim.x * gp.grid_dim.y + cy * gp.grid_dim.x + cx;
    int slot = atomicAdd(&grid_counts[cell], 1);
    if (slot < MAX_FACES_PER_CELL)
        grid_particles[cell * MAX_FACES_PER_CELL + slot] = f;
}

// ==================== 初始化：GPU 并行剔除鱼体内粒子 ====================

/** 与 CPU BuildInitFaceGeometry 相同布局，供初始化过滤核使用 */
struct InitFaceGeometry {
    float3 A, B, C, n;
};

__host__ __device__ void closest_point_on_triangle(
    const float3& A, const float3& B, const float3& C,
    const float3& P,
    float3& Q, float& u, float& v, float& w);

/**
 * 每个线程处理一个候选粒子：对全部三角面求最近点，与接触排斥同判据
 *   outside_signed = dot(P - Q, n)
 *   keep = (outside AABB) || (outside_signed > 0)
 */
__global__ void filter_init_particles_outside_fish_kernel(
    const float3* __restrict__ positions,
    int num_particles,
    const InitFaceGeometry* __restrict__ faces,
    int num_faces,
    float bb_xmin, float bb_ymin, float bb_zmin,
    float bb_xmax, float bb_ymax, float bb_zmax,
    unsigned char* __restrict__ keep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    const float3 P = positions[i];
    if (P.x < bb_xmin || P.x > bb_xmax ||
        P.y < bb_ymin || P.y > bb_ymax ||
        P.z < bb_zmin || P.z > bb_zmax) {
        keep[i] = 1;
        return;
    }

    float best_dist2 = 1e30f;
    float best_outside_signed = 1.0f;
    for (int f = 0; f < num_faces; ++f) {
        const InitFaceGeometry face = faces[f];
        float3 Q;
        float u, v, w;
        closest_point_on_triangle(face.A, face.B, face.C, P, Q, u, v, w);

        const float3 r_vec = P - Q;
        const float dist2 = dot(r_vec, r_vec);
        const float outside_signed = dot(r_vec, face.n);
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_outside_signed = outside_signed;
        }
    }
    keep[i] = (best_outside_signed > 0.0f) ? 1 : 0;
}

// ==================== 三角形最近点 + 重心坐标（接触排斥 / 初始化共用）====================

/**
 * 计算点 P 到三角形 ABC 的最近点 Q 及重心坐标 (u,v,w)
 *   Q = u·A + v·B + w·C,  u+v+w=1
 * 算法：Möller, "Fast Minimum Distance Between a Point and a Triangle"
 * 用途：将接触力按 (u,v,w) 分配到三个 FEM 顶点
 */
__host__ __device__ void closest_point_on_triangle(
    const float3& A, const float3& B, const float3& C,
    const float3& P,
    float3& Q, float& u, float& v, float& w)
{
    float3 E0  = B - A;
    float3 E1  = C - A;
    float3 D   = A - P;

    float a = dot(E0, E0);
    float b = dot(E0, E1);
    float c = dot(E1, E1);
    float d = dot(E0, D);
    float e = dot(E1, D);

    float det = a * c - b * b;
    float s   = b * e - c * d;
    float t   = b * d - a * e;

    if (s + t <= det) {
        if (s < 0.0f) {
            if (t < 0.0f) {
                // 区域 4: 最近点是 A
                if (d < 0.0f) {
                    // 最近点在 edge AB 延长线外
                    s = fmaxf(0.0f, fminf(1.0f, -d / fmaxf(a, 1e-12f)));
                } else {
                    s = 0.0f;
                }
                t = 0.0f;
            } else {
                // 区域 3: 最近点在 edge AC
                s = 0.0f;
                t = fmaxf(0.0f, fminf(1.0f, -e / fmaxf(c, 1e-12f)));
            }
        } else if (t < 0.0f) {
            // 区域 5: 最近点在 edge AB
            s = fmaxf(0.0f, fminf(1.0f, -d / fmaxf(a, 1e-12f)));
            t = 0.0f;
        } else {
            // 区域 0: 投影在三角形内部
            float inv_det = 1.0f / fmaxf(det, 1e-12f);
            s *= inv_det;
            t *= inv_det;
        }
    } else {
        if (s < 0.0f) {
            // 区域 2: vertex C
            s = 0.0f;
            t = 1.0f;
        } else if (t < 0.0f) {
            // 区域 6: vertex B
            s = 1.0f;
            t = 0.0f;
        } else {
            // 区域 1: edge BC
            float numer = c + e - b - d;
            if (numer <= 0.0f) {
                s = 0.0f;
                t = 1.0f;
            } else {
                float denom = a - 2.0f * b + c;
                if (numer >= denom) {
                    s = 1.0f;
                    t = 0.0f;
                } else {
                    s = numer / fmaxf(denom, 1e-12f);
                    t = 1.0f - s;
                }
            }
        }
    }

    Q = A + s * E0 + t * E1;
    u = 1.0f - s - t;
    v = s;
    w = t;
}

/**
 * 三角面非穿透接触排斥力（FSI 耦合之一）
 *
 * 对每个流体粒子，在面网格邻域内搜索三角面：
 *   1. 求粒子到三角面的最近点 Q 及重心坐标 (u,v,w)
 *   2. 仅当粒子在面外侧 (r·n > 0) 且 dist < repulsion_h 时施加力
 *   3. 穿透深度 δ = max(repulsion_h - dist, 0)
 *   4. 法向接触力：F_n = k·δ·(1 + β·max(v_rel·n,0))·n
 *   5. 流体 +F_n；FEM 顶点 -F_n·(u,v,w)（牛顿第三定律）
 */
__global__ void compute_triangle_repulsion_kernel(
    const float3* fluid_positions,
    const float3* fluid_velocities,
    float3*       fluid_forces,
    const float3* boundary_vertices,
    const float3* boundary_velocities,
    const int3*   face_indices,
    const float3* face_normals,
    float3*       boundary_forces,
    const int*    face_grid_particles,
    GridParams    fgp,
    float repulsion_h,
    float stiffness,
    float damping,
    int num_faces,
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float3 pos_i = fluid_positions[idx];
    float3 vel_i = fluid_velocities[idx];
    float3 f_total = make_float3(0, 0, 0);

    int cx = max(0, min((int)((pos_i.x - fgp.grid_min.x) * fgp.inv_cell_size), fgp.grid_dim.x - 1));
    int cy = max(0, min((int)((pos_i.y - fgp.grid_min.y) * fgp.inv_cell_size), fgp.grid_dim.y - 1));
    int cz = max(0, min((int)((pos_i.z - fgp.grid_min.z) * fgp.inv_cell_size), fgp.grid_dim.z - 1));

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= fgp.grid_dim.x || ny < 0 || ny >= fgp.grid_dim.y || nz < 0 || nz >= fgp.grid_dim.z)
                    continue;
                int cell = nz * fgp.grid_dim.x * fgp.grid_dim.y + ny * fgp.grid_dim.x + nx;

                for (int c = 0; c < MAX_FACES_PER_CELL; c++) {
                    int f_idx = face_grid_particles[cell * MAX_FACES_PER_CELL + c];
                    if (f_idx < 0) break;

                    int3 fi = face_indices[f_idx];
                    float3 A = boundary_vertices[fi.x];
                    float3 B = boundary_vertices[fi.y];
                    float3 C = boundary_vertices[fi.z];

                    // 最近点 + 重心坐标
                    float3 Q; float u, v, w;
                    closest_point_on_triangle(A, B, C, pos_i, Q, u, v, w);

                    float3 r_vec = pos_i - Q;
                    float  dist  = length(r_vec);
                    float3 n_f   = face_normals[f_idx];

                    // 仅对在面外侧的粒子施加力
                    float outside_signed = dot(r_vec, n_f);
                    if (outside_signed <= 0.0f) continue;

                    float penetration = fmaxf(repulsion_h - dist, 0.0f);
                    if (penetration <= 0.0f) continue;

                    // 边界点在最近点 Q 处的插值速度
                    float3 vel_b = boundary_velocities[fi.x] * u
                                 + boundary_velocities[fi.y] * v
                                 + boundary_velocities[fi.z] * w;
                    float3 vel_rel = vel_i - vel_b;
                    float  vel_n   = dot(vel_rel, n_f);
                    float  approach = fmaxf(vel_n, 0.0f);

                    // 非穿透法向接触力
                    float rep_mag = stiffness * penetration * (1.0f + damping * approach);
                    float3 f_contact = rep_mag * n_f;

                    f_total = f_total + f_contact;

                    // 反作用力按重心坐标分配到三个 FEM 顶点
                    atomicAdd(&boundary_forces[fi.x].x, -f_contact.x * u);
                    atomicAdd(&boundary_forces[fi.x].y, -f_contact.y * u);
                    atomicAdd(&boundary_forces[fi.x].z, -f_contact.z * u);

                    atomicAdd(&boundary_forces[fi.y].x, -f_contact.x * v);
                    atomicAdd(&boundary_forces[fi.y].y, -f_contact.y * v);
                    atomicAdd(&boundary_forces[fi.y].z, -f_contact.z * v);

                    atomicAdd(&boundary_forces[fi.z].x, -f_contact.x * w);
                    atomicAdd(&boundary_forces[fi.z].y, -f_contact.y * w);
                    atomicAdd(&boundary_forces[fi.z].z, -f_contact.z * w);
                }
            }
        }
    }
    fluid_forces[idx] = fluid_forces[idx] + f_total;
}

/**
 * 面基 SPH 水压力积分力（FSI 耦合之二）
 *
 * 在面心附近插值流体压力 p_avg，若 p_avg>0：
 *   F_hydro = p_avg · A_f · hydro_scale · n_f
 * 反作用力 -F_hydro/3 均分到三个 FEM 顶点
 */
__global__ void compute_hydro_pressure_kernel(
    const float3* fluid_positions,
    const float*  fluid_pressures,
    float3*       fluid_forces,         // 对流体施加反作用力
    const float3* boundary_vertices,
    const int3*   face_indices,
    const float3* face_centers,
    const float3* face_normals,
    const float*  face_areas,
    float3*       boundary_forces,
    const int*    fluid_grid_particles,
    GridParams    fluid_gp,
    float h,
    float rest_density,
    float hydro_force_scale,
    int num_faces,
    int num_particles)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    int3 fi   = face_indices[f];
    float3 fc = face_centers[f];
    float3 nf = face_normals[f];
    float  area = face_areas[f];

    // 在面心附近采样流体压力（按距离加权平均）
    float  pressure_sum    = 0.0f;
    float  weight_sum      = 0.0f;
    int    sample_count    = 0;

    int cx = max(0, min((int)((fc.x - fluid_gp.grid_min.x) * fluid_gp.inv_cell_size), fluid_gp.grid_dim.x - 1));
    int cy = max(0, min((int)((fc.y - fluid_gp.grid_min.y) * fluid_gp.inv_cell_size), fluid_gp.grid_dim.y - 1));
    int cz = max(0, min((int)((fc.z - fluid_gp.grid_min.z) * fluid_gp.inv_cell_size), fluid_gp.grid_dim.z - 1));

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= fluid_gp.grid_dim.x || ny < 0 || ny >= fluid_gp.grid_dim.y || nz < 0 || nz >= fluid_gp.grid_dim.z)
                    continue;
                int cell = nz * fluid_gp.grid_dim.x * fluid_gp.grid_dim.y + ny * fluid_gp.grid_dim.x + nx;

                for (int c = 0; c < MAX_PARTICLES_PER_CELL; c++) {
                    int p_idx = fluid_grid_particles[cell * MAX_PARTICLES_PER_CELL + c];
                    if (p_idx < 0) break;

                    float3 r = fc - fluid_positions[p_idx];
                    float  dist = length(r);
                    if (dist >= h) continue;

                    // SPH 核作为距离权重
                    float w = poly6_kernel(dot(r, r), h, h * h);
                    pressure_sum += fluid_pressures[p_idx] * w;
                    weight_sum   += w;
                    sample_count++;
                }
            }
        }
    }

    if (sample_count == 0 || weight_sum < 1e-12f) return;

    float p_avg = pressure_sum / weight_sum;

    // 只保留正压力（流体压缩时才有推力）
    if (p_avg <= 0.0f) return;

    // 水动力 = 压力 × 面积 × 法向量 × 缩放系数
    float hydro_scale = hydro_force_scale;
    float3 f_hydro = (p_avg * area * hydro_scale) * nf;

    // 反作用力等权分配到3个FEM顶点
    float w3 = 1.0f / 3.0f;
    atomicAdd(&boundary_forces[fi.x].x, -f_hydro.x * w3);
    atomicAdd(&boundary_forces[fi.x].y, -f_hydro.y * w3);
    atomicAdd(&boundary_forces[fi.x].z, -f_hydro.z * w3);

    atomicAdd(&boundary_forces[fi.y].x, -f_hydro.x * w3);
    atomicAdd(&boundary_forces[fi.y].y, -f_hydro.y * w3);
    atomicAdd(&boundary_forces[fi.y].z, -f_hydro.z * w3);

    atomicAdd(&boundary_forces[fi.z].x, -f_hydro.x * w3);
    atomicAdd(&boundary_forces[fi.z].y, -f_hydro.y * w3);
    atomicAdd(&boundary_forces[fi.z].z, -f_hydro.z * w3);

    // 对流体粒子的反作用力（散布到附近粒子）
    // 简化: 不再遍历第二遍粒子，由 SPH 压力力处理流体侧的反作用
    // 这是单向力传递（流体→固体），流体的反作用体现在下一子步的压力计算中
    (void)fluid_forces;
    (void)num_particles;
}

/**
 * VAP 合并网格版水压力积分（替代 compute_hydro_pressure_kernel）
 *
 * 在面心附近从 VAP 合并网格（仅含 band 流体 + ghost 顶点）采样压力，
 * 避免 SPH 网格中 99.9% 零压力粒子的稀释效应。
 * ghost 顶点不贡献压力，仅 band 流体粒子的 VAP 压力参与加权平均。
 */
__global__ void compute_vap_hydro_pressure_kernel(
    const float3* merged_positions,
    const float*  merged_pressures,
    const unsigned char* merged_attr,
    const float3* boundary_vertices,
    const int3*   face_indices,
    const float3* face_centers,
    const float3* face_normals,
    const float*  face_areas,
    float3*       boundary_forces,
    const int*    merged_grid_counts,
    const int*    merged_grid_particles,
    GridParams    gp,         // VAP 合并网格参数（与 SPH 网格相同）
    float h,
    float hydro_force_scale,
    int num_faces,
    int n_band)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    int3 fi   = face_indices[f];
    float3 fc = face_centers[f];
    float3 nf = face_normals[f];
    float  area = face_areas[f];
    // band 标记用 2.5h，面积分采样须与之匹配，否则面心附近无粒子 → 顶点零力
    const float sample_radius = h * VAP_BAND_H_MULT;

    // 在面心附近从 VAP 合并网格采样压力（仅 band 流体粒子）
    float  pressure_sum = 0.0f;
    float  weight_sum   = 0.0f;
    int    sample_count = 0;

    int cx = max(0, min((int)((fc.x - gp.grid_min.x) * gp.inv_cell_size), gp.grid_dim.x - 1));
    int cy = max(0, min((int)((fc.y - gp.grid_min.y) * gp.inv_cell_size), gp.grid_dim.y - 1));
    int cz = max(0, min((int)((fc.z - gp.grid_min.z) * gp.inv_cell_size), gp.grid_dim.z - 1));
    // h≈2Δx 时 ±1 格会漏采；与 band 标记一致用 ceil(h/cell)+1
    const int search_r = (int)ceilf(sample_radius * gp.inv_cell_size) + 1;

    for (int dz = -search_r; dz <= search_r; dz++) {
        for (int dy = -search_r; dy <= search_r; dy++) {
            for (int dx = -search_r; dx <= search_r; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= gp.grid_dim.x || ny < 0 || ny >= gp.grid_dim.y || nz < 0 || nz >= gp.grid_dim.z)
                    continue;
                int cell = nz * gp.grid_dim.x * gp.grid_dim.y + ny * gp.grid_dim.x + nx;

                for (int c = 0; c < VAP_MAX_MERGED_CELL; c++) {
                    int p_idx = merged_grid_particles[cell * VAP_MAX_MERGED_CELL + c];
                    if (p_idx < 0) break;
                    // 跳过 ghost 顶点（非压力 DOF）
                    if (merged_attr[p_idx] == VAP_ATTR_KINEMATIC_GHOST) continue;

                    float3 r = fc - merged_positions[p_idx];
                    float  dist = length(r);
                    if (dist >= sample_radius) continue;

                    // Poly6 核作为距离权重（支撑半径仍为 h）
                    float w = poly6_kernel(dot(r, r), h, h * h);
                    pressure_sum += merged_pressures[p_idx] * w;
                    weight_sum   += w;
                    sample_count++;
                }
            }
        }
    }

    // 网格哈希漏采时，对 band 流体粒子做线性回退
    if (sample_count == 0 && n_band > 0) {
        float nearest_dist = 1e30f;
        float nearest_p = 0.0f;
        for (int i = 0; i < n_band; i++) {
            if (merged_attr[i] == VAP_ATTR_KINEMATIC_GHOST) continue;
            float3 r = fc - merged_positions[i];
            float  dist = length(r);
            if (dist < nearest_dist) {
                nearest_dist = dist;
                nearest_p = merged_pressures[i];
            }
            if (dist >= sample_radius) continue;
            float w = poly6_kernel(dot(r, r), h, h * h);
            pressure_sum += merged_pressures[i] * w;
            weight_sum   += w;
            sample_count++;
        }
        // 面心附近完全无 band 粒子：用最近邻 band 压力外推（上限 2×sample_radius）
        if (sample_count == 0 && nearest_dist < sample_radius * 2.0f) {
            float w = poly6_kernel(nearest_dist * nearest_dist, h, h * h);
            if (w < 1e-12f) w = 1e-12f;
            pressure_sum = nearest_p * w;
            weight_sum = w;
            sample_count = 1;
        }
    }

    if (sample_count == 0 || weight_sum < 1e-12f) return;

    float p_avg = pressure_sum / weight_sum;

    // VAP 压力可正可负，正压→推力沿法向，负压→吸力逆法向
    // 不再做 p_avg <= 0 截断，保留负压力对形状变化的响应
    float3 f_hydro = (p_avg * area * hydro_force_scale) * nf;

    // 反作用力按重心坐标均分到三个 FEM 顶点
    float w3 = 1.0f / 3.0f;
    atomicAdd(&boundary_forces[fi.x].x, -f_hydro.x * w3);
    atomicAdd(&boundary_forces[fi.x].y, -f_hydro.y * w3);
    atomicAdd(&boundary_forces[fi.x].z, -f_hydro.z * w3);
    atomicAdd(&boundary_forces[fi.y].x, -f_hydro.x * w3);
    atomicAdd(&boundary_forces[fi.y].y, -f_hydro.y * w3);
    atomicAdd(&boundary_forces[fi.y].z, -f_hydro.z * w3);
    atomicAdd(&boundary_forces[fi.z].x, -f_hydro.x * w3);
    atomicAdd(&boundary_forces[fi.z].y, -f_hydro.y * w3);
    atomicAdd(&boundary_forces[fi.z].z, -f_hydro.z * w3);
}

/**
 * 流体域边界条件（周期性来流 + 侧壁反射）
 *
 * z 方向（主流向）：超出上边界则 z -= L_z 并重置速度为 ramped_flow（周期通道）
 * x/y 方向：在 margin 内 clamp 位置，法向速度反射并乘以 damping（软壁）
 */
__global__ void constrain_domain_kernel(
    float3* positions, float3* velocities,
    float3 domain_min, float3 domain_max,
    float3 flow_velocity, float damping, float margin,
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    float3 p = positions[idx];
    float3 v = velocities[idx];
    float3 extent = make_float3(domain_max.x - domain_min.x,
                                domain_max.y - domain_min.y,
                                domain_max.z - domain_min.z);

    if (p.z > domain_max.z - margin) {
        p.z -= extent.z;
        v = flow_velocity;
    } else if (p.z < domain_min.z + margin) {
        p.z += extent.z;
        v = flow_velocity;
    }

    if (p.x < domain_min.x + margin) {
        p.x = domain_min.x + margin;
        if (v.x < 0) v.x *= -damping;
    } else if (p.x > domain_max.x - margin) {
        p.x = domain_max.x - margin;
        if (v.x > 0) v.x *= -damping;
    }
    if (p.y < domain_min.y + margin) {
        p.y = domain_min.y + margin;
        if (v.y < 0) v.y *= -damping;
    } else if (p.y > domain_max.y - margin) {
        p.y = domain_max.y - margin;
        if (v.y > 0) v.y *= -damping;
    }

    positions[idx]  = p;
    velocities[idx] = v;
}

/**
 * 来流速度线性混合：v ← (1-ramp)·v + ramp·v_target
 * 与子步内 ramped_flow 配合，使流体逐渐加速到目标来流
 */
__global__ void apply_ramped_flow_kernel(
    float3* velocities, float3 target_flow, float ramp, int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    float3 v = velocities[idx];
    float one_minus = 1.0f - ramp;
    velocities[idx] = make_float3(
        v.x * one_minus + target_flow.x * ramp,
        v.y * one_minus + target_flow.y * ramp,
        v.z * one_minus + target_flow.z * ramp);
}

/**
 * 显式 Euler 积分（子步 dt_sub）
 *   a = F / m
 *   v ← (v + a·dt) · damping     （damping=0.999 数值耗散）
 *   x ← x + v·dt
 * 加速度分量 clamp 到 ±1e4 防止爆炸
 */
__global__ void integrate_kernel(
    float3* positions, float3* velocities,
    const float3* forces, const float* densities,
    float dt, float damping, float mass_inv, int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    float3 accel = forces[idx] * mass_inv;
    float ax = fminf(fmaxf(accel.x, -1e4f), 1e4f);
    float ay = fminf(fmaxf(accel.y, -1e4f), 1e4f);
    float az = fminf(fmaxf(accel.z, -1e4f), 1e4f);
    velocities[idx] = velocities[idx] + make_float3(ax, ay, az) * dt;
    velocities[idx] = velocities[idx] * damping;
    positions[idx]  = positions[idx] + velocities[idx] * dt;
}

// ---- 清除与累积辅助核函数 ----
__global__ void clear_forces_kernel(float3* forces, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    forces[idx] = make_float3(0, 0, 0);
}

__global__ void clear_counts_kernel(int* grid_counts, int* grid_particles, int total_cells, int max_per_cell) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_cells) return;
    grid_counts[idx] = 0;
    for (int i = 0; i < max_per_cell; i++)
        grid_particles[idx * max_per_cell + i] = -1;
}

__global__ void accumulate_forces_kernel(
    const float3* src,
    float3* dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 s = src[i];
    dst[i] = make_float3(dst[i].x + s.x, dst[i].y + s.y, dst[i].z + s.z);
}


// ==================== 公共接口（CPU 侧调度 GPU 核函数）====================

PeridynoBridge::PeridynoBridge() : m_impl(new Impl()), m_initialized(false) {}

PeridynoBridge::~PeridynoBridge() { Destroy(); delete m_impl; }

/**
 * 初始化流体域与 SPH 参数
 *
 * @param domain_min/max     流体计算域 AABB [m]
 * @param particle_spacing   粒子间距 Δx [m] → h=2Δx, m=ρ₀Δx³
 * @param flow_velocity      目标来流速度 [m/s]（Environment: (0,0,-0.05)）
 * @param rest_density       参考密度 ρ₀ [kg/m³]（默认 1000，近似水）
 * @param viscosity          人工粘性 μ（默认 0.01）
 * @param sound_speed        Tait EOS 声速 c_s [m/s]（默认 20）
 * @param boundary_thickness 预留参数（当前未直接使用）
 * @param surface_vertices   鱼体 FEM 顶点（用于初始化时剔除体内粒子）
 * @param surface_faces      鱼体表面三角面（与 SetFishBoundary / 接触排斥一致）
 */
void PeridynoBridge::Initialize(
    const Eigen::Vector3d& domain_min, const Eigen::Vector3d& domain_max,
    float particle_spacing,
    const Eigen::Vector3d& flow_velocity,
    float rest_density, float viscosity,
    float sound_speed, float boundary_thickness,
    const Eigen::VectorXd& surface_vertices,
    const std::vector<Eigen::Vector3i>& surface_faces)
{
    if (m_impl->initialized) Destroy();

    m_domain_min = domain_min; m_domain_max = domain_max;
    m_particle_spacing = particle_spacing;
    m_flow_velocity = flow_velocity;
    m_rest_density = rest_density; m_viscosity = viscosity;
    m_sound_speed = sound_speed; m_boundary_thickness = boundary_thickness;
    m_init_surface_vertices = surface_vertices;
    m_init_surface_faces = surface_faces;

    auto& p = *m_impl;
    p.domain_min = domain_min; p.domain_max = domain_max;
    p.particle_spacing = particle_spacing;
    p.rest_density = rest_density; p.viscosity = viscosity;
    p.sound_speed = sound_speed;
    p.h  = particle_spacing * SPH_KERNEL_RADIUS_RATIO;  // 光滑长度 h = 2Δx
    p.h2 = p.h * p.h;
    p.mass = rest_density * particle_spacing * particle_spacing * particle_spacing;  // m = ρ₀Δx³
    p.flow_velocity = {(float)flow_velocity.x(), (float)flow_velocity.y(), (float)flow_velocity.z()};

    // 同步子步进和接触参数
    p.n_substeps = m_n_substeps;
    p.contact_stiffness = m_contact_stiffness;
    p.contact_damping   = m_contact_damping;
    p.flow_ramp_time = m_flow_ramp_time;
    p.hydro_force_scale = m_hydro_force_scale;
    p.max_vertex_force = m_max_vertex_force;

    // 流体网格参数
    float3 dmin = {(float)domain_min.x(), (float)domain_min.y(), (float)domain_min.z()};
    float3 dmax = {(float)domain_max.x(), (float)domain_max.y(), (float)domain_max.z()};
    p.grid_params.grid_min = dmin - make_float3(p.h, p.h, p.h);
    p.grid_params.cell_size = p.h;
    p.grid_params.inv_cell_size = 1.0f / p.h;
    float3 ext = dmax - dmin + make_float3(2 * p.h, 2 * p.h, 2 * p.h);
    p.grid_params.grid_dim.x = (int)(ext.x * p.grid_params.inv_cell_size) + 1;
    p.grid_params.grid_dim.y = (int)(ext.y * p.grid_params.inv_cell_size) + 1;
    p.grid_params.grid_dim.z = (int)(ext.z * p.grid_params.inv_cell_size) + 1;
    p.grid_params.total_cells = p.grid_params.grid_dim.x * p.grid_params.grid_dim.y * p.grid_params.grid_dim.z;

    p.domain_min_f3 = dmin;
    p.domain_max_f3 = dmax;

    InitFluidParticles();

    // 分配流体网格内存
    int tc = p.grid_params.total_cells;
    cudaMalloc(&p.d_grid_counts, tc * sizeof(int));
    cudaMalloc(&p.d_grid_cell_start, tc * sizeof(int));
    cudaMalloc(&p.d_grid_cell_end, tc * sizeof(int));
    cudaMalloc(&p.d_grid_particles, tc * MAX_PARTICLES_PER_CELL * sizeof(int));

    p.face_grid_allocated = false;
    p.initialized = true;
    m_initialized = true;

    std::cout << "[PeridynoBridge] Initialized: " << p.num_particles << " fluid particles, "
              << "spacing=" << particle_spacing << ", h=" << p.h
              << ", domain " << domain_min.transpose() << " to " << domain_max.transpose()
              << "\n  substeps=" << p.n_substeps
              << " contact(stiffness=" << p.contact_stiffness
              << " damping=" << p.contact_damping << ")"
#if USE_VAP_HYDRO
              << " vap_hydro=on"
#endif
              << std::endl;
}

namespace {

/** 预计算所有表面三角形的顶点与 outward 法向（与 update_face_data_kernel 一致） */
std::vector<InitFaceGeometry> BuildInitFaceGeometry(
    const Eigen::VectorXd& vertices,
    const std::vector<Eigen::Vector3i>& faces)
{
    std::vector<InitFaceGeometry> geom;
    geom.reserve(faces.size());
    for (const Eigen::Vector3i& fi : faces) {
        InitFaceGeometry g;
        g.A = make_float3(
            (float)vertices(3 * fi[0]), (float)vertices(3 * fi[0] + 1), (float)vertices(3 * fi[0] + 2));
        g.B = make_float3(
            (float)vertices(3 * fi[1]), (float)vertices(3 * fi[1] + 1), (float)vertices(3 * fi[1] + 2));
        g.C = make_float3(
            (float)vertices(3 * fi[2]), (float)vertices(3 * fi[2] + 1), (float)vertices(3 * fi[2] + 2));
        const float3 e0 = g.B - g.A;
        const float3 e1 = g.C - g.A;
        g.n = normalize(make_float3(
            e0.y * e1.z - e0.z * e1.y,
            e0.z * e1.x - e0.x * e1.z,
            e0.x * e1.y - e0.y * e1.x));
        geom.push_back(g);
    }
    return geom;
}

/** GPU 并行过滤：复用 closest_point_on_triangle + dot(P-Q,n)>0，CPU 侧 compact */
int FilterParticlesOutsideFishGPU(
    std::vector<float3>& particles,
    const std::vector<InitFaceGeometry>& faces,
    float bb_xmin, float bb_ymin, float bb_zmin,
    float bb_xmax, float bb_ymax, float bb_zmax)
{
    const int n = (int)particles.size();
    if (n == 0 || faces.empty()) return 0;

    InitFaceGeometry* d_faces = nullptr;
    float3* d_pos = nullptr;
    unsigned char* d_keep = nullptr;
    cudaMalloc(&d_faces, faces.size() * sizeof(InitFaceGeometry));
    cudaMalloc(&d_pos, n * sizeof(float3));
    cudaMalloc(&d_keep, n * sizeof(unsigned char));
    cudaMemcpy(d_faces, faces.data(), faces.size() * sizeof(InitFaceGeometry), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, particles.data(), n * sizeof(float3), cudaMemcpyHostToDevice);

    const int bs = 256;
    const int gs = (n + bs - 1) / bs;
    filter_init_particles_outside_fish_kernel<<<gs, bs>>>(
        d_pos, n, d_faces, (int)faces.size(),
        bb_xmin, bb_ymin, bb_zmin, bb_xmax, bb_ymax, bb_zmax,
        d_keep);
    cudaDeviceSynchronize();

    std::vector<unsigned char> keep(n, 0);
    cudaMemcpy(keep.data(), d_keep, n * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    int idx = 0;
    int skipped = 0;
    for (int i = 0; i < n; ++i) {
        if (keep[i])
            particles[idx++] = particles[i];
        else
            skipped++;
    }
    particles.resize(idx);

    cudaFree(d_faces);
    cudaFree(d_pos);
    cudaFree(d_keep);
    return skipped;
}

void ComputeMeshAABB(
    const Eigen::VectorXd& vertices,
    float& xmin, float& ymin, float& zmin,
    float& xmax, float& ymax, float& zmax)
{
    const int n = vertices.size() / 3;
    xmin = ymin = zmin = 1e30f;
    xmax = ymax = zmax = -1e30f;
    for (int i = 0; i < n; ++i) {
        const float x = (float)vertices(3 * i);
        const float y = (float)vertices(3 * i + 1);
        const float z = (float)vertices(3 * i + 2);
        xmin = fminf(xmin, x); xmax = fmaxf(xmax, x);
        ymin = fminf(ymin, y); ymax = fmaxf(ymax, y);
        zmin = fminf(zmin, z); zmax = fmaxf(zmax, z);
    }
}

}  // namespace

/** 在规则网格上生成流体粒子；若提供鱼体表面 mesh，剔除体内粒子（与接触排斥同判据） */
void PeridynoBridge::InitFluidParticles()
{
    auto& p = *m_impl;
    float sp = p.particle_spacing;
    Eigen::Vector3d size = p.domain_max - p.domain_min;
    int nx = (int)(size.x() / sp) + 1;
    int ny = (int)(size.y() / sp) + 1;
    int nz = (int)(size.z() / sp) + 1;
    p.num_particles = nx * ny * nz;

    std::vector<float3> cpu_pos(p.num_particles);
    const bool filter_inside_fish = !m_init_surface_faces.empty();
    std::vector<InitFaceGeometry> init_faces;
    float bb_xmin = 0, bb_ymin = 0, bb_zmin = 0, bb_xmax = 0, bb_ymax = 0, bb_zmax = 0;
    if (filter_inside_fish) {
        init_faces = BuildInitFaceGeometry(m_init_surface_vertices, m_init_surface_faces);
        ComputeMeshAABB(m_init_surface_vertices, bb_xmin, bb_ymin, bb_zmin, bb_xmax, bb_ymax, bb_zmax);
    }

    int idx = 0;
    for (int iz = 0; iz < nz; iz++)
        for (int iy = 0; iy < ny; iy++)
            for (int ix = 0; ix < nx; ix++) {
                float x = (float)p.domain_min.x() + ix * sp + sp * 0.5f;
                float y = (float)p.domain_min.y() + iy * sp + sp * 0.5f;
                float z = (float)p.domain_min.z() + iz * sp + sp * 0.5f;
                x += 0.0001f * (rand() % 100 - 50);
                y += 0.0001f * (rand() % 100 - 50);
                z += 0.0001f * (rand() % 100 - 50);
                cpu_pos[idx++] = {x, y, z};
            }

    int skipped = 0;
    if (filter_inside_fish) {
        auto t0 = std::chrono::high_resolution_clock::now();
        skipped = FilterParticlesOutsideFishGPU(
            cpu_pos, init_faces,
            bb_xmin, bb_ymin, bb_zmin, bb_xmax, bb_ymax, bb_zmax);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double filter_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (skipped > 0) {
            printf("[PeridynoBridge] GPU excluded %d particles inside fish surface mesh "
                   "(%zu triangles, dot(P-Q,n)>0, %.1f ms)\n",
                   skipped, m_init_surface_faces.size(), filter_ms);
        }
    }
    p.num_particles = (int)cpu_pos.size();

    cudaMalloc(&p.d_positions, p.num_particles * sizeof(float3));
    cudaMalloc(&p.d_velocities, p.num_particles * sizeof(float3));
    cudaMalloc(&p.d_densities, p.num_particles * sizeof(float));
    cudaMalloc(&p.d_pressures, p.num_particles * sizeof(float));
    cudaMalloc(&p.d_forces, p.num_particles * sizeof(float3));
    cudaMemcpy(p.d_positions, cpu_pos.data(), p.num_particles * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemset(p.d_velocities, 0, p.num_particles * sizeof(float3));
    cudaMemset(p.d_forces, 0, p.num_particles * sizeof(float3));
    p.current_time = 0.0f;
    p.last_ramp    = 0.0f;
    p.frame_count  = 0;
}

void PeridynoBridge::SetSubsteps(int n)
{
    m_n_substeps = (n >= 1) ? n : 1;
    if (m_impl) m_impl->n_substeps = m_n_substeps;
}

int PeridynoBridge::GetSubsteps() const { return m_n_substeps; }

void PeridynoBridge::SetContactParams(float stiffness, float damping)
{
    m_contact_stiffness = stiffness;
    m_contact_damping   = damping;
    if (m_impl) {
        m_impl->contact_stiffness = stiffness;
        m_impl->contact_damping   = damping;
    }
}

void PeridynoBridge::SetFlowRampTime(float seconds)
{
    m_flow_ramp_time = std::max(0.01f, seconds);
    if (m_impl) m_impl->flow_ramp_time = m_flow_ramp_time;
}

void PeridynoBridge::SetHydroForceScale(float scale)
{
    m_hydro_force_scale = std::max(0.0f, scale);
    if (m_impl) m_impl->hydro_force_scale = m_hydro_force_scale;
}

void PeridynoBridge::SetMaxVertexForce(float max_force)
{
    m_max_vertex_force = std::max(0.0f, max_force);
    if (m_impl) m_impl->max_vertex_force = m_max_vertex_force;
}

void PeridynoBridge::SetEnableContactRepulsion(bool enable)
{
    m_enable_contact_repulsion = enable;
}

void PeridynoBridge::SetEnableHydroPressure(bool enable)
{
    m_enable_hydro_pressure = enable;
}

FsiForceDiagnostics PeridynoBridge::ConsumeSecondFsiForceDiagnostics()
{
    FsiForceDiagnostics d;
    if (m_roll_frame_count > 0) {
        const double inv = 1.0 / static_cast<double>(m_roll_frame_count);
        d.contact_total_avg_n = m_roll_sum_contact_total * inv;
        d.hydro_total_avg_n   = m_roll_sum_hydro_total * inv;
        d.contact_peak_n      = m_roll_peak_contact;
        d.hydro_peak_n        = m_roll_peak_hydro;
        d.frames              = m_roll_frame_count;
    }
    m_roll_sum_contact_total = 0.0;
    m_roll_sum_hydro_total   = 0.0;
    m_roll_peak_contact      = 0.0;
    m_roll_peak_hydro        = 0.0;
    m_roll_frame_count       = 0;
    return d;
}

void PeridynoBridge::Reset()
{
    if (!m_impl->initialized) return;
    Destroy();
    Initialize(m_domain_min, m_domain_max, m_particle_spacing, m_flow_velocity,
               m_rest_density, m_viscosity, m_sound_speed, m_boundary_thickness,
               m_init_surface_vertices, m_init_surface_faces);
}

float PeridynoBridge::GetFlowRampFactor() const
{
    if (!m_impl || !m_impl->initialized) return 0.0f;
    return m_impl->last_ramp;
}

void PeridynoBridge::Destroy()
{
    auto& p = *m_impl;
    CUDA_SAFE_FREE(p.d_positions); CUDA_SAFE_FREE(p.d_velocities);
    CUDA_SAFE_FREE(p.d_densities); CUDA_SAFE_FREE(p.d_pressures);
    CUDA_SAFE_FREE(p.d_forces);

    CUDA_SAFE_FREE(p.d_boundary_vertices);
    CUDA_SAFE_FREE(p.d_boundary_velocities);
    CUDA_SAFE_FREE(p.d_boundary_forces_contact_sub);
    CUDA_SAFE_FREE(p.d_boundary_forces_hydro_sub);
    CUDA_SAFE_FREE(p.d_boundary_forces_contact_frame);
    CUDA_SAFE_FREE(p.d_boundary_forces_hydro_frame);

    CUDA_SAFE_FREE(p.d_face_indices);
    CUDA_SAFE_FREE(p.d_face_centers);
    CUDA_SAFE_FREE(p.d_face_normals);
    CUDA_SAFE_FREE(p.d_face_areas);
    CUDA_SAFE_FREE(p.d_face_grid_counts);
    CUDA_SAFE_FREE(p.d_face_grid_particles);

    CUDA_SAFE_FREE(p.d_grid_counts); CUDA_SAFE_FREE(p.d_grid_cell_start);
    CUDA_SAFE_FREE(p.d_grid_cell_end); CUDA_SAFE_FREE(p.d_grid_particles);

    free(p.h_boundary_forces_contact_frame); p.h_boundary_forces_contact_frame = nullptr;
    free(p.h_boundary_forces_hydro_frame);     p.h_boundary_forces_hydro_frame = nullptr;

    p.num_particles = 0;
    p.num_boundary_vertices = 0;
    p.num_faces = 0;
    p.face_grid_allocated = false;
    p.initialized = false;
    m_initialized = false;
    m_roll_sum_contact_total = 0.0;
    m_roll_sum_hydro_total   = 0.0;
    m_roll_peak_contact      = 0.0;
    m_roll_peak_hydro        = 0.0;
    m_roll_frame_count       = 0;
    FreeVapGpuBuffers();
}

/** 由三角面法向累加得到顶点法向（供 VAP ghost 粒子） */
static std::vector<float3> ComputeVertexNormals(
    int nv,
    const Eigen::VectorXd& vertices,
    const std::vector<Eigen::Vector3i>& faces)
{
    std::vector<float3> nrm(nv, make_float3(0, 0, 0));
    for (const Eigen::Vector3i& fi : faces) {
        float3 v0 = make_float3((float)vertices(3*fi[0]), (float)vertices(3*fi[0]+1), (float)vertices(3*fi[0]+2));
        float3 v1 = make_float3((float)vertices(3*fi[1]), (float)vertices(3*fi[1]+1), (float)vertices(3*fi[1]+2));
        float3 v2 = make_float3((float)vertices(3*fi[2]), (float)vertices(3*fi[2]+1), (float)vertices(3*fi[2]+2));
        float3 e0 = v1 - v0, e1 = v2 - v0;
        float3 fn = make_float3(e0.y*e1.z - e0.z*e1.y, e0.z*e1.x - e0.x*e1.z, e0.x*e1.y - e0.y*e1.x);
        for (int k = 0; k < 3; k++) {
            int vi = fi[k];
            nrm[vi].x += fn.x; nrm[vi].y += fn.y; nrm[vi].z += fn.z;
        }
    }
    for (int i = 0; i < nv; i++) nrm[i] = normalize(nrm[i]);
    return nrm;
}

void PeridynoBridge::RebuildSurfaceAdjacency(
    int nv, const std::vector<Eigen::Vector3i>& faces)
{
    m_on_surface.assign(nv, 0);
    m_surface_adj.assign(nv, {});
    for (const Eigen::Vector3i& f : faces) {
        for (int k = 0; k < 3; k++) {
            int a = f[k];
            if (a < 0 || a >= nv) continue;
            m_on_surface[a] = 1;
            for (int j = 0; j < 3; j++) {
                if (j == k) continue;
                int b = f[j];
                if (b < 0 || b >= nv || b == a) continue;
                auto& adj = m_surface_adj[a];
                if (std::find(adj.begin(), adj.end(), b) == adj.end())
                    adj.push_back(b);
            }
        }
    }
}

void PeridynoBridge::FillZeroHydroFromNeighbors(
    std::vector<float>& hx, std::vector<float>& hy, std::vector<float>& hz, int nv) const
{
    if ((int)m_on_surface.size() != nv) return;
    constexpr float kEps = 1e-10f;
    // 两轮：传播邻居非零水压到零力表面顶点（面力矢量抵消时的补救）
    for (int pass = 0; pass < 2; pass++) {
        std::vector<float> nx = hx, ny = hy, nz = hz;
        for (int i = 0; i < nv; i++) {
            if (!m_on_surface[i]) continue;
            const float hm = sqrtf(hx[i]*hx[i] + hy[i]*hy[i] + hz[i]*hz[i]);
            if (hm > kEps) continue;
            float sx = 0, sy = 0, sz = 0;
            int cnt = 0;
            for (int j : m_surface_adj[i]) {
                if (j < 0 || j >= nv) continue;
                const float nm = sqrtf(hx[j]*hx[j] + hy[j]*hy[j] + hz[j]*hz[j]);
                if (nm <= kEps) continue;
                sx += hx[j]; sy += hy[j]; sz += hz[j];
                cnt++;
            }
            if (cnt > 0) {
                nx[i] = sx / cnt;
                ny[i] = sy / cnt;
                nz[i] = sz / cnt;
            }
        }
        hx.swap(nx); hy.swap(ny); hz.swap(nz);
    }
}

namespace {

float VapHostDot(float* d_a, float* d_b, float* d_partial, float* h_partial, int n, int bs, int gs) {
    vap_dot_kernel<<<gs, bs>>>(d_a, d_b, d_partial, n);
    cudaMemcpy(h_partial, d_partial, gs * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (int i = 0; i < gs; i++) s += h_partial[i];
    return (float)s;
}

} // namespace

void PeridynoBridge::EnsureVapGpuCapacity(int num_fluid, int num_ghost)
{
    auto& p = *m_impl;
    if (!p.d_vap_band_counter)
        cudaMalloc(&p.d_vap_band_counter, sizeof(int));
    int need_merged = num_fluid + num_ghost + 16;
    if (p.vap_merged_capacity >= need_merged) return;
    p.vap_merged_capacity = need_merged;
    CUDA_SAFE_FREE(p.d_vap_band_mask);
    CUDA_SAFE_FREE(p.d_vap_prefix);
    CUDA_SAFE_FREE(p.d_vap_fluid_to_merged);
    CUDA_SAFE_FREE(p.d_vap_merged_pos);
    CUDA_SAFE_FREE(p.d_vap_merged_vel);
    CUDA_SAFE_FREE(p.d_vap_merged_nrm);
    CUDA_SAFE_FREE(p.d_vap_merged_attr);
    CUDA_SAFE_FREE(p.d_vap_merged_density);
    CUDA_SAFE_FREE(p.d_vap_neighbor_count);
    CUDA_SAFE_FREE(p.d_vap_neighbors);
    CUDA_SAFE_FREE(p.d_vap_alpha);
    CUDA_SAFE_FREE(p.d_vap_aii);
    CUDA_SAFE_FREE(p.d_vap_aii_fluid);
    CUDA_SAFE_FREE(p.d_vap_aii_total);
    CUDA_SAFE_FREE(p.d_vap_pressure);
    CUDA_SAFE_FREE(p.d_vap_divergence);
    CUDA_SAFE_FREE(p.d_vap_is_surface);
    CUDA_SAFE_FREE(p.d_vap_ax);
    CUDA_SAFE_FREE(p.d_vap_r);
    CUDA_SAFE_FREE(p.d_vap_p);
    CUDA_SAFE_FREE(p.d_vap_merged_grid_counts);
    CUDA_SAFE_FREE(p.d_vap_merged_grid_particles);
    CUDA_SAFE_FREE(p.d_vap_dot_partial);
    CUDA_SAFE_FREE(p.d_vap_band_counter);

    cudaMalloc(&p.d_vap_band_mask, num_fluid * sizeof(unsigned char));
    cudaMalloc(&p.d_vap_prefix, num_fluid * sizeof(int));
    cudaMalloc(&p.d_vap_fluid_to_merged, num_fluid * sizeof(int));
    cudaMalloc(&p.d_vap_merged_pos, need_merged * sizeof(float3));
    cudaMalloc(&p.d_vap_merged_vel, need_merged * sizeof(float3));
    cudaMalloc(&p.d_vap_merged_nrm, need_merged * sizeof(float3));
    cudaMalloc(&p.d_vap_merged_attr, need_merged * sizeof(unsigned char));
    cudaMalloc(&p.d_vap_merged_density, need_merged * sizeof(float));
    cudaMalloc(&p.d_vap_neighbor_count, need_merged * sizeof(int));
    cudaMalloc(&p.d_vap_neighbors, need_merged * MAX_VAP_NEIGHBORS * sizeof(int));
    cudaMalloc(&p.d_vap_alpha, need_merged * sizeof(float));
    cudaMalloc(&p.d_vap_aii, need_merged * sizeof(float));
    cudaMalloc(&p.d_vap_aii_fluid, need_merged * sizeof(float));
    cudaMalloc(&p.d_vap_aii_total, need_merged * sizeof(float));
    cudaMalloc(&p.d_vap_pressure, need_merged * sizeof(float));
    cudaMalloc(&p.d_vap_divergence, need_merged * sizeof(float));
    cudaMalloc(&p.d_vap_is_surface, need_merged * sizeof(unsigned char));
    cudaMalloc(&p.d_vap_ax, need_merged * sizeof(float));
    cudaMalloc(&p.d_vap_r, need_merged * sizeof(float));
    cudaMalloc(&p.d_vap_p, need_merged * sizeof(float));
    int tc = p.grid_params.total_cells;
    cudaMalloc(&p.d_vap_merged_grid_counts, tc * sizeof(int));
    cudaMalloc(&p.d_vap_merged_grid_particles, tc * VAP_MAX_MERGED_CELL * sizeof(int));
    cudaMalloc(&p.d_vap_dot_partial, 4096 * sizeof(float));
    cudaMalloc(&p.d_vap_band_counter, sizeof(int));
    p.vap_dot_partial_cap = 4096;
}

void PeridynoBridge::FreeVapGpuBuffers()
{
    auto& p = *m_impl;
    CUDA_SAFE_FREE(p.d_vap_band_mask);
    CUDA_SAFE_FREE(p.d_vap_prefix);
    CUDA_SAFE_FREE(p.d_vap_fluid_to_merged);
    CUDA_SAFE_FREE(p.d_vap_merged_pos);
    CUDA_SAFE_FREE(p.d_vap_merged_vel);
    CUDA_SAFE_FREE(p.d_vap_merged_nrm);
    CUDA_SAFE_FREE(p.d_vap_merged_attr);
    CUDA_SAFE_FREE(p.d_vap_merged_density);
    CUDA_SAFE_FREE(p.d_vap_neighbor_count);
    CUDA_SAFE_FREE(p.d_vap_neighbors);
    CUDA_SAFE_FREE(p.d_vap_alpha);
    CUDA_SAFE_FREE(p.d_vap_aii);
    CUDA_SAFE_FREE(p.d_vap_aii_fluid);
    CUDA_SAFE_FREE(p.d_vap_aii_total);
    CUDA_SAFE_FREE(p.d_vap_pressure);
    CUDA_SAFE_FREE(p.d_vap_divergence);
    CUDA_SAFE_FREE(p.d_vap_is_surface);
    CUDA_SAFE_FREE(p.d_vap_ax);
    CUDA_SAFE_FREE(p.d_vap_r);
    CUDA_SAFE_FREE(p.d_vap_p);
    CUDA_SAFE_FREE(p.d_vap_merged_grid_counts);
    CUDA_SAFE_FREE(p.d_vap_merged_grid_particles);
    CUDA_SAFE_FREE(p.d_vap_dot_partial);
    CUDA_SAFE_FREE(p.d_vap_band_counter);
    CUDA_SAFE_FREE(p.d_boundary_normals);
    p.vap_merged_capacity = 0;
    p.vap_alpha_max = 0.0f;
    p.vap_a_max = 0.0f;
}

void PeridynoBridge::RunVapSubstepImpl(float dt_sub, float ghost_time_offset, int bs)
{
    auto& p = *m_impl;
    const int N = p.num_particles;
    const int G = p.num_boundary_vertices;
    if (N <= 0 || G <= 0 || !p.face_grid_allocated || !p.d_boundary_normals) return;
    if (!p.d_boundary_velocities) return;

    EnsureVapGpuCapacity(N, G);
    const int gsN = (N + bs - 1) / bs;
    const float band_h = p.h * VAP_BAND_H_MULT;
    const int cells = p.grid_params.total_cells;
    const int cgs = (cells + bs - 1) / bs;

    vap_clear_uchar_kernel<<<gsN, bs>>>(p.d_vap_band_mask, N);
    vap_mark_band_kernel<<<gsN, bs>>>(
        p.d_vap_band_mask, p.d_positions, p.d_face_centers,
        p.d_face_grid_particles, p.face_grid_params, band_h, N, p.num_faces);

    vap_reset_counter_kernel<<<1, 1>>>(p.d_vap_band_counter);
    vap_scatter_fluid_atomic_kernel<<<gsN, bs>>>(
        p.d_positions, p.d_velocities, p.d_densities, p.d_vap_band_mask,
        p.d_vap_merged_pos, p.d_vap_merged_vel, p.d_vap_merged_nrm, p.d_vap_merged_attr,
        p.d_vap_merged_density, p.d_vap_fluid_to_merged, p.d_vap_band_counter, N);
    int n_band = 0;
    cudaMemcpy(&n_band, p.d_vap_band_counter, sizeof(int), cudaMemcpyDeviceToHost);
    const int n_merged = n_band + G;
    if (n_merged <= G) {
        p.vap_last_n_band = 0;
        return;
    }
    p.vap_last_n_band = n_band;
    vap_scatter_ghost_kernel<<<(G + bs - 1) / bs, bs>>>(
        p.d_boundary_vertices, p.d_boundary_velocities, p.d_boundary_normals,
        p.d_vap_merged_pos, p.d_vap_merged_vel, p.d_vap_merged_nrm, p.d_vap_merged_attr,
        n_band, G, ghost_time_offset);

    vap_clear_merged_grid_kernel<<<cgs, bs>>>(p.d_vap_merged_grid_counts, p.d_vap_merged_grid_particles, cells);
    vap_build_merged_grid_kernel<<<(n_merged + bs - 1) / bs, bs>>>(
        p.d_vap_merged_pos, p.d_vap_merged_grid_counts, p.d_vap_merged_grid_particles,
        p.grid_params, n_merged);
    vap_build_neighbors_kernel<<<(n_merged + bs - 1) / bs, bs>>>(
        p.d_vap_merged_pos, p.d_vap_merged_attr, p.d_vap_neighbor_count, p.d_vap_neighbors,
        p.d_vap_merged_grid_counts, p.d_vap_merged_grid_particles, p.grid_params, p.h, n_merged);

    const int gsM = (n_merged + bs - 1) / bs;
    vap_zero_kernel<<<gsM, bs>>>(p.d_vap_alpha, n_merged);
    vap_compute_alpha_kernel<<<gsM, bs>>>(
        p.d_vap_alpha, p.d_vap_merged_pos, p.d_vap_merged_attr,
        p.d_vap_neighbor_count, p.d_vap_neighbors, p.h, n_merged);

    if (p.vap_alpha_max <= 0.0f) {
        std::vector<float> h_alpha(n_merged);
        cudaMemcpy(h_alpha.data(), p.d_vap_alpha, n_merged * sizeof(float), cudaMemcpyDeviceToHost);
        float mx = 1e-6f;
        for (float v : h_alpha) mx = fmaxf(mx, v);
        p.vap_alpha_max = mx;
    }
    vap_correct_alpha_kernel2<<<gsM, bs>>>(p.d_vap_alpha, p.d_vap_merged_attr, p.vap_alpha_max, n_merged);

    vap_zero_kernel<<<gsM, bs>>>(p.d_vap_aii_fluid, n_merged);
    vap_zero_kernel<<<gsM, bs>>>(p.d_vap_aii_total, n_merged);
    vap_compute_diagonal_kernel<<<gsM, bs>>>(
        p.d_vap_aii_fluid, p.d_vap_aii_total, p.d_vap_alpha, p.d_vap_merged_pos,
        p.d_vap_merged_attr, p.d_vap_neighbor_count, p.d_vap_neighbors, p.h, n_merged);

    if (p.vap_a_max <= 0.0f) {
        std::vector<float> h_a(n_merged);
        cudaMemcpy(h_a.data(), p.d_vap_aii_fluid, n_merged * sizeof(float), cudaMemcpyDeviceToHost);
        float mx = 1e-6f;
        for (float v : h_a) mx = fmaxf(mx, v);
        p.vap_a_max = mx;
    }

    vap_zero_kernel<<<gsM, bs>>>(p.d_vap_aii, n_merged);
    vap_detect_surface_kernel<<<gsM, bs>>>(
        p.d_vap_aii, p.d_vap_is_surface, p.d_vap_aii_fluid, p.d_vap_aii_total,
        p.d_vap_merged_attr, p.d_vap_neighbor_count, p.d_vap_neighbors,
        p.vap_a_max, n_merged);

    vap_zero_kernel<<<gsM, bs>>>(p.d_vap_divergence, n_merged);
    vap_compute_divergence_kernel<<<gsM, bs>>>(
        p.d_vap_divergence, p.d_vap_alpha, p.d_vap_merged_pos, p.d_vap_merged_vel,
        p.d_vap_merged_nrm, p.d_vap_merged_attr, p.d_vap_neighbor_count, p.d_vap_neighbors,
        VAP_SEPARATION, VAP_TANGENTIAL, p.rest_density, p.h, dt_sub, n_merged);
    vap_compensate_source_kernel<<<gsM, bs>>>(
        p.d_vap_divergence, p.d_vap_merged_density, p.d_vap_merged_attr, p.rest_density, dt_sub, n_merged);

    vap_zero_kernel<<<gsM, bs>>>(p.d_vap_pressure, n_merged);
    vap_zero_kernel<<<gsM, bs>>>(p.d_vap_ax, n_merged);
    vap_compute_ax_kernel<<<gsM, bs>>>(
        p.d_vap_ax, p.d_vap_pressure, p.d_vap_aii, p.d_vap_alpha, p.d_vap_merged_pos,
        p.d_vap_merged_attr, p.d_vap_neighbor_count, p.d_vap_neighbors, p.h, n_merged);

    vap_subtract_kernel<<<gsM, bs>>>(p.d_vap_r, p.d_vap_divergence, p.d_vap_ax, n_merged);
    cudaMemcpy(p.d_vap_p, p.d_vap_r, n_merged * sizeof(float), cudaMemcpyDeviceToDevice);

    const int dot_gs = (n_merged + bs - 1) / bs;
    std::vector<float> h_part(dot_gs, 0.0f);
    float rr = VapHostDot(p.d_vap_r, p.d_vap_r, p.d_vap_dot_partial, h_part.data(), n_merged, bs, dot_gs);
    float err = sqrtf(fmaxf(rr / fmaxf(n_merged, 1), 0.0f));

    for (int it = 0; it < VAP_CG_MAX_ITER && err > VAP_CG_TOL; ++it) {
        vap_zero_kernel<<<gsM, bs>>>(p.d_vap_ax, n_merged);
        vap_compute_ax_kernel<<<gsM, bs>>>(
            p.d_vap_ax, p.d_vap_p, p.d_vap_aii, p.d_vap_alpha, p.d_vap_merged_pos,
            p.d_vap_merged_attr, p.d_vap_neighbor_count, p.d_vap_neighbors, p.h, n_merged);
        float pAy = VapHostDot(p.d_vap_p, p.d_vap_ax, p.d_vap_dot_partial, h_part.data(), n_merged, bs, dot_gs);
        if (fabsf(pAy) < 1e-12f) break;
        float alpha_cg = rr / pAy;
        vap_saxpy_kernel<<<gsM, bs>>>(p.d_vap_pressure, p.d_vap_p, alpha_cg, n_merged);
        vap_saxpy_kernel<<<gsM, bs>>>(p.d_vap_r, p.d_vap_ax, -alpha_cg, n_merged);
        float rr_old = rr;
        rr = VapHostDot(p.d_vap_r, p.d_vap_r, p.d_vap_dot_partial, h_part.data(), n_merged, bs, dot_gs);
        float beta = rr / fmaxf(rr_old, 1e-12f);
        vap_update_p_kernel<<<gsM, bs>>>(p.d_vap_p, p.d_vap_r, beta, n_merged);
        err = sqrtf(fmaxf(rr / fmaxf(n_merged, 1), 0.0f));
    }

    vap_update_velocity_kernel<<<gsM, bs>>>(
        p.d_vap_pressure, p.d_vap_aii, p.d_vap_is_surface, p.d_vap_merged_vel, p.d_vap_merged_pos,
        p.d_vap_merged_attr, p.d_vap_neighbor_count, p.d_vap_neighbors,
        p.rest_density, p.h, dt_sub, n_merged);

    vap_clamp_scale_pressure_kernel<<<gsM, bs>>>(
        p.d_vap_pressure, VAP_PRESSURE_ABS_MAX, VAP_PRESSURE_TO_HYDRO, n_merged);

    vap_scatter_velocity_back_kernel<<<gsN, bs>>>(p.d_velocities, p.d_vap_merged_vel, p.d_vap_fluid_to_merged, N);
    vap_scatter_pressure_back_kernel<<<gsN, bs>>>(p.d_pressures, p.d_vap_pressure, p.d_vap_fluid_to_merged, N);
}

/**
 * 上传 FEM 鱼体边界（每 FEM 帧调用一次）
 *
 * @param surface_vertices   全部 FEM 顶点位置 (3×N)，double→float H2D
 * @param surface_velocities 对应顶点速度
 * @param surface_faces      表面三角面索引 (14284 个面)
 *
 * 拓扑变化时：分配 GPU 缓冲、上传面索引、预计算初始面积
 * 每帧：上传最新顶点位置/速度供 FSI 使用
 */
void PeridynoBridge::SetFishBoundary(
    const Eigen::VectorXd& surface_vertices,
    const Eigen::VectorXd& surface_velocities,
    const std::vector<Eigen::Vector3i>& surface_faces)
{
    auto& p = *m_impl;
    if (!p.initialized) return;
    int nv = surface_vertices.size() / 3;
    int nf = surface_faces.size();

    // ---- 仅在顶点数或面数变化时重新分配 GPU 内存 ----
    bool vertices_changed = (nv != p.num_boundary_vertices);
    bool faces_changed    = (nf != p.num_faces);

    if (vertices_changed) {
        CUDA_SAFE_FREE(p.d_boundary_vertices);
        CUDA_SAFE_FREE(p.d_boundary_velocities);
        CUDA_SAFE_FREE(p.d_boundary_forces_contact_sub);
        CUDA_SAFE_FREE(p.d_boundary_forces_hydro_sub);
        CUDA_SAFE_FREE(p.d_boundary_forces_contact_frame);
        CUDA_SAFE_FREE(p.d_boundary_forces_hydro_frame);
        free(p.h_boundary_forces_contact_frame); p.h_boundary_forces_contact_frame = nullptr;
        free(p.h_boundary_forces_hydro_frame);     p.h_boundary_forces_hydro_frame = nullptr;

        p.num_boundary_vertices = nv;
        cudaMalloc(&p.d_boundary_vertices,   nv * sizeof(float3));
        cudaMalloc(&p.d_boundary_velocities, nv * sizeof(float3));
        cudaMalloc(&p.d_boundary_forces_contact_sub,   nv * sizeof(float3));
        cudaMalloc(&p.d_boundary_forces_hydro_sub,     nv * sizeof(float3));
        cudaMalloc(&p.d_boundary_forces_contact_frame, nv * sizeof(float3));
        cudaMalloc(&p.d_boundary_forces_hydro_frame,   nv * sizeof(float3));
        p.h_boundary_forces_contact_frame = (float3*)malloc(nv * sizeof(float3));
        p.h_boundary_forces_hydro_frame   = (float3*)malloc(nv * sizeof(float3));
        CUDA_SAFE_FREE(p.d_boundary_normals);
        cudaMalloc(&p.d_boundary_normals, nv * sizeof(float3));
        p.vap_alpha_max = 0.0f;
        p.vap_a_max = 0.0f;

        std::cout << "[PeridynoBridge] Boundary: " << nv << " vertices" << std::endl;
    }

    if (faces_changed) {
        // 上传三角面索引
        CUDA_SAFE_FREE(p.d_face_indices);
        CUDA_SAFE_FREE(p.d_face_centers);
        CUDA_SAFE_FREE(p.d_face_normals);
        CUDA_SAFE_FREE(p.d_face_areas);

        p.num_faces = nf;
        std::vector<int3> cpu_faces(nf);
        for (int i = 0; i < nf; i++) {
            cpu_faces[i] = make_int3(surface_faces[i][0], surface_faces[i][1], surface_faces[i][2]);
        }
        cudaMalloc(&p.d_face_indices, nf * sizeof(int3));
        cudaMalloc(&p.d_face_centers, nf * sizeof(float3));
        cudaMalloc(&p.d_face_normals, nf * sizeof(float3));
        cudaMalloc(&p.d_face_areas,   nf * sizeof(float));
        cudaMemcpy(p.d_face_indices, cpu_faces.data(), nf * sizeof(int3), cudaMemcpyHostToDevice);

        // 预计算三角面面积（基于初始顶点位置，CPU 计算一次即可）
        std::vector<float> cpu_face_areas(nf, 0.0f);
        double total_face_area = 0.0;
        for (int i = 0; i < nf; i++) {
            Eigen::Vector3d v0 = surface_vertices.segment<3>(3 * surface_faces[i][0]);
            Eigen::Vector3d v1 = surface_vertices.segment<3>(3 * surface_faces[i][1]);
            Eigen::Vector3d v2 = surface_vertices.segment<3>(3 * surface_faces[i][2]);
            double area = 0.5 * (v1 - v0).cross(v2 - v0).norm();
            cpu_face_areas[i] = (float)area;
            total_face_area += area;
        }
        cudaMemcpy(p.d_face_areas, cpu_face_areas.data(), nf * sizeof(float), cudaMemcpyHostToDevice);
        std::cout << "[PeridynoBridge] Faces: " << nf << " triangles"
                  << ", total surface area=" << total_face_area << " m^2" << std::endl;

        // 构建面网格参数（面心变化后每帧重建）
        float3 b_min = {(float)m_domain_min.x(), (float)m_domain_min.y(), (float)m_domain_min.z()};
        float  f_cell_size = p.particle_spacing;  // 面网格 cellsize = particle_spacing
        p.face_grid_params.grid_min = b_min - make_float3(f_cell_size, f_cell_size, f_cell_size);
        p.face_grid_params.cell_size = f_cell_size;
        p.face_grid_params.inv_cell_size = 1.0f / f_cell_size;
        float3 b_max = {(float)m_domain_max.x(), (float)m_domain_max.y(), (float)m_domain_max.z()};
        float3 fext = b_max - b_min + make_float3(2 * f_cell_size, 2 * f_cell_size, 2 * f_cell_size);
        p.face_grid_params.grid_dim.x = max(1, (int)(fext.x * p.face_grid_params.inv_cell_size) + 1);
        p.face_grid_params.grid_dim.y = max(1, (int)(fext.y * p.face_grid_params.inv_cell_size) + 1);
        p.face_grid_params.grid_dim.z = max(1, (int)(fext.z * p.face_grid_params.inv_cell_size) + 1);
        int fcells = p.face_grid_params.grid_dim.x * p.face_grid_params.grid_dim.y * p.face_grid_params.grid_dim.z;
        p.face_grid_params.total_cells = fcells;

        CUDA_SAFE_FREE(p.d_face_grid_counts);
        CUDA_SAFE_FREE(p.d_face_grid_particles);
        cudaMalloc(&p.d_face_grid_counts, fcells * sizeof(int));
        cudaMalloc(&p.d_face_grid_particles, fcells * MAX_FACES_PER_CELL * sizeof(int));
        p.face_grid_allocated = true;
        RebuildSurfaceAdjacency(nv, surface_faces);
    }
    if (m_on_surface.empty() && nf > 0)
        RebuildSurfaceAdjacency(nv, surface_faces);

    // ---- 上传顶点位置和速度（每帧）----
    std::vector<float3> cpu_v(nv), cpu_vel(nv);
    for (int i = 0; i < nv; i++) {
        cpu_v[i]   = {(float)surface_vertices(3*i),   (float)surface_vertices(3*i+1),   (float)surface_vertices(3*i+2)};
        cpu_vel[i] = {(float)surface_velocities(3*i),  (float)surface_velocities(3*i+1), (float)surface_velocities(3*i+2)};
    }
    cudaMemcpy(p.d_boundary_vertices,   cpu_v.data(),   nv * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(p.d_boundary_velocities, cpu_vel.data(), nv * sizeof(float3), cudaMemcpyHostToDevice);

    if (p.d_boundary_normals && nf > 0) {
        std::vector<float3> vtx_nrm = ComputeVertexNormals(nv, surface_vertices, surface_faces);
        cudaMemcpy(p.d_boundary_normals, vtx_nrm.data(), nv * sizeof(float3), cudaMemcpyHostToDevice);
    }

    m_last_fsi_snapshot.positions = surface_vertices;
}

/**
 * 核心：执行 n_substeps 次 SPH 子步，返回 FEM 顶点上的时间平均流体力
 *
 * @param dt  FEM 时间步长 [s]（Environment 中 dt = 1/240 s）
 * @return    长度 3×N 的 Eigen::VectorXd [N]，单位 [N]，直接作为 FEM 外力
 *
 * 后处理顺序：
 *   1. 子步力求和 → 除以 n_substeps（时间平均）
 *   2. 若 max_vertex_force>0，按顶点截断力模长
 *   3. 若 ramp<1，再乘 ramp 缩放（与子步内 ramp 叠加，需注意）
 */
Eigen::VectorXd PeridynoBridge::ComputeFluidForces(float dt)
{
    auto& p = *m_impl;
    if (!p.initialized) return Eigen::VectorXd();

    int N  = p.num_particles;
    int bs = 256;
    int gs = (N + bs - 1) / bs;
    int cells = p.grid_params.total_cells;
    int cgs   = (cells + bs - 1) / bs;

    int n_substeps = p.n_substeps;
    float dt_sub   = dt / (float)n_substeps;

    // ---- 更新三角面数据（每帧一次）----
    if (p.num_faces > 0) {
        int fgs = (p.num_faces + bs - 1) / bs;
        update_face_data_kernel<<<fgs, bs>>>(
            p.d_boundary_vertices, p.d_face_indices,
            p.d_face_centers, p.d_face_normals, p.num_faces);

        // 重建三角面网格（面心在帧内不变）
        int fcells = p.face_grid_params.total_cells;
        clear_counts_kernel<<<(fcells + bs - 1) / bs, bs>>>(
            p.d_face_grid_counts, p.d_face_grid_particles, fcells, MAX_FACES_PER_CELL);
        build_face_grid_kernel<<<fgs, bs>>>(
            p.d_face_centers,
            p.d_face_grid_counts, p.d_face_grid_particles,
            p.face_grid_params, p.num_faces);
    }

    // ---- 清零帧累积力（接触 / 水压力分轨）----
    if (p.num_boundary_vertices > 0) {
        int nv_clear = p.num_boundary_vertices;
        clear_forces_kernel<<<(nv_clear + bs - 1) / bs, bs>>>(
            p.d_boundary_forces_contact_frame, nv_clear);
        clear_forces_kernel<<<(nv_clear + bs - 1) / bs, bs>>>(
            p.d_boundary_forces_hydro_frame, nv_clear);
    }

    // ---- 子步进循环：显式弱耦合 FSI，帧末对边界力做时间平均 ----
#if USE_VAP_HYDRO
    const bool use_vap_hydro = m_enable_hydro_pressure;
#else
    const bool use_vap_hydro = false;
#endif
    const float mass_inv = 1.0f / (p.mass + 1e-6f);
    const bool vap_hybrid_sph = use_vap_hydro && p.num_faces > 0 && p.num_boundary_vertices > 0;
    if (vap_hybrid_sph)
        EnsureVapGpuCapacity(N, p.num_boundary_vertices);

    for (int sub = 0; sub < n_substeps; sub++) {
        clear_forces_kernel<<<gs, bs>>>(p.d_forces, N);
        clear_counts_kernel<<<cgs, bs>>>(p.d_grid_counts, p.d_grid_particles, cells, MAX_PARTICLES_PER_CELL);
        build_grid_kernel<<<gs, bs>>>(p.d_positions, p.d_grid_counts, p.d_grid_particles, p.grid_params, N);

        compute_density_pressure_kernel<<<gs, bs>>>(
            p.d_positions, p.d_densities, p.d_pressures,
            p.d_grid_particles, p.grid_params,
            p.h, p.h2, p.mass, p.rest_density, p.sound_speed, N);

        if (use_vap_hydro) {
            // band 外 WCSPH 压力+粘性；band 内仅粘性 → VAP 投影供面力
            if (vap_hybrid_sph && p.d_vap_band_mask && p.face_grid_allocated) {
                const float band_h = p.h * VAP_BAND_H_MULT;
                vap_clear_uchar_kernel<<<gs, bs>>>(p.d_vap_band_mask, N);
                vap_mark_band_kernel<<<gs, bs>>>(
                    p.d_vap_band_mask, p.d_positions, p.d_face_centers,
                    p.d_face_grid_particles, p.face_grid_params, band_h, N, p.num_faces);
                compute_sph_forces_hybrid_kernel<<<gs, bs>>>(
                    p.d_positions, p.d_velocities,
                    p.d_densities, p.d_pressures,
                    p.d_vap_band_mask,
                    p.d_forces, p.d_grid_particles, p.grid_params,
                    p.h, p.mass, p.rest_density, p.viscosity, p.gravity_y, N);
            } else {
                compute_viscosity_forces_kernel<<<gs, bs>>>(
                    p.d_positions, p.d_velocities, p.d_densities, p.d_forces,
                    p.d_grid_particles, p.grid_params,
                    p.h, p.mass, p.viscosity, p.gravity_y, N);
            }
            integrate_velocity_predictor_kernel<<<gs, bs>>>(
                p.d_velocities, p.d_forces, mass_inv, dt_sub, N);

            cudaMemset(p.d_pressures, 0, N * sizeof(float));
            const float ghost_t = (float)sub * dt_sub;
            RunVapSubstepImpl(dt_sub, ghost_t, bs);
        } else {
            compute_sph_forces_kernel<<<gs, bs>>>(
                p.d_positions, p.d_velocities,
                p.d_densities, p.d_pressures,
                p.d_forces, p.d_grid_particles, p.grid_params,
                p.h, p.mass, p.rest_density, p.viscosity, p.gravity_y, N);
        }

        if (p.num_boundary_vertices > 0 && p.num_faces > 0 && p.face_grid_allocated) {
            int nv = p.num_boundary_vertices;
            clear_forces_kernel<<<(nv + bs - 1) / bs, bs>>>(
                p.d_boundary_forces_contact_sub, nv);
            clear_forces_kernel<<<(nv + bs - 1) / bs, bs>>>(
                p.d_boundary_forces_hydro_sub, nv);

#if ENABLE_CONTACT_REPULSION
            if (m_enable_contact_repulsion) {
                if (use_vap_hydro) clear_forces_kernel<<<gs, bs>>>(p.d_forces, N);
                compute_triangle_repulsion_kernel<<<gs, bs>>>(
                    p.d_positions, p.d_velocities,
                    p.d_forces,
                    p.d_boundary_vertices, p.d_boundary_velocities,
                    p.d_face_indices, p.d_face_normals,
                    p.d_boundary_forces_contact_sub,
                    p.d_face_grid_particles, p.face_grid_params,
                    p.particle_spacing*1.5f,
                    p.contact_stiffness,
                    p.contact_damping,
                    p.num_faces, N);
                if (use_vap_hydro) {
                    integrate_velocity_predictor_kernel<<<gs, bs>>>(
                        p.d_velocities, p.d_forces, mass_inv, dt_sub, N);
                }
            }
#endif

#if ENABLE_HYDRO_PRESSURE
            if (m_enable_hydro_pressure) {
                if (use_vap_hydro) {
                    // VAP 模式：从合并网格采样压力（避免零压力粒子稀释）
                    compute_vap_hydro_pressure_kernel<<<(p.num_faces + bs - 1) / bs, bs>>>(
                        p.d_vap_merged_pos, p.d_vap_pressure, p.d_vap_merged_attr,
                        p.d_boundary_vertices,
                        p.d_face_indices, p.d_face_centers, p.d_face_normals, p.d_face_areas,
                        p.d_boundary_forces_hydro_sub,
                        p.d_vap_merged_grid_counts, p.d_vap_merged_grid_particles,
                        p.grid_params,
                        p.h, p.hydro_force_scale,
                        p.num_faces, p.vap_last_n_band);
                } else {
                    compute_hydro_pressure_kernel<<<(p.num_faces + bs - 1) / bs, bs>>>(
                        p.d_positions, p.d_pressures,
                        p.d_forces,
                        p.d_boundary_vertices,
                        p.d_face_indices, p.d_face_centers, p.d_face_normals, p.d_face_areas,
                        p.d_boundary_forces_hydro_sub,
                        p.d_grid_particles, p.grid_params,
                        p.h, p.rest_density, p.hydro_force_scale,
                        p.num_faces, N);
                }
            }
#endif

            accumulate_forces_kernel<<<(nv + bs - 1) / bs, bs>>>(
                p.d_boundary_forces_contact_sub, p.d_boundary_forces_contact_frame, nv);
            accumulate_forces_kernel<<<(nv + bs - 1) / bs, bs>>>(
                p.d_boundary_forces_hydro_sub, p.d_boundary_forces_hydro_frame, nv);
        }

        p.current_time += dt_sub;
        float ramp  = fminf(p.current_time / p.flow_ramp_time, 1.0f);
        float3 ramped_flow = make_float3(p.flow_velocity.x * ramp,
                                         p.flow_velocity.y * ramp,
                                         p.flow_velocity.z * ramp);

        constrain_domain_kernel<<<gs, bs>>>(
            p.d_positions, p.d_velocities,
            p.domain_min_f3, p.domain_max_f3,
            ramped_flow, 0.5f, p.h * 0.5f, N);

        apply_ramped_flow_kernel<<<gs, bs>>>(
            p.d_velocities, ramped_flow, ramp, N);

        if (use_vap_hydro) {
            integrate_position_kernel<<<gs, bs>>>(p.d_positions, p.d_velocities, dt_sub, N);
        } else {
            integrate_kernel<<<gs, bs>>>(
                p.d_positions, p.d_velocities,
                p.d_forces, p.d_densities,
                dt_sub, 0.999f, mass_inv, N);
        }
    }

    p.frame_count++;
    p.last_ramp = fminf(p.current_time / p.flow_ramp_time, 1.0f);

    // ---- D2H：分轨时间平均 → 合并 → 顶点截断 → ramp → 返回 FEM ----
    Eigen::VectorXd result = Eigen::VectorXd::Zero(3 * p.num_boundary_vertices);
    if (p.num_boundary_vertices > 0 &&
        p.d_boundary_forces_contact_frame && p.d_boundary_forces_hydro_frame) {
        cudaDeviceSynchronize();
        cudaMemcpy(p.h_boundary_forces_contact_frame, p.d_boundary_forces_contact_frame,
                   p.num_boundary_vertices * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(p.h_boundary_forces_hydro_frame, p.d_boundary_forces_hydro_frame,
                   p.num_boundary_vertices * sizeof(float3), cudaMemcpyDeviceToHost);

        const float avg_scale = 1.0f / static_cast<float>(n_substeps);
        const float force_scale = p.last_ramp;

        const int nv = p.num_boundary_vertices;
        m_last_fsi_snapshot.contact.resize(3 * nv);
        m_last_fsi_snapshot.hydro.resize(3 * nv);
        m_last_fsi_snapshot.total.resize(3 * nv);
        m_last_fsi_snapshot.flow_ramp = p.last_ramp;
        m_last_fsi_snapshot.valid = true;

        float frame_contact_total = 0.0f;
        float frame_hydro_total   = 0.0f;
        float frame_contact_peak  = 0.0f;
        float frame_hydro_peak    = 0.0f;
        float total_force_mag     = 0.0f;
        float max_force           = 0.0f;

        std::vector<float> hx(nv), hy(nv), hz(nv);
        std::vector<float> cx(nv), cy(nv), cz(nv);
        for (int i = 0; i < nv; i++) {
            cx[i] = p.h_boundary_forces_contact_frame[i].x * avg_scale * force_scale;
            cy[i] = p.h_boundary_forces_contact_frame[i].y * avg_scale * force_scale;
            cz[i] = p.h_boundary_forces_contact_frame[i].z * avg_scale * force_scale;
            hx[i] = p.h_boundary_forces_hydro_frame[i].x * avg_scale * force_scale;
            hy[i] = p.h_boundary_forces_hydro_frame[i].y * avg_scale * force_scale;
            hz[i] = p.h_boundary_forces_hydro_frame[i].z * avg_scale * force_scale;
        }
        for (int i = 0; i < p.num_boundary_vertices; i++) {
            const float contact_mag = sqrtf(cx[i] * cx[i] + cy[i] * cy[i] + cz[i] * cz[i]);
            const float hydro_mag   = sqrtf(hx[i] * hx[i] + hy[i] * hy[i] + hz[i] * hz[i]);
            frame_contact_total += contact_mag;
            frame_hydro_total   += hydro_mag;
            if (contact_mag > frame_contact_peak) frame_contact_peak = contact_mag;
            if (hydro_mag > frame_hydro_peak)     frame_hydro_peak = hydro_mag;

            float fx = cx[i] + hx[i];
            float fy = cy[i] + hy[i];
            float fz = cz[i] + hz[i];

            if (p.max_vertex_force > 0.0f) {
                const float fm = sqrtf(fx * fx + fy * fy + fz * fz);
                if (fm > p.max_vertex_force) {
                    const float clip = p.max_vertex_force / fmaxf(fm, 1e-12f);
                    fx *= clip;
                    fy *= clip;
                    fz *= clip;
                }
            }

            result(3 * i)     = fx;
            result(3 * i + 1) = fy;
            result(3 * i + 2) = fz;

            m_last_fsi_snapshot.contact(3 * i)     = cx[i];
            m_last_fsi_snapshot.contact(3 * i + 1) = cy[i];
            m_last_fsi_snapshot.contact(3 * i + 2) = cz[i];
            m_last_fsi_snapshot.hydro(3 * i)       = hx[i];
            m_last_fsi_snapshot.hydro(3 * i + 1)   = hy[i];
            m_last_fsi_snapshot.hydro(3 * i + 2)   = hz[i];
            m_last_fsi_snapshot.total(3 * i)       = fx;
            m_last_fsi_snapshot.total(3 * i + 1)   = fy;
            m_last_fsi_snapshot.total(3 * i + 2)   = fz;

            const float fm = sqrtf(fx * fx + fy * fy + fz * fz);
            total_force_mag += fm;
            if (fm > max_force) max_force = fm;
        }

        m_roll_sum_contact_total += frame_contact_total;
        m_roll_sum_hydro_total   += frame_hydro_total;
        if (frame_contact_peak > m_roll_peak_contact) m_roll_peak_contact = frame_contact_peak;
        if (frame_hydro_peak > m_roll_peak_hydro)     m_roll_peak_hydro = frame_hydro_peak;
        ++m_roll_frame_count;

        if (p.frame_count % 100 == 1) {
            const float ramp = p.last_ramp;
            const float3 rflow = make_float3(p.flow_velocity.x * ramp,
                                             p.flow_velocity.y * ramp,
                                             p.flow_velocity.z * ramp);
            printf("[SPH t=%.2f ramp=%.2f flow_z=%.3f frame=%d sub=%d] "
                   "contact(h=%.3f k=%.0f) hydro force=%.1f N max_vtx=%.4f N\n",
                   p.current_time, ramp, rflow.z, p.frame_count, n_substeps,
                   p.particle_spacing, p.contact_stiffness,
                   total_force_mag, max_force);
        }
    }

    return result;
}

int PeridynoBridge::GetFluidParticleCount() const { return m_impl->num_particles; }

void PeridynoBridge::GetFluidParticles(std::vector<float>& positions) const
{
    auto& p = *m_impl;
    if (!p.initialized || p.num_particles == 0) return;
    positions.resize(p.num_particles * 3);
    std::vector<float3> cpu(p.num_particles);
    cudaMemcpy(cpu.data(), p.d_positions, p.num_particles * sizeof(float3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < p.num_particles; i++) {
        positions[3*i] = cpu[i].x; positions[3*i+1] = cpu[i].y; positions[3*i+2] = cpu[i].z;
    }
}

void PeridynoBridge::GetFluidVelocities(std::vector<float>& velocities) const
{
    auto& p = *m_impl;
    if (!p.initialized || p.num_particles == 0) return;
    velocities.resize(p.num_particles * 3);
    std::vector<float3> cpu(p.num_particles);
    cudaMemcpy(cpu.data(), p.d_velocities, p.num_particles * sizeof(float3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < p.num_particles; i++) {
        velocities[3*i] = cpu[i].x; velocities[3*i+1] = cpu[i].y; velocities[3*i+2] = cpu[i].z;
    }
}
