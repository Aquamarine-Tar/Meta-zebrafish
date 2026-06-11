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
 *      a. 重建流体均匀网格 → 密度/压力 → SPH 内力
 *      b. 三角面排斥（非穿透）+ 面基水压力积分 → 边界力
 *      c. 域边界约束 + 来流 ramp → 显式积分
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

// ===== 编译期开关 =====
// ENABLE_HYDRO_PRESSURE: 是否启用水压力积分力（1=启用，0=仅接触排斥）
#define ENABLE_HYDRO_PRESSURE 1

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

    // ---- FSI 边界力缓冲 (GPU/CPU) ----
    float3* d_boundary_forces_sub   = nullptr;  // 单个子步内累积的 FEM 顶点力
    float3* d_boundary_forces_frame = nullptr;  // 整帧所有子步力之和（帧末再除以 n_substeps）
    float3* h_boundary_forces_frame = nullptr;  // D2H 后 CPU 侧处理（平均/截断/ramp）

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

// ==================== 建议一：三角形最近点 + 重心坐标力分配 ====================

/**
 * 计算点 P 到三角形 ABC 的最近点 Q 及重心坐标 (u,v,w)
 *   Q = u·A + v·B + w·C,  u+v+w=1
 * 算法：Möller, "Fast Minimum Distance Between a Point and a Triangle"
 * 用途：将接触力按 (u,v,w) 分配到三个 FEM 顶点
 */
__device__ void closest_point_on_triangle(
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

#define CUDA_SAFE_FREE(ptr) do { if (ptr) { cudaFree(ptr); ptr = nullptr; } } while(0)

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
 * @param exclusion_*        鱼体 AABB 排除区，避免流体粒子初始嵌入鱼体
 */
void PeridynoBridge::Initialize(
    const Eigen::Vector3d& domain_min, const Eigen::Vector3d& domain_max,
    float particle_spacing,
    const Eigen::Vector3d& flow_velocity,
    float rest_density, float viscosity,
    float sound_speed, float boundary_thickness,
    const Eigen::Vector3d& exclusion_min,
    const Eigen::Vector3d& exclusion_max,
    float exclusion_margin,
    bool use_exclusion)
{
    if (m_impl->initialized) Destroy();

    m_domain_min = domain_min; m_domain_max = domain_max;
    m_particle_spacing = particle_spacing;
    m_flow_velocity = flow_velocity;
    m_rest_density = rest_density; m_viscosity = viscosity;
    m_sound_speed = sound_speed; m_boundary_thickness = boundary_thickness;
    m_exclusion_min = exclusion_min;
    m_exclusion_max = exclusion_max;
    m_exclusion_margin = exclusion_margin;
    m_use_exclusion = use_exclusion;

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
              << std::endl;
}

/** 在规则网格上生成流体粒子，跳过鱼体 exclusion AABB 内的格点 */
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
    float excl_xmin = (float)m_exclusion_min.x() - m_exclusion_margin;
    float excl_xmax = (float)m_exclusion_max.x() + m_exclusion_margin;
    float excl_ymin = (float)m_exclusion_min.y() - m_exclusion_margin;
    float excl_ymax = (float)m_exclusion_max.y() + m_exclusion_margin;
    float excl_zmin = (float)m_exclusion_min.z() - m_exclusion_margin;
    float excl_zmax = (float)m_exclusion_max.z() + m_exclusion_margin;
    int idx = 0, skipped = 0;
    for (int iz = 0; iz < nz; iz++)
        for (int iy = 0; iy < ny; iy++)
            for (int ix = 0; ix < nx; ix++) {
                float x = (float)p.domain_min.x() + ix * sp + sp * 0.5f;
                float y = (float)p.domain_min.y() + iy * sp + sp * 0.5f;
                float z = (float)p.domain_min.z() + iz * sp + sp * 0.5f;
                if (m_use_exclusion &&
                    x > excl_xmin && x < excl_xmax &&
                    y > excl_ymin && y < excl_ymax &&
                    z > excl_zmin && z < excl_zmax) {
                    skipped++;
                    continue;
                }
                cpu_pos[idx++] = {x + 0.0001f * (rand() % 100 - 50),
                                  y + 0.0001f * (rand() % 100 - 50),
                                  z + 0.0001f * (rand() % 100 - 50)};
            }
    p.num_particles = idx;
    if (skipped > 0)
        printf("[PeridynoBridge] Excluded %d particles inside exclusion AABB "
               "[%.3f,%.3f,%.3f]-[%.3f,%.3f,%.3f] margin=%.3f\n",
               skipped, excl_xmin, excl_ymin, excl_zmin,
               excl_xmax, excl_ymax, excl_zmax, m_exclusion_margin);

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

void PeridynoBridge::Reset()
{
    if (!m_impl->initialized) return;
    Destroy();
    Initialize(m_domain_min, m_domain_max, m_particle_spacing, m_flow_velocity,
               m_rest_density, m_viscosity, m_sound_speed, m_boundary_thickness,
               m_exclusion_min, m_exclusion_max, m_exclusion_margin, m_use_exclusion);
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
    CUDA_SAFE_FREE(p.d_boundary_forces_sub);
    CUDA_SAFE_FREE(p.d_boundary_forces_frame);

    CUDA_SAFE_FREE(p.d_face_indices);
    CUDA_SAFE_FREE(p.d_face_centers);
    CUDA_SAFE_FREE(p.d_face_normals);
    CUDA_SAFE_FREE(p.d_face_areas);
    CUDA_SAFE_FREE(p.d_face_grid_counts);
    CUDA_SAFE_FREE(p.d_face_grid_particles);

    CUDA_SAFE_FREE(p.d_grid_counts); CUDA_SAFE_FREE(p.d_grid_cell_start);
    CUDA_SAFE_FREE(p.d_grid_cell_end); CUDA_SAFE_FREE(p.d_grid_particles);

    free(p.h_boundary_forces_frame); p.h_boundary_forces_frame = nullptr;

    p.num_particles = 0;
    p.num_boundary_vertices = 0;
    p.num_faces = 0;
    p.face_grid_allocated = false;
    p.initialized = false;
    m_initialized = false;
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
        CUDA_SAFE_FREE(p.d_boundary_forces_sub);
        CUDA_SAFE_FREE(p.d_boundary_forces_frame);
        free(p.h_boundary_forces_frame); p.h_boundary_forces_frame = nullptr;

        p.num_boundary_vertices = nv;
        cudaMalloc(&p.d_boundary_vertices,   nv * sizeof(float3));
        cudaMalloc(&p.d_boundary_velocities, nv * sizeof(float3));
        cudaMalloc(&p.d_boundary_forces_sub,   nv * sizeof(float3));
        cudaMalloc(&p.d_boundary_forces_frame, nv * sizeof(float3));
        p.h_boundary_forces_frame = (float3*)malloc(nv * sizeof(float3));

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
    }

    // ---- 上传顶点位置和速度（每帧）----
    std::vector<float3> cpu_v(nv), cpu_vel(nv);
    for (int i = 0; i < nv; i++) {
        cpu_v[i]   = {(float)surface_vertices(3*i),   (float)surface_vertices(3*i+1),   (float)surface_vertices(3*i+2)};
        cpu_vel[i] = {(float)surface_velocities(3*i),  (float)surface_velocities(3*i+1), (float)surface_velocities(3*i+2)};
    }
    cudaMemcpy(p.d_boundary_vertices,   cpu_v.data(),   nv * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(p.d_boundary_velocities, cpu_vel.data(), nv * sizeof(float3), cudaMemcpyHostToDevice);
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
 *   3. 若 ramp<1，再乘 ramp² 缩放（与子步内 ramp 叠加，需注意）
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

    // ---- 清零帧累积力 ----
    if (p.num_boundary_vertices > 0) {
        clear_forces_kernel<<<(p.num_boundary_vertices + bs - 1) / bs, bs>>>(
            p.d_boundary_forces_frame, p.num_boundary_vertices);
    }

    // ---- 子步进循环：显式弱耦合 FSI，帧末对边界力做时间平均 ----
    for (int sub = 0; sub < n_substeps; sub++) {
        // 清除本子步的流体力和网格
        clear_forces_kernel<<<gs, bs>>>(p.d_forces, N);
        clear_counts_kernel<<<cgs, bs>>>(p.d_grid_counts, p.d_grid_particles, cells, MAX_PARTICLES_PER_CELL);

        // 构建流体网格
        build_grid_kernel<<<gs, bs>>>(p.d_positions, p.d_grid_counts, p.d_grid_particles, p.grid_params, N);

        // 密度 + 压力
        compute_density_pressure_kernel<<<gs, bs>>>(
            p.d_positions, p.d_densities, p.d_pressures,
            p.d_grid_particles, p.grid_params,
            p.h, p.h2, p.mass, p.rest_density, p.sound_speed, N);

        // SPH 流体内力
        compute_sph_forces_kernel<<<gs, bs>>>(
            p.d_positions, p.d_velocities,
            p.d_densities, p.d_pressures,
            p.d_forces, p.d_grid_particles, p.grid_params,
            p.h, p.mass, p.rest_density, p.viscosity, p.gravity_y, N);

        // ==== 边界力（三角面基元）====
        if (p.num_boundary_vertices > 0 && p.num_faces > 0 && p.face_grid_allocated) {
            int nv = p.num_boundary_vertices;

            // 清除本子步边界力
            clear_forces_kernel<<<(nv + bs - 1) / bs, bs>>>(
                p.d_boundary_forces_sub, nv);

            // 非穿透接触排斥力（建议一 + 建议六前半）
            // 面网格已在帧首构建，不重复构建
            compute_triangle_repulsion_kernel<<<gs, bs>>>(
                p.d_positions, p.d_velocities,
                p.d_forces,
                p.d_boundary_vertices, p.d_boundary_velocities,
                p.d_face_indices, p.d_face_normals,
                p.d_boundary_forces_sub,
                p.d_face_grid_particles, p.face_grid_params,
                p.particle_spacing,   // repulsion_h = particle_spacing
                p.contact_stiffness,
                p.contact_damping,
                p.num_faces, N);

#if ENABLE_HYDRO_PRESSURE
            // 流体压力积分力（建议六后半）
            // 使用当前子步的流体网格和压力
            compute_hydro_pressure_kernel<<<(p.num_faces + bs - 1) / bs, bs>>>(
                p.d_positions, p.d_pressures,
                p.d_forces,
                p.d_boundary_vertices,
                p.d_face_indices, p.d_face_centers, p.d_face_normals, p.d_face_areas,
                p.d_boundary_forces_sub,
                p.d_grid_particles, p.grid_params,
                p.h, p.rest_density, p.hydro_force_scale,
                p.num_faces, N);
#endif

            // 累积到帧级力缓冲区
            accumulate_forces_kernel<<<(nv + bs - 1) / bs, bs>>>(
                p.d_boundary_forces_sub, p.d_boundary_forces_frame, nv);
        }

        // 来流 ramp：ramp = min(t / T_ramp, 1)，子步内逐步加速流体
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

        // 积分
        float mass_inv = 1.0f / (p.mass + 1e-6f);
        integrate_kernel<<<gs, bs>>>(
            p.d_positions, p.d_velocities,
            p.d_forces, p.d_densities,
            dt_sub, 0.999f, mass_inv, N);
    }

    p.frame_count++;
    p.last_ramp = fminf(p.current_time / p.flow_ramp_time, 1.0f);

    // ---- D2H：子步力求平均 → 顶点截断 → ramp² 二次缩放 → 返回给 FEM ----
    Eigen::VectorXd result = Eigen::VectorXd::Zero(3 * p.num_boundary_vertices);
    if (p.num_boundary_vertices > 0 && p.d_boundary_forces_frame) {
        cudaDeviceSynchronize();
        cudaMemcpy(p.h_boundary_forces_frame, p.d_boundary_forces_frame,
                   p.num_boundary_vertices * sizeof(float3), cudaMemcpyDeviceToHost);

        float avg_scale = 1.0f / (float)n_substeps;
        float total_force_mag = 0.0f;
        float max_force = 0.0f;
        float max_force_u = 0.0f;  // 未平均的最大力（用于诊断）

        for (int i = 0; i < p.num_boundary_vertices; i++) {
            float fx = p.h_boundary_forces_frame[i].x;
            float fy = p.h_boundary_forces_frame[i].y;
            float fz = p.h_boundary_forces_frame[i].z;

            float fm_raw = sqrtf(fx * fx + fy * fy + fz * fz);
            if (fm_raw > max_force_u) max_force_u = fm_raw;

            // 时间平均
            fx *= avg_scale;
            fy *= avg_scale;
            fz *= avg_scale;

            if (p.max_vertex_force > 0.0f) {
                float fm_avg = sqrtf(fx * fx + fy * fy + fz * fz);
                if (fm_avg > p.max_vertex_force) {
                    float clip = p.max_vertex_force / fmaxf(fm_avg, 1e-12f);
                    fx *= clip;
                    fy *= clip;
                    fz *= clip;
                }
            }

            result(3*i)   = fx;
            result(3*i+1) = fy;
            result(3*i+2) = fz;

            float fm = sqrtf(fx * fx + fy * fy + fz * fz);
            total_force_mag += fm;
            if (fm > max_force) max_force = fm;
        }

        // 来流 ramp 缩放水动力
        float flow_ramp = p.last_ramp;
        float force_scale = flow_ramp * flow_ramp;
        if (force_scale < 1.0f)
            result *= force_scale;

        if (p.frame_count % 100 == 1) {
            float ramp  = p.last_ramp;
            float3 rflow = make_float3(p.flow_velocity.x * ramp,
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
