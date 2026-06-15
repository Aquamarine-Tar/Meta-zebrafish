// VAP 窄带投影 + ghost 边界粒子（由 PeridynoBridge.cu include）
#ifndef PERIDYNO_VAP_HYDRO_INL
#define PERIDYNO_VAP_HYDRO_INL

#ifndef USE_VAP_HYDRO
#define USE_VAP_HYDRO 1
#endif

#define MAX_VAP_NEIGHBORS   128
#define VAP_GHOST_NEIGHBOR_RESERVE  16
#define VAP_BAND_H_MULT     5.0f
#define VAP_CG_MAX_ITER     50
#define VAP_CG_TOL          0.1f
#define VAP_SEPARATION      0.1f
#define VAP_TANGENTIAL      0.1f
#define VAP_MAX_MERGED_CELL 96

// 粒子属性：0=流体（压力 DOF）；2=运动学 ghost（非 DOF，但 pos/vel 来自 FEM，参与 BC）
#define VAP_ATTR_FLUID             0u
#define VAP_ATTR_KINEMATIC_GHOST   2u
// VAP 压力场 → 面力积分前的全局缩放（Tait 与投影压量纲不同，需远小于 1）
#define VAP_PRESSURE_TO_HYDRO      1.0f
#define VAP_PRESSURE_ABS_MAX       500000.0f
/** ghost 邻居在 AiiTotal 中的对角权重（Peridyno 静态壁默认 2.0） */
#define VAP_GHOST_DIAG_COEFF       1.0f

__device__ float vap_poly6_w(float r2, float h, float h2) {
    if (r2 >= h2) return 0.0f;
    float d = h2 - r2;
    return (315.0f / (64.0f * M_PI * powf(h, 9))) * d * d * d;
}

__device__ float vap_spiky_grad_mag(float r, float h) {
    if (r < 1e-8f || r >= h) return 0.0f;
    float diff = h - r;
    return (45.0f / (M_PI * h * h * h * h * h * h)) * diff * diff;
}

__device__ float vap_wrr(float r, float h) {
    if (r < 1e-8f || r >= h) return 0.0f;
    return vap_poly6_w(r * r, h, h * h) / (r * r + 1e-8f);
}

__global__ void vap_mark_band_kernel(
    unsigned char* band_mask,
    const float3* fluid_pos,
    const float3* face_centers,
    const int* face_grid_particles,
    GridParams fgp,
    float band_h,
    int num_fluid,
    int num_faces)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fluid) return;
    float3 p = fluid_pos[idx];
    // 面网格 cell_size = particle_spacing = Δx ≈ h/2
    // band_h = h * VAP_BAND_H_MULT ≈ 5Δx → 需搜索 ±ceil(band_h/cell_size) ≈ ±5 格
    int search_r = (int)ceilf(band_h * fgp.inv_cell_size) + 1;
    int cx = max(0, min((int)((p.x - fgp.grid_min.x) * fgp.inv_cell_size), fgp.grid_dim.x - 1));
    int cy = max(0, min((int)((p.y - fgp.grid_min.y) * fgp.inv_cell_size), fgp.grid_dim.y - 1));
    int cz = max(0, min((int)((p.z - fgp.grid_min.z) * fgp.inv_cell_size), fgp.grid_dim.z - 1));
    for (int dz = -search_r; dz <= search_r; dz++)
        for (int dy = -search_r; dy <= search_r; dy++)
            for (int dx = -search_r; dx <= search_r; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= fgp.grid_dim.x || ny < 0 || ny >= fgp.grid_dim.y || nz < 0 || nz >= fgp.grid_dim.z)
                    continue;
                int cell = nz * fgp.grid_dim.x * fgp.grid_dim.y + ny * fgp.grid_dim.x + nx;
                for (int c = 0; c < MAX_FACES_PER_CELL; c++) {
                    int f = face_grid_particles[cell * MAX_FACES_PER_CELL + c];
                    if (f < 0) break;
                    float3 d = p - face_centers[f];
                    if (dot(d, d) <= band_h * band_h) {
                        band_mask[idx] = 1;
                        return;
                    }
                }
            }
}

__global__ void vap_clear_uchar_kernel(unsigned char* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 0;
}

__global__ void vap_reset_counter_kernel(int* counter) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *counter = 0;
}

__global__ void vap_scatter_fluid_atomic_kernel(
    const float3* fluid_pos,
    const float3* fluid_vel,
    const float* fluid_density,
    const unsigned char* band_mask,
    float3* merged_pos,
    float3* merged_vel,
    float3* merged_nrm,
    unsigned char* merged_attr,
    float* merged_density,
    int* fluid_to_merged,
    int* band_counter,
    int num_fluid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_fluid) return;
    if (!band_mask[i]) {
        fluid_to_merged[i] = -1;
        return;
    }
    int slot = atomicAdd(band_counter, 1);
    fluid_to_merged[i] = slot;
    merged_pos[slot] = fluid_pos[i];
    merged_vel[slot] = fluid_vel[i];
    merged_nrm[slot] = make_float3(0, 0, 0);
    merged_attr[slot] = VAP_ATTR_FLUID;
    merged_density[slot] = fluid_density[i];
}

__global__ void vap_subtract_kernel(float* dst, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] - b[i];
}

__global__ void vap_update_p_kernel(float* p, const float* r, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i] + beta * p[i];
}

__global__ void vap_scatter_fluid_kernel(
    const float3* fluid_pos,
    const float3* fluid_vel,
    const float* fluid_density,
    const unsigned char* band_mask,
    const int* prefix,
    float3* merged_pos,
    float3* merged_vel,
    float3* merged_nrm,
    unsigned char* merged_attr,
    float* merged_density,
    int* fluid_to_merged,
    int num_fluid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_fluid) return;
    if (!band_mask[i]) {
        fluid_to_merged[i] = -1;
        return;
    }
    int slot = prefix[i] - 1;
    fluid_to_merged[i] = slot;
    merged_pos[slot] = fluid_pos[i];
    merged_vel[slot] = fluid_vel[i];
    merged_nrm[slot] = make_float3(0, 0, 0);
    merged_attr[slot] = VAP_ATTR_FLUID;
    merged_density[slot] = fluid_density[i];
}

// ghost：运动学边界（PeriDyno 原版 setFixed 但 GhostFluid 仍传入 bVel）
//   merged_vel = FEM 顶点速度（非零、非固定壁）
//   merged_pos = 帧初位置 + vel * ghost_time_offset（子步内随鱼体运动）
__global__ void vap_scatter_ghost_kernel(
    const float3* b_pos,
    const float3* b_vel,
    const float3* b_nrm,
    float3* merged_pos,
    float3* merged_vel,
    float3* merged_nrm,
    unsigned char* merged_attr,
    int ghost_offset,
    int num_ghost,
    float ghost_time_offset)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= num_ghost) return;
    int idx = ghost_offset + g;
    float3 pos0 = b_pos[g];
    float3 vel_g = b_vel[g];
    merged_pos[idx] = pos0 + vel_g * ghost_time_offset;
    merged_vel[idx] = vel_g;
    merged_nrm[idx] = b_nrm[g];
    merged_attr[idx] = VAP_ATTR_KINEMATIC_GHOST;
}

__global__ void vap_build_merged_grid_kernel(
    const float3* merged_pos,
    int* grid_counts,
    int* grid_particles,
    GridParams gp,
    int num_merged)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_merged) return;
    float3 pos = merged_pos[idx];
    int cx = max(0, min((int)((pos.x - gp.grid_min.x) * gp.inv_cell_size), gp.grid_dim.x - 1));
    int cy = max(0, min((int)((pos.y - gp.grid_min.y) * gp.inv_cell_size), gp.grid_dim.y - 1));
    int cz = max(0, min((int)((pos.z - gp.grid_min.z) * gp.inv_cell_size), gp.grid_dim.z - 1));
    int cell = cz * gp.grid_dim.x * gp.grid_dim.y + cy * gp.grid_dim.x + cx;
    int slot = atomicAdd(&grid_counts[cell], 1);
    if (slot < VAP_MAX_MERGED_CELL)
        grid_particles[cell * VAP_MAX_MERGED_CELL + slot] = idx;
}

__global__ void vap_build_neighbors_kernel(
    const float3* merged_pos,
    const unsigned char* merged_attr,
    int* neighbor_count,
    int* neighbors,
    const int* grid_counts,
    const int* grid_particles,
    GridParams gp,
    float h,
    int num_merged)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_merged) return;
    if (merged_attr[i] == VAP_ATTR_KINEMATIC_GHOST) {
        neighbor_count[i] = 0;
        return;
    }
    float3 pi = merged_pos[i];
    int cx = max(0, min((int)((pi.x - gp.grid_min.x) * gp.inv_cell_size), gp.grid_dim.x - 1));
    int cy = max(0, min((int)((pi.y - gp.grid_min.y) * gp.inv_cell_size), gp.grid_dim.y - 1));
    int cz = max(0, min((int)((pi.z - gp.grid_min.z) * gp.inv_cell_size), gp.grid_dim.z - 1));
    const int search_r = (int)ceilf(h * gp.inv_cell_size) + 1;
    const int base = i * MAX_VAP_NEIGHBORS;
    int ghost_cnt = 0;
    int fluid_cnt = 0;
    const int max_fluid = MAX_VAP_NEIGHBORS - VAP_GHOST_NEIGHBOR_RESERVE;

    // 先收集 ghost，再收集 fluid，避免 48/128 上限把 ghost 挤掉
    for (int dz = -search_r; dz <= search_r; dz++)
        for (int dy = -search_r; dy <= search_r; dy++)
            for (int dx = -search_r; dx <= search_r; dx++) {
                if (ghost_cnt >= VAP_GHOST_NEIGHBOR_RESERVE) break;
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= gp.grid_dim.x || ny < 0 || ny >= gp.grid_dim.y || nz < 0 || nz >= gp.grid_dim.z)
                    continue;
                int cell = nz * gp.grid_dim.x * gp.grid_dim.y + ny * gp.grid_dim.x + nx;
                for (int c = 0; c < VAP_MAX_MERGED_CELL && ghost_cnt < VAP_GHOST_NEIGHBOR_RESERVE; c++) {
                    int j = grid_particles[cell * VAP_MAX_MERGED_CELL + c];
                    if (j < 0) break;
                    if (j == i || merged_attr[j] != VAP_ATTR_KINEMATIC_GHOST) continue;
                    float3 r = pi - merged_pos[j];
                    float dist = length(r);
                    if (dist >= h || dist < 1e-8f) continue;
                    neighbors[base + ghost_cnt++] = j;
                }
            }

    for (int dz = -search_r; dz <= search_r && fluid_cnt < max_fluid; dz++)
        for (int dy = -search_r; dy <= search_r && fluid_cnt < max_fluid; dy++)
            for (int dx = -search_r; dx <= search_r && fluid_cnt < max_fluid; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= gp.grid_dim.x || ny < 0 || ny >= gp.grid_dim.y || nz < 0 || nz >= gp.grid_dim.z)
                    continue;
                int cell = nz * gp.grid_dim.x * gp.grid_dim.y + ny * gp.grid_dim.x + nx;
                for (int c = 0; c < VAP_MAX_MERGED_CELL && fluid_cnt < max_fluid; c++) {
                    int j = grid_particles[cell * VAP_MAX_MERGED_CELL + c];
                    if (j < 0) break;
                    if (j == i || merged_attr[j] == VAP_ATTR_KINEMATIC_GHOST) continue;
                    float3 r = pi - merged_pos[j];
                    float dist = length(r);
                    if (dist >= h || dist < 1e-8f) continue;
                    neighbors[base + VAP_GHOST_NEIGHBOR_RESERVE + fluid_cnt++] = j;
                }
            }

    // 紧凑排列：ghost 在前，fluid 紧随其后
    if (ghost_cnt < VAP_GHOST_NEIGHBOR_RESERVE && fluid_cnt > 0) {
        const int shift = VAP_GHOST_NEIGHBOR_RESERVE - ghost_cnt;
        for (int k = 0; k < fluid_cnt; ++k)
            neighbors[base + ghost_cnt + k] = neighbors[base + VAP_GHOST_NEIGHBOR_RESERVE + k];
    }
    neighbor_count[i] = ghost_cnt + fluid_cnt;
}

__global__ void vap_compute_alpha_kernel(
    float* alpha,
    const float3* pos,
    const unsigned char* attr,
    const int* neighbor_count,
    const int* neighbors,
    float h,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (attr[i] == VAP_ATTR_KINEMATIC_GHOST) { alpha[i] = 1.0f; return; }
    float ai = 0.0f;
    float3 pi = pos[i];
    int base = i * MAX_VAP_NEIGHBORS;
    int nc = neighbor_count[i];
    for (int k = 0; k < nc; k++) {
        int j = neighbors[base + k];
        float3 r = pi - pos[j];
        float dist = length(r);
        ai += vap_poly6_w(dist * dist, h, h * h);
    }
    alpha[i] = fmaxf(ai, 1e-6f);
}

__global__ void vap_correct_alpha_kernel2(float* alpha, const unsigned char* attr, float alpha_max, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (attr[i] == VAP_ATTR_KINEMATIC_GHOST) return;
    alpha[i] = fmaxf(alpha[i], alpha_max);
}

__global__ void vap_compute_diagonal_kernel(
    float* aii_fluid,
    float* aii_total,
    const float* alpha,
    const float3* pos,
    const unsigned char* attr,
    const int* neighbor_count,
    const int* neighbors,
    float h,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (attr[i] == VAP_ATTR_KINEMATIC_GHOST) return;
    float inv_a = 1.0f / alpha[i];
    float3 pi = pos[i];
    float dia_f = 0.0f, dia_t = 0.0f;
    int base = i * MAX_VAP_NEIGHBORS;
    int nc = neighbor_count[i];
    for (int k = 0; k < nc; k++) {
        int j = neighbors[base + k];
        float3 r = pi - pos[j];
        float dist = length(r);
        if (dist < 1e-8f) continue;
        float wrr = inv_a * vap_wrr(dist, h);
        if (attr[j] != VAP_ATTR_KINEMATIC_GHOST) {
            dia_t += wrr;
            dia_f += wrr;
            atomicAdd(&aii_fluid[j], wrr);
            atomicAdd(&aii_total[j], wrr);
        } else {
            dia_t += VAP_GHOST_DIAG_COEFF * wrr;
        }
    }
    atomicAdd(&aii_fluid[i], dia_f);
    atomicAdd(&aii_total[i], dia_t);
}

__global__ void vap_detect_surface_kernel(
    float* aii,
    unsigned char* is_surface,
    const float* aii_fluid,
    const float* aii_total,
    const unsigned char* attr,
    const int* neighbor_count,
    const int* neighbors,
    float max_a,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (attr[i] == VAP_ATTR_KINEMATIC_GHOST) return;
    bool near_wall = false;
    int base = i * MAX_VAP_NEIGHBORS;
    int nc = neighbor_count[i];
    for (int k = 0; k < nc; k++) {
        if (attr[neighbors[base + k]] == VAP_ATTR_KINEMATIC_GHOST) { near_wall = true; break; }
    }
    float diag_f = aii_fluid[i];
    float diag_t = aii_total[i];
    float out_a = diag_f;
    unsigned char surf = 0;
    const float threshold = 1.5f;
    if (near_wall && diag_t < threshold * max_a) {
        surf = 1;
        out_a = threshold * max_a - (diag_t - diag_f);
    } else if (!near_wall && diag_f < max_a) {
        surf = 1;
        out_a = max_a;
    }
    is_surface[i] = surf;
    aii[i] = fmaxf(out_a, 1e-6f);
}

__global__ void vap_compute_divergence_kernel(
    float* divergence,
    const float* alpha,
    const float3* pos,
    const float3* vel,
    const float3* normals,
    const unsigned char* attr,
    const int* neighbor_count,
    const int* neighbors,
    float separation,
    float tangential,
    float rest_density,
    float h,
    float dt,
    float ghost_div_scale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (attr[i] == VAP_ATTR_KINEMATIC_GHOST) return;
    float inv_a = 1.0f / alpha[i];
    float3 pi = pos[i], vi = vel[i];
    int base = i * MAX_VAP_NEIGHBORS;
    int nc = neighbor_count[i];
    for (int k = 0; k < nc; k++) {
        int j = neighbors[base + k];
        float3 r = pi - pos[j];
        float dist = length(r);
        if (dist < 1e-8f) continue;
        float wr = vap_spiky_grad_mag(dist, h);
        float3 g = -inv_a * r * (wr / dist);
        if (attr[j] != VAP_ATTR_KINEMATIC_GHOST) {
            float div_ij = 0.5f * dot(vi - vel[j], g) * rest_density / dt;
            atomicAdd(&divergence[i], div_ij);
            atomicAdd(&divergence[j], div_ij);
        } else {
            // 运动学 ghost：dvel = v_fluid - v_fish，vel[j] 为 FEM 顶点速度
            float3 nj = normals[j];
            float3 dvel = vi - vel[j];
            float mag_n = dot(dvel, nj);
            float3 n_vel = mag_n * nj;
            float3 t_vel = dvel - n_vel;
            float3 bc_vel;
            if (mag_n < -1e-6f)
                bc_vel = 2.0f * (n_vel + tangential * t_vel);
            else
                bc_vel = 2.0f * (separation * n_vel + tangential * t_vel);
            float div_ij = dot(bc_vel, g) * rest_density / dt;
            atomicAdd(&divergence[i], ghost_div_scale * div_ij);
        }
    }
}

__global__ void vap_compensate_source_kernel(
    float* divergence,
    const float* density,
    const unsigned char* attr,
    float rest_density,
    float dt,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (attr[i] == VAP_ATTR_KINEMATIC_GHOST) return;
    if (density[i] > rest_density) {
        float ratio = (density[i] - rest_density) / rest_density;
        // 原系数 5.0 在 dt≈1.74e-4 时贡献 ~1e12 量级源项，改为 0.05 使密度补偿与速度散度量级匹配
        atomicAdd(&divergence[i], 0.05f * rest_density * ratio / (dt * dt));
    }
}

__global__ void vap_compute_ax_kernel(
    float* ax,
    const float* pressure,
    const float* aii,
    const float* alpha,
    const float3* pos,
    const unsigned char* attr,
    const int* neighbor_count,
    const int* neighbors,
    float h,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (attr[i] == VAP_ATTR_KINEMATIC_GHOST) return;
    atomicAdd(&ax[i], aii[i] * pressure[i]);
    float inv_a = 1.0f / alpha[i];
    float3 pi = pos[i];
    int base = i * MAX_VAP_NEIGHBORS;
    int nc = neighbor_count[i];
    for (int k = 0; k < nc; k++) {
        int j = neighbors[base + k];
        if (attr[j] == VAP_ATTR_KINEMATIC_GHOST) continue;
        float3 r = pi - pos[j];
        float dist = length(r);
        if (dist < 1e-8f) continue;
        float a_ij = -inv_a * vap_wrr(dist, h);
        atomicAdd(&ax[i], a_ij * pressure[j]);
        atomicAdd(&ax[j], a_ij * pressure[i]);
    }
}

__global__ void vap_update_velocity_kernel(
    const float* pressure,
    const float* aii,
    const unsigned char* is_surface,
    float3* vel,
    const float3* pos,
    const unsigned char* attr,
    const int* neighbor_count,
    const int* neighbors,
    float rest_density,
    float h,
    float dt,
    float ghost_vel_scale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (attr[i] == VAP_ATTR_KINEMATIC_GHOST) return;
    float inv_aii = 1.0f / aii[i];
    float3 pi = pos[i];
    float3 dv = make_float3(0, 0, 0);
    bool near_wall = false;
    int base = i * MAX_VAP_NEIGHBORS;
    int nc = neighbor_count[i];
    for (int k = 0; k < nc; k++) {
        if (attr[neighbors[base + k]] == VAP_ATTR_KINEMATIC_GHOST) near_wall = true;
    }
    for (int k = 0; k < nc; k++) {
        int j = neighbors[base + k];
        float3 r = pi - pos[j];
        float dist = length(r);
        if (dist < 1e-8f) continue;
        float wr = vap_spiky_grad_mag(dist, h);
        float3 dn = -inv_aii * r * (dt / rest_density * wr / dist);
        if (attr[j] != VAP_ATTR_KINEMATIC_GHOST) {
            float3 dvp = (pressure[j] - pressure[i]) * dn;
            if (is_surface[i] && !near_wall)
                dv = dv + pressure[j] * dn;
            else
                dv = dv + dvp;
        } else {
            // 速度惩罚：相对速度 v_fluid - v_ghost（ghost 带 FEM 速度，非零固定壁）
            float w = inv_aii * vap_wrr(dist, h);
            dv = dv + ghost_vel_scale * w * dot(vel[j] - vel[i], r) * r;
        }
    }
    vel[i] = vel[i] + dv;
}

__global__ void vap_scatter_velocity_back_kernel(
    float3* fluid_vel,
    const float3* merged_vel,
    const int* fluid_to_merged,
    int num_fluid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_fluid) return;
    int m = fluid_to_merged[i];
    if (m >= 0) fluid_vel[i] = merged_vel[m];
}

__global__ void vap_clamp_scale_pressure_kernel(
    float* pressure, float abs_max, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float p = pressure[i];
    if (!isfinite(p)) p = 0.0f;
    p = fmaxf(-abs_max, fminf(abs_max, p));
    pressure[i] = p * scale;
}

__global__ void vap_scatter_pressure_back_kernel(
    float* fluid_pressure,
    const float* merged_pressure,
    const int* fluid_to_merged,
    int num_fluid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_fluid) return;
    int m = fluid_to_merged[i];
    if (m >= 0) fluid_pressure[i] = merged_pressure[m];
}

__global__ void compute_viscosity_forces_kernel(
    const float3* positions, const float3* velocities,
    const float* densities, float3* forces,
    const int* grid_particles, GridParams gp,
    float h, float mass, float viscosity, float gravity_y, int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    float3 pos_i = positions[idx], vel_i = velocities[idx];
    float3 force = make_float3(0, mass * gravity_y, 0);
    int cx = max(0, min((int)((pos_i.x - gp.grid_min.x) * gp.inv_cell_size), gp.grid_dim.x - 1));
    int cy = max(0, min((int)((pos_i.y - gp.grid_min.y) * gp.inv_cell_size), gp.grid_dim.y - 1));
    int cz = max(0, min((int)((pos_i.z - gp.grid_min.z) * gp.inv_cell_size), gp.grid_dim.z - 1));
    for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= gp.grid_dim.x || ny < 0 || ny >= gp.grid_dim.y || nz < 0 || nz >= gp.grid_dim.z)
                    continue;
                int cell = nz * gp.grid_dim.x * gp.grid_dim.y + ny * gp.grid_dim.x + nx;
                for (int c = 0; c < MAX_PARTICLES_PER_CELL; c++) {
                    int j = grid_particles[cell * MAX_PARTICLES_PER_CELL + c];
                    if (j < 0) break;
                    if (j <= idx) continue;
                    float3 r = pos_i - positions[j];
                    float dist = length(r);
                    if (dist >= h || dist < 1e-8f) continue;
                    float rho_j = densities[j];
                    float lapl = viscosity_laplacian(dist, h);
                    float3 f_v = mass * viscosity * (velocities[j] - vel_i) / rho_j * lapl;
                    force = force + f_v;
                    forces[j] = forces[j] - f_v;
                }
            }
    forces[idx] = forces[idx] + force;
}

__global__ void integrate_velocity_predictor_kernel(
    float3* velocities, const float3* forces, float mass_inv, float dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 a = forces[i] * mass_inv;
    velocities[i] = velocities[i] + a * dt;
}

__global__ void integrate_position_kernel(
    float3* positions, const float3* velocities, float dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    positions[i] = positions[i] + velocities[i] * dt;
}

__global__ void vap_dot_kernel(const float* a, const float* b, float* partial, int n) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float v = 0.0f;
    if (i < n) v = a[i] * b[i];
    cache[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = cache[0];
}

__global__ void vap_saxpy_kernel(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void vap_copy_kernel(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void vap_zero_kernel(float* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 0.0f;
}

__global__ void vap_clear_merged_grid_kernel(int* counts, int* particles, int cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cells) return;
    counts[idx] = 0;
    for (int k = 0; k < VAP_MAX_MERGED_CELL; k++)
        particles[idx * VAP_MAX_MERGED_CELL + k] = -1;
}

/**
 * band 内每个流体粒子（merged 索引 [0, n_band)）的邻域与 VAP 影响分解。
 * fluid_influence / ghost_influence：分别累加速度更新 + 散度源项中来自 fluid / ghost 邻域项的 |贡献|。
 */
__global__ void vap_band_particle_diag_kernel(
    int n_band,
    const float* pressure,
    const float* aii,
    const unsigned char* is_surface,
    const float* alpha,
    const float3* pos,
    const float3* vel,
    const float3* normals,
    const unsigned char* attr,
    const int* neighbor_count,
    const int* neighbors,
    float h,
    float rest_density,
    float dt,
    float separation,
    float tangential,
    float ghost_div_scale,
    float ghost_vel_scale,
    int* out_n_fluid_nbr,
    int* out_n_ghost_nbr,
    float* out_ghost_fluid_ratio,
    float* out_fluid_influence,
    float* out_ghost_influence,
    float* out_fluid_vel_influence,
    float* out_ghost_vel_influence)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_band) return;
    if (attr[i] != VAP_ATTR_FLUID) return;

    const float inv_aii = 1.0f / fmaxf(aii[i], 1e-12f);
    const float inv_a = 1.0f / fmaxf(alpha[i], 1e-12f);
    const float3 pi = pos[i];
    const float3 vi = vel[i];
    int nf = 0, ng = 0;
    float fluid_inf = 0.0f, ghost_inf = 0.0f;
    float fluid_vel_inf = 0.0f, ghost_vel_inf = 0.0f;
    bool near_wall = false;
    const int base = i * MAX_VAP_NEIGHBORS;
    const int nc = neighbor_count[i];
    for (int k = 0; k < nc; k++) {
        if (attr[neighbors[base + k]] == VAP_ATTR_KINEMATIC_GHOST)
            near_wall = true;
    }
    for (int k = 0; k < nc; k++) {
        const int j = neighbors[base + k];
        float3 r = pi - pos[j];
        const float dist = length(r);
        if (dist < 1e-8f) continue;
        const float wr = vap_spiky_grad_mag(dist, h);
        const float3 g = -inv_a * r * (wr / dist);

        if (attr[j] != VAP_ATTR_KINEMATIC_GHOST) {
            ++nf;
            const float3 dn = -inv_aii * r * (dt / rest_density * wr / dist);
            const float3 dvp = (pressure[j] - pressure[i]) * dn;
            const float3 vel_contrib =
                (is_surface[i] && !near_wall) ? (pressure[j] * dn) : dvp;
            fluid_vel_inf += length(vel_contrib);
            const float div_ij = 0.5f * dot(vi - vel[j], g) * rest_density / dt;
            fluid_inf += length(vel_contrib) + fabsf(div_ij);
        } else {
            ++ng;
            const float w = inv_aii * vap_wrr(dist, h);
            const float3 vel_contrib = ghost_vel_scale * w * dot(vel[j] - vi, r) * r;
            ghost_vel_inf += length(vel_contrib);
            const float wrr = inv_a * vap_wrr(dist, h);
            const float3 nj = normals[j];
            const float3 dvel = vi - vel[j];
            const float mag_n = dot(dvel, nj);
            const float3 n_vel = mag_n * nj;
            const float3 t_vel = dvel - n_vel;
            float3 bc_vel;
            if (mag_n < -1e-6f)
                bc_vel = 2.0f * (n_vel + tangential * t_vel);
            else
                bc_vel = 2.0f * (separation * n_vel + tangential * t_vel);
            const float div_ij = ghost_div_scale * dot(bc_vel, g) * rest_density / dt;
            ghost_inf += length(vel_contrib) + VAP_GHOST_DIAG_COEFF * wrr + fabsf(div_ij);
        }
    }
    out_n_fluid_nbr[i] = nf;
    out_n_ghost_nbr[i] = ng;
    out_ghost_fluid_ratio[i] = (float)ng / fmaxf((float)nf, 1.0f);
    out_fluid_influence[i] = fluid_inf;
    out_ghost_influence[i] = ghost_inf;
    out_fluid_vel_influence[i] = fluid_vel_inf;
    out_ghost_vel_influence[i] = ghost_vel_inf;
}

/** 每个表面 ghost 顶点到最近 WCSPH 流体粒子的距离 [m] */
__global__ void vap_surface_nearest_fluid_kernel(
    const float3* surface_pos,
    int num_surface,
    const float3* fluid_pos,
    const int* grid_particles,
    GridParams gp,
    float max_search_dist,
    float* out_min_dist)
{
    int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= num_surface) return;
    const float3 sp = surface_pos[si];
    float min_d2 = max_search_dist * max_search_dist;
    const int search_r = (int)ceilf(max_search_dist * gp.inv_cell_size) + 1;
    const int cx = max(0, min((int)((sp.x - gp.grid_min.x) * gp.inv_cell_size), gp.grid_dim.x - 1));
    const int cy = max(0, min((int)((sp.y - gp.grid_min.y) * gp.inv_cell_size), gp.grid_dim.y - 1));
    const int cz = max(0, min((int)((sp.z - gp.grid_min.z) * gp.inv_cell_size), gp.grid_dim.z - 1));
    for (int dz = -search_r; dz <= search_r; dz++)
        for (int dy = -search_r; dy <= search_r; dy++)
            for (int dx = -search_r; dx <= search_r; dx++) {
                const int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= gp.grid_dim.x || ny < 0 || ny >= gp.grid_dim.y || nz < 0 || nz >= gp.grid_dim.z)
                    continue;
                const int cell = nz * gp.grid_dim.x * gp.grid_dim.y + ny * gp.grid_dim.x + nx;
                for (int c = 0; c < MAX_PARTICLES_PER_CELL; c++) {
                    const int j = grid_particles[cell * MAX_PARTICLES_PER_CELL + c];
                    if (j < 0) break;
                    const float3 d = sp - fluid_pos[j];
                    const float d2 = dot(d, d);
                    if (d2 < min_d2) min_d2 = d2;
                }
            }
    out_min_dist[si] = sqrtf(min_d2);
}

#endif // PERIDYNO_VAP_HYDRO_INL
