#include "WorldGPU.h"
#include "Constraint/Constraint.h"
#include "Constraint/ConstraintHeader.h"

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstring>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
    } \
} while(0)

namespace FEM {

namespace {
    constexpr double EPS = 1e-9;
    constexpr int BLOCK_SIZE = 256;
}

// ====================================================================
// GPU sparse矩阵向量乘 CUDA kernel (CSR 格式)
// y = A * x
// ====================================================================
__global__ void spmv_csr_kernel(
    const int* __restrict__ row_ptr,    // [rows+1]
    const int* __restrict__ col_ind,    // [nnz]
    const double* __restrict__ values,   // [nnz]
    const double* __restrict__ x_vec,   // [cols]
    double* __restrict__ y_vec,         // [rows]
    int rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    double sum = 0.0;
    int start = row_ptr[row];
    int end   = row_ptr[row + 1];
    for (int j = start; j < end; ++j) {
        sum += values[j] * x_vec[col_ind[j]];
    }
    y_vec[row] = sum;
}

// ====================================================================
// 向量运算 kernel
// ====================================================================
__global__ void axpy_kernel(double* y, const double* x, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

__global__ void scale_kernel(double* y, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= alpha;
}

__global__ void add_kernel(double* y, const double* a, const double* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

__global__ void sub_kernel(double* res, const double* a, const double* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] = a[i] - b[i];
}

__global__ void copy_kernel(double* dst, const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void compute_qn_kernel(
    double* qn, const double* x, const double* v,
    const double* inv_mass, const double* f_ext,
    double dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    qn[i] = x[i] + dt * v[i] + dt * dt * inv_mass[i] * f_ext[i];
}

// ====================================================================
// WorldGPU 实现
// ====================================================================

WorldGPU::WorldGPU(double time_step, int max_iteration, double damping_coeff)
    : time_step(time_step),
      max_iteration(max_iteration),
      damping_coeff(damping_coeff),
      num_vertices(0),
      constraint_dofs(0),
      is_initialized(false),
      use_gpu_solver(false),
      d_x(nullptr), d_v(nullptr), d_external_force(nullptr),
      d_qn(nullptr), d_b(nullptr), d_jd(nullptr), d_x_new(nullptr),
      csr_nnz(0),
      solver_setup_done(false),
      d_csrRowPtrA(nullptr), d_csrColIndA(nullptr), d_csrValA(nullptr),
      cusolver_handle(nullptr), cusparse_handle(nullptr)
{
}

WorldGPU::~WorldGPU() {
    if (d_x)             { cudaFree(d_x); d_x = nullptr; }
    if (d_v)             { cudaFree(d_v); d_v = nullptr; }
    if (d_external_force){ cudaFree(d_external_force); d_external_force = nullptr; }
    if (d_qn)            { cudaFree(d_qn); d_qn = nullptr; }
    if (d_b)             { cudaFree(d_b); d_b = nullptr; }
    if (d_jd)            { cudaFree(d_jd); d_jd = nullptr; }
    if (d_x_new)         { cudaFree(d_x_new); d_x_new = nullptr; }
    if (d_csrRowPtrA)    { cudaFree(d_csrRowPtrA); d_csrRowPtrA = nullptr; }
    if (d_csrColIndA)    { cudaFree(d_csrColIndA); d_csrColIndA = nullptr; }
    if (d_csrValA)       { cudaFree(d_csrValA); d_csrValA = nullptr; }
    if (cusolver_handle) { cusolverSpDestroy(cusolver_handle); cusolver_handle = nullptr; }
    if (cusparse_handle) { cusparseDestroy(cusparse_handle); cusparse_handle = nullptr; }
    constraint_data.Destroy();
}

void WorldGPU::InitializeFromCPU(
    const Eigen::VectorXd& x0,
    const Eigen::VectorXd& v0,
    const std::vector<double>& unit_mass,
    const Eigen::SparseMatrix<double>& mass_matrix,
    const Eigen::SparseMatrix<double>& inv_mass_matrix,
    const Eigen::SparseMatrix<double>& J_matrix,
    const Eigen::SparseMatrix<double>& L_matrix,
    const std::vector<Constraint*>& constraints,
    bool use_gpu_solver_flag)
{
    num_vertices = x0.rows() / 3;
    constraint_dofs = 0;
    for (auto* c : constraints) constraint_dofs += c->GetDof();
    use_gpu_solver = use_gpu_solver_flag;

    // ---- 分配 Device 内存 ----
    int n3 = 3 * num_vertices;
    CUDA_CHECK(cudaMalloc(&d_x, n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v, n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_external_force, n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_qn, n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_new, n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_jd, n3 * sizeof(double)));

    // 初始化状态
    cudaMemcpy(d_x, x0.data(), n3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v0.data(), n3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_external_force, 0, n3 * sizeof(double));
    cudaMemset(d_qn, 0, n3 * sizeof(double));
    cudaMemset(d_b, 0, n3 * sizeof(double));
    cudaMemset(d_jd, 0, n3 * sizeof(double));
    cudaMemset(d_x_new, 0, n3 * sizeof(double));

    // 保存 unit mass
    cpu_unit_mass = unit_mass;

    // ---- 约束数据提取 + 上传 GPU ----
    constraint_data.UploadFromCPU(constraints, num_vertices);

    // ---- 矩阵保存及 LDLT 分解（带正则化回退）----
    cpu_J = J_matrix;
    cpu_H2ML = (1.0 / (time_step * time_step)) * mass_matrix + L_matrix;
    cpu_LDLT_solver.analyzePattern(cpu_H2ML);
    cpu_LDLT_solver.factorize(cpu_H2ML);

    // 因子化失败时的正则化回退
    if (cpu_LDLT_solver.info() != Eigen::Success) {
        Eigen::SparseMatrix<double> A_prime = cpu_H2ML;
        Eigen::SparseMatrix<double> I(A_prime.rows(), A_prime.cols());
        I.setIdentity();
        double reg = 1e-6;
        while (cpu_LDLT_solver.info() != Eigen::Success && reg < 1e3) {
            A_prime = cpu_H2ML + reg * I;
            cpu_LDLT_solver.factorize(A_prime);
            reg *= 10.0;
        }
        if (cpu_LDLT_solver.info() != Eigen::Success) {
            std::cerr << "[WorldGPU] ERROR: LDLT factorization failed even with regularization" << std::endl;
            is_initialized = false;
            return;
        }
    }

    // 分配 CPU 临时缓冲区
    cpu_b.resize(n3);
    cpu_d.resize(3 * constraint_dofs);
    cpu_jd.resize(n3);
    cpu_x_new.resize(n3);
    cpu_x_new_old.resize(n3);
    cpu_x_new_old.setZero();

    is_initialized = true;
    std::cout << "[WorldGPU] Initialized: " << num_vertices << " vertices, "
              << constraint_dofs << " constraint DOFs, GPU solver="
              << (use_gpu_solver ? "yes" : "no") << std::endl;
}

void WorldGPU::SetExternalForceDevice(const double* d_force) {
    int n3 = 3 * num_vertices;
    cudaMemcpy(d_external_force, d_force, n3 * sizeof(double), cudaMemcpyDeviceToDevice);
}

void WorldGPU::SetExternalForceCPU(const Eigen::VectorXd& force) {
    int n3 = 3 * num_vertices;
    cudaMemcpy(d_external_force, force.data(), n3 * sizeof(double), cudaMemcpyHostToDevice);
}

void WorldGPU::DownloadToCPU(Eigen::VectorXd& x, Eigen::VectorXd& v) {
    int n3 = 3 * num_vertices;
    x.resize(n3);
    v.resize(n3);
    cudaMemcpy(x.data(), d_x, n3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), d_v, n3 * sizeof(double), cudaMemcpyDeviceToHost);
}

void WorldGPU::TimeStepping(Eigen::VectorXd& x_out, Eigen::VectorXd& v_out) {
    if (!is_initialized) return;

    int n3 = 3 * num_vertices;
    int gs = (n3 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1) qn = x + dt*v + dt^2 * invM * f_ext
    // 构建 invM 的对角数组
    std::vector<double> inv_mass_vec(n3);
    for (int i = 0; i < num_vertices; ++i) {
        double inv_m = 1.0 / cpu_unit_mass[i];
        inv_mass_vec[i*3+0] = inv_m;
        inv_mass_vec[i*3+1] = inv_m;
        inv_mass_vec[i*3+2] = inv_m;
    }
    double* d_inv_mass;
    CUDA_CHECK(cudaMalloc(&d_inv_mass, n3 * sizeof(double)));
    cudaMemcpy(d_inv_mass, inv_mass_vec.data(), n3 * sizeof(double), cudaMemcpyHostToDevice);

    compute_qn_kernel<<<gs, BLOCK_SIZE>>>(d_qn, d_x, d_v, d_inv_mass, d_external_force, time_step, n3);
    CUDA_CHECK(cudaFree(d_inv_mass));

    // 2) b = (1/dt^2) * M * qn  （M 是对角阵，逐元素计算）
    double h_inv2 = 1.0 / (time_step * time_step);
    std::vector<double> h_qn(n3), h_b_vec(n3);
    cudaMemcpy(h_qn.data(), d_qn, n3 * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n3; ++i)
        h_b_vec[i] = h_inv2 * cpu_unit_mass[i / 3] * h_qn[i];
    cudaMemcpy(d_b, h_b_vec.data(), n3 * sizeof(double), cudaMemcpyHostToDevice);

    // 3) x_n1 = qn (初始猜测)
    copy_kernel<<<gs, BLOCK_SIZE>>>(d_x_new, d_qn, n3);

    // 4) PD 迭代
    double total_eval_ms = 0, total_solve_ms = 0;
    int iter;
    for (iter = 0; iter < max_iteration; ++iter) {
        // 4a) GPU 约束投影 → d
        auto t_eval0 = std::chrono::high_resolution_clock::now();
        EvaluateDVectorGPU_launch(constraint_data, d_x_new);
        auto t_eval1 = std::chrono::high_resolution_clock::now();

        // 4b) 下载 d 到 CPU
        cudaMemcpy(cpu_d.data(), constraint_data.d_d_output,
                   3 * constraint_dofs * sizeof(double), cudaMemcpyDeviceToHost);

        // 4c) CPU 侧计算 J * d
        cpu_jd.setZero(n3);
        for (int k = 0; k < cpu_J.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(cpu_J, k); it; ++it) {
                cpu_jd[it.row()] += it.value() * cpu_d[it.col()];
            }
        }

        // 4d) CPU 侧 LDLT solve: x_{new} = solver.solve(b + J*d)
        auto t_solve0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n3; ++i) cpu_b[i] = h_b_vec[i] + cpu_jd[i];
        cpu_x_new = cpu_LDLT_solver.solve(cpu_b);
        auto t_solve1 = std::chrono::high_resolution_clock::now();

        total_eval_ms += std::chrono::duration<double, std::milli>(t_eval1 - t_eval0).count();
        total_solve_ms += std::chrono::duration<double, std::milli>(t_solve1 - t_solve0).count();

        // 4e) 收敛检测：|x_{k+1} - x_k| / n
        double norm_diff = 0.0;
        for (int i = 0; i < n3; ++i) {
            double diff = cpu_x_new[i] - cpu_x_new_old[i];
            norm_diff += diff * diff;
        }
        norm_diff = sqrt(norm_diff) / n3;

        // 保存当前解用于下次比较
        cpu_x_new_old = cpu_x_new;

        // 上传 x_new 回 GPU 用于下次迭代的约束投影
        cudaMemcpy(d_x_new, cpu_x_new.data(), n3 * sizeof(double), cudaMemcpyHostToDevice);

        if (norm_diff < EPS && iter > 0) {
            ++iter;
            break;
        }
    }

    // 5) 更新 v 和 x
    // v = (x_new - x) / dt
    sub_kernel<<<gs, BLOCK_SIZE>>>(d_v, d_x_new, d_x, n3);
    scale_kernel<<<gs, BLOCK_SIZE>>>(d_v, 1.0 / time_step, n3);
    // v *= damping
    if (fabs(damping_coeff - 1.0) > 1e-12)
        scale_kernel<<<gs, BLOCK_SIZE>>>(d_v, damping_coeff, n3);
    // x = x_new
    copy_kernel<<<gs, BLOCK_SIZE>>>(d_x, d_x_new, n3);

    cudaDeviceSynchronize();

    // 下载结果
    DownloadToCPU(x_out, v_out);

    // 日志
    static int frame_count = 0;
    if (++frame_count % 60 == 0) {
        std::cout << "[WorldGPU] PD iter=" << iter
                  << " eval_ms=" << total_eval_ms
                  << " solve_ms=" << total_solve_ms << std::endl;
    }
}

// ====================================================================
// Sparse solver setup (Phase 3 stub)
// ====================================================================
void WorldGPU::SetupSparseSolverGPU(const Eigen::SparseMatrix<double>& A) {
    // 将 Eigen 稀疏矩阵转换为 CSR 格式并上传 GPU
    // 使用 cuSOLVER 做一次 Cholesky 分解，后续每步只 solve
    csr_nnz = A.nonZeros();

    if (csr_nnz == 0) { use_gpu_solver = false; return; }

    int rows = A.rows();
    std::vector<int> hRowPtr(rows + 1, 0);
    std::vector<int> hColInd(csr_nnz);
    std::vector<double> hVal(csr_nnz);

    // 构建 CSR
    int nnz_idx = 0;
    hRowPtr[0] = 0;
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            hColInd[nnz_idx] = (int)it.col();
            hVal[nnz_idx]    = it.value();
            nnz_idx++;
        }
        hRowPtr[k + 1] = nnz_idx;
    }
    // 转置排序（Eigen 是列主序，CSR 按行序）
    // 简化处理：重新按行构建
    struct Entry { int col; double val; };
    std::vector<std::vector<Entry>> rows_data(rows);
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            rows_data[it.row()].push_back({(int)it.col(), it.value()});
        }
    }
    nnz_idx = 0;
    hRowPtr[0] = 0;
    for (int r = 0; r < rows; ++r) {
        for (auto& e : rows_data[r]) {
            hColInd[nnz_idx] = e.col;
            hVal[nnz_idx]    = e.val;
            nnz_idx++;
        }
        hRowPtr[r + 1] = nnz_idx;
    }

    // 上传 CSR 到 GPU
    cusolverSpCreate(&cusolver_handle);
    cusparseCreate(&cusparse_handle);

    CUDA_CHECK(cudaMalloc(&d_csrRowPtrA, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIndA, csr_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrValA,    csr_nnz * sizeof(double)));
    cudaMemcpy(d_csrRowPtrA, hRowPtr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, hColInd.data(), csr_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValA,    hVal.data(),    csr_nnz * sizeof(double), cudaMemcpyHostToDevice);

    use_gpu_solver = true;
    solver_setup_done = true;

    std::cout << "[WorldGPU] GPU sparse solver initialized: " << rows << "x" << rows
              << " nnz=" << csr_nnz << std::endl;
    std::cout << "[WorldGPU] WARNING: GPU LDLT solve not yet integrated in PD loop "
              << "(Phase 3), using CPU fallback for solves." << std::endl;
    use_gpu_solver = false;  // 暂时回退 CPU LDLT（Phase 3 完整实现待后续迭代）
}

void WorldGPU::SetupSparseSolverCPU(const Eigen::SparseMatrix<double>& A,
                                     const Eigen::SparseMatrix<double>& L) {
    cpu_H2ML = A;
    cpu_LDLT_solver.analyzePattern(cpu_H2ML);
    cpu_LDLT_solver.factorize(cpu_H2ML);
}

void WorldGPU::ExtractConstraintData(const std::vector<Constraint*>& constraints) {
    constraint_data.UploadFromCPU(constraints, num_vertices);
}

// 为 GPU 约束投影预留的接口（已在 ConstraintGPU.cu 中实现为 EvaluateDVectorGPU_launch）
void WorldGPU::EvaluateDVectorGPU(const double* d_positions, double* d_d_output) {
    EvaluateDVectorGPU_launch(constraint_data, d_positions);
}

void WorldGPU::SparseMatVecMulGPU(const Eigen::SparseMatrix<double>& M,
                                   const double* d_vec, double* d_result) {
    // Stub: 在实际部署中会将 M 转为 CSR 并上传 GPU，用 spmv_csr_kernel 计算
    // 当前 Phase 2 中 J*d 的计算在 CPU 侧完成 → 已通过 DownloadD→CPU→sparse mv 实现
}

void WorldGPU::SpMV_cpu(const Eigen::SparseMatrix<double>& M,
                         const Eigen::VectorXd& x, Eigen::VectorXd& y) {
    y.setZero(M.rows());
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            y[it.row()] += it.value() * x[it.col()];
        }
    }
}

} // namespace FEM
