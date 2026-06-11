#ifndef __WORLD_GPU_H__
#define __WORLD_GPU_H__

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cuda_runtime.h>
#include <vector>
#include <cusolverSp.h>
#include <cusparse.h>

#include "Constraint/ConstraintGPU.h"

namespace FEM {

class Constraint;

/// GPU 加速的 World，兼容 FEM::World 接口
/// Phase 2: GPU 约束投影 + CPU LDLT solve（混合模式）
/// Phase 3: 完整 GPU PD 循环（含 cuSOLVER）
class WorldGPU {
public:
    WorldGPU(double time_step = 1.0/100.0,
             int max_iteration = 100,
             double damping_coeff = 0.999);

    ~WorldGPU();

    // ---- 初始化（从现有 CPU World 复制矩阵） ----
    void InitializeFromCPU(
        const Eigen::VectorXd& x0,
        const Eigen::VectorXd& v0,
        const std::vector<double>& unit_mass,
        const Eigen::SparseMatrix<double>& mass_matrix,
        const Eigen::SparseMatrix<double>& inv_mass_matrix,
        const Eigen::SparseMatrix<double>& J_matrix,
        const Eigen::SparseMatrix<double>& L_matrix,
        const std::vector<Constraint*>& constraints,
        bool use_gpu_solver = false);

    // ---- 时间步进 ----
    void TimeStepping(Eigen::VectorXd& x_out, Eigen::VectorXd& v_out);

    // ---- 设置外力 (所有顶点, 直接写入 Device) ----
    void SetExternalForceDevice(const double* d_force);
    void SetExternalForceCPU(const Eigen::VectorXd& force);

    // ---- 获取状态 ----
    const double* GetDevicePositions() const { return d_x; }
    const double* GetDeviceVelocities() const { return d_v; }
    bool IsInitialized() const { return is_initialized; }
    int GetNumVertices() const { return num_vertices; }

    // ---- 将 GPU 状态回读到 CPU ----
    void DownloadToCPU(Eigen::VectorXd& x, Eigen::VectorXd& v);

    // ---- 设置激活水平（每帧在 muscle 更新后调用） ----
    void SyncActivations(const std::vector<Constraint*>& constraints) {
        constraint_data.UpdateActivations(constraints);
    }

private:
    void SetupSparseSolverGPU(const Eigen::SparseMatrix<double>& A);
    void SetupSparseSolverCPU(const Eigen::SparseMatrix<double>& A,
                               const Eigen::SparseMatrix<double>& L);

    double time_step;
    int max_iteration;
    double damping_coeff;
    int num_vertices;
    int constraint_dofs;
    bool is_initialized;
    bool use_gpu_solver;

    // ---- Device 端状态 ----
    double* d_x;          // [3 * num_vertices]
    double* d_v;
    double* d_external_force;
    double* d_qn;         // prediction position
    double* d_b;          // RHS for PD solve
    double* d_jd;         // J * d product
    double* d_x_new;      // temporary

    // ---- 约束数据 (GPU) ----
    ConstraintGPU constraint_data;

    // ---- 矩阵数据 (CPU copy, 用于 LDLT solve) ----
    Eigen::SparseMatrix<double> cpu_J;
    Eigen::SparseMatrix<double> cpu_H2ML; // H + L
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> cpu_LDLT_solver;
    Eigen::VectorXd cpu_b;      // temporary buffer
    Eigen::VectorXd cpu_jd;     // temporary buffer
    Eigen::VectorXd cpu_d;      // d vector (downloaded from GPU)
    Eigen::VectorXd cpu_x_new;  // result from solve
    Eigen::VectorXd cpu_x_new_old; // previous iterate for convergence
    std::vector<double> cpu_unit_mass;

    // ---- GPU sparse solver handle (Phase 3) ----
    cusolverSpHandle_t cusolver_handle;
    cusparseHandle_t   cusparse_handle;
    int* d_csrRowPtrA;
    int* d_csrColIndA;
    double* d_csrValA;
    int csr_nnz;
    bool solver_setup_done;

    // ---- 私有辅助方法 ----
    void ExtractConstraintData(const std::vector<Constraint*>& constraints);
    void EvaluateDVectorGPU(const double* d_positions, double* d_d_output);
    void SparseMatVecMulGPU(const Eigen::SparseMatrix<double>& M,
                            const double* d_vec, double* d_result);
    void SpMV_cpu(const Eigen::SparseMatrix<double>& M,
                  const Eigen::VectorXd& x, Eigen::VectorXd& y);
};

} // namespace FEM

#endif
