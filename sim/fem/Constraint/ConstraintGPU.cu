#include "ConstraintGPU.h"
#include "CorotateFEMConstraint.h"
#include "LinearMuscleConstraint.h"
#include "TriangleMuscleConstraint.h"
#include "TriangleBendingConstraint.h"
#include "TriangleStrainConstraint.h"
#include "AttachmentConstraint.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <Eigen/Core>

// ====================================================================
// GPU helper: 3×3 SVD via analytical Jacobi (for corotated FEM)
// 参考 McAdams et al. 2011 "Computing the SVD of 3x3 matrices ..."
// ====================================================================

namespace {

#define GAMMA 5.828427124746190097603377448424e-1f  // sqrt(2) - 1 ≈ 0.414...

__device__ void givens(double a, double b, double& c, double& s) {
    if (b == 0.0) { c = 1.0; s = 0.0; return; }
    if (fabs(b) > fabs(a)) {
        double t = -a / b;
        s = 1.0 / sqrt(1.0 + t * t);
        c = s * t;
    } else {
        double t = -b / a;
        c = 1.0 / sqrt(1.0 + t * t);
        s = c * t;
    }
}

// 3x3 矩阵乘法 C = A * B
__device__ void mat3_mul(const double A[9], const double B[9], double C[9]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C[i*3+j] = A[i*3+0]*B[0*3+j] + A[i*3+1]*B[1*3+j] + A[i*3+2]*B[2*3+j];
        }
    }
}

// 矩阵转置
__device__ void mat3_transpose(const double A[9], double AT[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            AT[i*3+j] = A[j*3+i];
}

// 解析 3×3 SVD：A = U * diag(S) * V^T
// 返回值: S[0],S[1],S[2], U[9], V[9]
__device__ void svd3x3(const double A[9], double S[3], double U[9], double V[9]) {
    // 计算 A^T * A 的特征分解来得到右奇异向量
    double ATA[9], eigU[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ATA[i*3+j] = 0.0;
            for (int k = 0; k < 3; ++k)
                ATA[i*3+j] += A[k*3+i] * A[k*3+j];
        }
    }

    // Jacobi 特征分解 A^T*A = V * diag(S^2) * V^T
    // 初始化 V 为单位矩阵
    for (int i = 0; i < 9; ++i) eigU[i] = (i%4==0) ? 1.0 : 0.0;
    for (int i = 0; i < 9; ++i) V[i] = eigU[i];

    double A_copy[9]; // 工作的对称矩阵
    for (int i = 0; i < 9; ++i) A_copy[i] = ATA[i];

    // Jacobi 迭代（最多 10 次扫描）
    for (int sweep = 0; sweep < 10; ++sweep) {
        double off_norm = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = i+1; j < 3; ++j)
                off_norm += A_copy[i*3+j] * A_copy[i*3+j];
        if (off_norm < 1e-12) break;

        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                double app = A_copy[p*3+p];
                double aqq = A_copy[q*3+q];
                double apq = A_copy[p*3+q];
                if (fabs(apq) < 1e-14) continue;

                double alpha = (app - aqq) * 0.5;
                double beta = apq;
                double t;
                if (fabs(alpha) > fabs(beta)) {
                    t = beta / alpha;
                    t = (t < 0 ? -(fabs(t) + sqrt(1.0 + t*t))
                             : fabs(t) + sqrt(1.0+t*t)) * (alpha > 0 ? 1.0 : -1.0);
                    t = 1.0 / fabs(t);
                } else {
                    t = alpha / beta;
                    t = (alpha < 0 ? -fabs(t) + sqrt(1.0+t*t) : fabs(t) + sqrt(1.0+t*t));
                }
                double c = 1.0 / sqrt(1.0 + t*t);
                double s = c * t;

                // 更新对角块
                double new_app = c*c*app - 2.0*c*s*apq + s*s*aqq;
                double new_aqq = s*s*app + 2.0*c*s*apq + c*c*aqq;
                A_copy[p*3+p] = new_app;
                A_copy[q*3+q] = new_aqq;
                A_copy[p*3+q] = 0.0;
                A_copy[q*3+p] = 0.0;

                // 更新剩余行列
                for (int k = 0; k < 3; ++k) {
                    if (k != p && k != q) {
                        double akp = A_copy[k*3+p], akq = A_copy[k*3+q];
                        A_copy[k*3+p] = c*akp - s*akq;
                        A_copy[p*3+k] = A_copy[k*3+p];
                        A_copy[k*3+q] = s*akp + c*akq;
                        A_copy[q*3+k] = A_copy[k*3+q];
                    }
                }
                // 更新 V
                for (int k = 0; k < 3; ++k) {
                    double vkp = V[k*3+p], vkq = V[k*3+q];
                    V[k*3+p] = c*vkp - s*vkq;
                    V[k*3+q] = s*vkp + c*vkq;
                }
            }
        }
    }

    // A_copy 现在是对角矩阵 → S^2
    for (int i = 0; i < 3; ++i) {
        double val = A_copy[i*3+i];
        S[i] = (val > 1e-20) ? sqrt(val) : 0.0;
    }

    // 计算 U = A * V * diag(1/S)
    double AV[9];
    mat3_mul(A, V, AV);
    for (int i = 0; i < 9; ++i) U[i] = AV[i] * (S[i/3] > 1e-12 ? 1.0/S[i/3] : 0.0);
}

} // anonymous namespace

// ====================================================================
// CUDA 核函数：各约束类型的 EvaluateDVector
// ====================================================================

// CorotateFEM tet 约束的投影
__global__ void evaluate_corotate_kernel(
    const double* __restrict__ d_positions,    // [3 * num_vertices]
    const int*    __restrict__ d_tet_indices,   // [4 * num_tets]
    const double* __restrict__ d_invDm,         // [9 * num_tets]
    const double* __restrict__ d_volume,
    const double* __restrict__ d_stiffness_mu,
    const double* __restrict__ d_stiffness_lambda,
    double*       __restrict__ d_d_output,      // 输出 d 向量
    int num_tets,
    int d_offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tets) return;

    // 读取 4 个顶点的位置
    int i0 = d_tet_indices[tid*4 + 0];
    int i1 = d_tet_indices[tid*4 + 1];
    int i2 = d_tet_indices[tid*4 + 2];
    int i3 = d_tet_indices[tid*4 + 3];

    double x0[3] = { d_positions[i0*3+0], d_positions[i0*3+1], d_positions[i0*3+2] };
    double x1[3] = { d_positions[i1*3+0], d_positions[i1*3+1], d_positions[i1*3+2] };
    double x2[3] = { d_positions[i2*3+0], d_positions[i2*3+1], d_positions[i2*3+2] };
    double x3[3] = { d_positions[i3*3+0], d_positions[i3*3+1], d_positions[i3*3+2] };

    // Ds = [x1-x0, x2-x0, x3-x0]
    double Ds[9];
    for (int j = 0; j < 3; ++j) {
        Ds[j*3+0] = x1[j] - x0[j];
        Ds[j*3+1] = x2[j] - x0[j];
        Ds[j*3+2] = x3[j] - x0[j];
    }

    // F = Ds * invDm
    const double* invDm = d_invDm + tid * 9;
    double F[9];
    mat3_mul(Ds, invDm, F);

    // SVD(F) → U, S, V → R = U * V^T
    double S[3], U[9], V[9];
    svd3x3(F, S, U, V);

    double R[9];  // R = U * V^T
    {
        double VT[9];
        mat3_transpose(V, VT);
        mat3_mul(U, VT, R);
    }

    // 计算 F 的行列式
    double detF = F[0]*(F[4]*F[8]-F[5]*F[7])
                - F[1]*(F[3]*F[8]-F[5]*F[6])
                + F[2]*(F[3]*F[7]-F[4]*F[6]);

    // md = R
    double md[9];
    for (int j = 0; j < 9; ++j) md[j] = R[j];
    // 反转列: md(:,2) *= -1 if detF < 0
    if (detF < 0.0) {
        md[2] = -md[2];
        md[5] = -md[5];
        md[8] = -md[8];
    }

    // Volume-preserving Newton iteration (5 steps)
    double Sc[3] = { S[0], S[1], S[2] };
    double Dc[3] = { 0.0, 0.0, 0.0 };
    for (int iter = 0; iter < 5; ++iter) {
        double prod = (Sc[0]+Dc[0]) * (Sc[1]+Dc[1]) * (Sc[2]+Dc[2]) - 1.0;
        double grad[3] = {
            (Sc[1]+Dc[1])*(Sc[2]+Dc[2]),
            (Sc[0]+Dc[0])*(Sc[2]+Dc[2]),
            (Sc[0]+Dc[0])*(Sc[1]+Dc[1])
        };
        double grad2 = grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2];
        double ddot = Dc[0]*grad[0] + Dc[1]*grad[1] + Dc[2]*grad[2];
        double lam = (grad2 > 1e-12) ? (ddot - prod) / grad2 : 0.0;
        for (int j = 0; j < 3; ++j) Dc[j] = lam * grad[j];
    }

    // md_volume = U * diag(S+Dc) * V^T
    double S_corr[9] = {0}; // diag(Sc + Dc)
    S_corr[0] = Sc[0] + Dc[0];
    S_corr[4] = Sc[1] + Dc[1];
    S_corr[8] = Sc[2] + Dc[2];
    double tmp[9], VT[9], md_vol[9];
    mat3_mul(U, S_corr, tmp);
    mat3_transpose(V, VT);
    mat3_mul(tmp, VT, md_vol);

    // 写入 d 向量：md 的列 (3×3) + md_volume 的列 (3×3) = 6*3 = 18 scalars
    int offset = d_offset + tid * 18;
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
            d_d_output[offset + col*3 + row] = md[row*3 + col];
        }
    }
    offset += 9;
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
            d_d_output[offset + col*3 + row] = md_vol[row*3 + col];
        }
    }
}

// LinearMuscle 约束投影
__global__ void evaluate_linear_muscle_kernel(
    const double* __restrict__ d_positions,
    const int*    __restrict__ d_tet_indices,
    const double* __restrict__ d_fiber,
    const double* __restrict__ d_activation,
    const double* __restrict__ d_invDm,
    double*       __restrict__ d_d_output,
    int num_muscles,
    int d_offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_muscles) return;

    int i0 = d_tet_indices[tid*4 + 0];
    int i1 = d_tet_indices[tid*4 + 1];
    int i2 = d_tet_indices[tid*4 + 2];
    int i3 = d_tet_indices[tid*4 + 3];

    double x0[3] = { d_positions[i0*3+0], d_positions[i0*3+1], d_positions[i0*3+2] };
    double x1[3] = { d_positions[i1*3+0], d_positions[i1*3+1], d_positions[i1*3+2] };
    double x2[3] = { d_positions[i2*3+0], d_positions[i2*3+1], d_positions[i2*3+2] };
    double x3[3] = { d_positions[i3*3+0], d_positions[i3*3+1], d_positions[i3*3+2] };

    double Ds[9];
    for (int j = 0; j < 3; ++j) {
        Ds[j*3+0] = x1[j] - x0[j];
        Ds[j*3+1] = x2[j] - x0[j];
        Ds[j*3+2] = x3[j] - x0[j];
    }

    const double* invDm = d_invDm + tid * 9;
    double F[9];
    mat3_mul(Ds, invDm, F);

    const double* fiber_dir = d_fiber + tid * 3;
    double act = d_activation[tid];

    double mp0[3];
    for (int j = 0; j < 3; ++j) {
        double sum = 0.0;
        for (int k = 0; k < 3; ++k) sum += F[j*3+k] * fiber_dir[k];
        mp0[j] = (1.0 - act) * sum;
    }

    int offset = d_offset + tid * 3;
    d_d_output[offset+0] = mp0[0];
    d_d_output[offset+1] = mp0[1];
    d_d_output[offset+2] = mp0[2];
}

// Attachment 约束投影
__global__ void evaluate_attachment_kernel(
    const int*    __restrict__ d_indices,
    const double* __restrict__ d_target,
    double*       __restrict__ d_d_output,
    int num_attach,
    int d_offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_attach) return;

    int idx = d_indices[tid];
    const double* p = d_target + tid * 3;
    int off = d_offset + tid * 3;
    d_d_output[off+0] = p[0];
    d_d_output[off+1] = p[1];
    d_d_output[off+2] = p[2];
}

// TriangleStrain 约束投影 (2 DOFs)
__global__ void evaluate_strain_kernel(
    const double* __restrict__ d_positions,
    const int*    __restrict__ d_tri_indices,
    const double* __restrict__ d_invDm,
    double*       __restrict__ d_d_output,
    int num_strains,
    int d_offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_strains) return;

    int i0 = d_tri_indices[tid*3 + 0];
    int i1 = d_tri_indices[tid*3 + 1];
    int i2 = d_tri_indices[tid*3 + 2];

    double x0[3] = { d_positions[i0*3+0], d_positions[i0*3+1], d_positions[i0*3+2] };
    double x1[3] = { d_positions[i1*3+0], d_positions[i1*3+1], d_positions[i1*3+2] };
    double x2[3] = { d_positions[i2*3+0], d_positions[i2*3+1], d_positions[i2*3+2] };

    // Ds = [x1-x0, x2-x0] (3x2)
    double Ds[6];
    for (int j = 0; j < 3; ++j) {
        Ds[j*2+0] = x1[j] - x0[j];
        Ds[j*2+1] = x2[j] - x0[j];
    }

    // P 的第一列 = normalize(Ds[:,0])
    double len0 = sqrt(Ds[0]*Ds[0] + Ds[2]*Ds[2] + Ds[4]*Ds[4]);
    double P0[3] = { len0>1e-12 ? Ds[0]/len0:0, len0>1e-12 ? Ds[2]/len0:0, len0>1e-12 ? Ds[4]/len0:0 };

    // P 的第二列 = normalize(Ds[:,1] - dot(Ds[:,1],P0)*P0)
    double dot1 = Ds[1]*P0[0] + Ds[3]*P0[1] + Ds[5]*P0[2];
    double u1 = Ds[1] - dot1*P0[0];
    double u2 = Ds[3] - dot1*P0[1];
    double u3 = Ds[5] - dot1*P0[2];
    double len1 = sqrt(u1*u1 + u2*u2 + u3*u3);
    double P1[3] = { len1>1e-12 ? u1/len1:0, len1>1e-12 ? u2/len1:0, len1>1e-12 ? u3/len1:0 };

    // P_transpose * Ds (2x2)
    double PT_Ds[4];
    for (int j = 0; j < 2; ++j) {
        double pa = (j==0) ? P0[0] : P1[0];
        double pb = (j==0) ? P0[1] : P1[1];
        double pc = (j==0) ? P0[2] : P1[2];
        PT_Ds[j*2+0] = pa*Ds[0] + pb*Ds[2] + pc*Ds[4];
        PT_Ds[j*2+1] = pa*Ds[1] + pb*Ds[3] + pc*Ds[5];
    }

    // F = PT_Ds * invDm (2x2)
    const double* invDm = d_invDm + tid * 4;
    double F2[4];
    F2[0] = PT_Ds[0]*invDm[0] + PT_Ds[1]*invDm[2];
    F2[1] = PT_Ds[0]*invDm[1] + PT_Ds[1]*invDm[3];
    F2[2] = PT_Ds[2]*invDm[0] + PT_Ds[3]*invDm[2];
    F2[3] = PT_Ds[2]*invDm[1] + PT_Ds[3]*invDm[3];

    // 2x2 SVD of F2
    // 用 Jacobi 对角化 F2^T*F2
    double F2TF2[4];
    F2TF2[0] = F2[0]*F2[0] + F2[2]*F2[2];
    F2TF2[1] = F2[0]*F2[1] + F2[2]*F2[3];
    F2TF2[2] = F2[1]*F2[0] + F2[3]*F2[2];
    F2TF2[3] = F2[1]*F2[1] + F2[3]*F2[3];

    double theta;
    double diff = F2TF2[0] - F2TF2[3];
    if (fabs(F2TF2[1]) < 1e-14) {
        if (fabs(diff) < 1e-14) theta = 0.0;
        else theta = (diff > 0) ? 0.0 : M_PI_2;
    } else {
        theta = 0.5 * atan2(2.0 * F2TF2[1], diff);
    }

    double c = cos(theta), s = sin(theta);
    double V2[4] = { c, -s, s, c };  // V

    // 奇异值
    double sv_F2TF2[2];
    sv_F2TF2[0] = c*c*F2TF2[0] + 2.0*c*s*F2TF2[1] + s*s*F2TF2[3];
    sv_F2TF2[1] = s*s*F2TF2[0] - 2.0*c*s*F2TF2[1] + c*c*F2TF2[3];
    double S2[2];
    S2[0] = (sv_F2TF2[0] > 1e-20) ? sqrt(sv_F2TF2[0]) : 0.0;
    S2[1] = (sv_F2TF2[1] > 1e-20) ? sqrt(sv_F2TF2[1]) : 0.0;

    // U2 = F2 * V2 * diag(1/S2)
    double U2[4];
    for (int j = 0; j < 2; ++j) {
        U2[j*2+0] = F2[j*2+0]*V2[0] + F2[j*2+1]*V2[2];
        U2[j*2+1] = F2[j*2+0]*V2[1] + F2[j*2+1]*V2[3];
    }
    if (S2[0] > 1e-12) { U2[0]/=S2[0]; U2[2]/=S2[0]; }
    if (S2[1] > 1e-12) { U2[1]/=S2[1]; U2[3]/=S2[1]; }

    // R = U2 * V2^T
    double R2[4];
    R2[0] = U2[0]*V2[0] + U2[1]*V2[1];
    R2[1] = U2[0]*V2[2] + U2[1]*V2[3];
    R2[2] = U2[2]*V2[0] + U2[3]*V2[1];
    R2[3] = U2[2]*V2[2] + U2[3]*V2[3];

    // md = P * R2 (3x2)
    double md[6];
    for (int j = 0; j < 2; ++j) {
        md[j*3+0] = P0[0]*R2[0+j] + P1[0]*R2[2+j];
        md[j*3+1] = P0[1]*R2[0+j] + P1[1]*R2[2+j];
        md[j*3+2] = P0[2]*R2[0+j] + P1[2]*R2[2+j];
    }

    // 写入 d: 2*3 = 6 scalars
    int off = d_offset + tid * 6;
    for (int k = 0; k < 6; ++k) d_d_output[off+k] = md[k];
}

// TriangleBending 约束投影
__global__ void evaluate_bending_kernel(
    const double* __restrict__ d_positions,
    const int*    __restrict__ d_quad_indices,
    const double* __restrict__ d_weights,
    const double* __restrict__ d_rest_n,
    double*       __restrict__ d_d_output,
    int num_bends,
    int d_offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_bends) return;

    int i0 = d_quad_indices[tid*4 + 0];
    int i1 = d_quad_indices[tid*4 + 1];
    int i2 = d_quad_indices[tid*4 + 2];
    int i3 = d_quad_indices[tid*4 + 3];

    const double* w = d_weights + tid * 4;
    double n_rest = d_rest_n[tid];

    double e[3] = {0, 0, 0};
    if (n_rest > 1e-6) {
        for (int j = 0; j < 3; ++j) {
            e[j] += w[0] * d_positions[i0*3+j];
            e[j] += w[1] * d_positions[i1*3+j];
            e[j] += w[2] * d_positions[i2*3+j];
            e[j] += w[3] * d_positions[i3*3+j];
        }
        double l = sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
        if (l > 1e-6) {
            double scale = n_rest / l;
            e[0] *= scale; e[1] *= scale; e[2] *= scale;
        }
    }
    int off = d_offset + tid * 3;
    d_d_output[off+0] = e[0];
    d_d_output[off+1] = e[1];
    d_d_output[off+2] = e[2];
}

// TriangleMuscle 约束投影
__global__ void evaluate_tri_muscle_kernel(
    const double* __restrict__ d_positions,
    const int*    __restrict__ d_tri_indices,
    const double* __restrict__ d_fiber_2d,
    const double* __restrict__ d_activation,
    const double* __restrict__ d_invDm,
    double*       __restrict__ d_d_output,
    int num_tris,
    int d_offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tris) return;

    int i0 = d_tri_indices[tid*3 + 0];
    int i1 = d_tri_indices[tid*3 + 1];
    int i2 = d_tri_indices[tid*3 + 2];

    double p0[3] = { d_positions[i0*3+0], d_positions[i0*3+1], d_positions[i0*3+2] };
    double p1[3] = { d_positions[i1*3+0], d_positions[i1*3+1], d_positions[i1*3+2] };
    double p2[3] = { d_positions[i2*3+0], d_positions[i2*3+1], d_positions[i2*3+2] };

    // Ds = [p1-p0, p2-p0] (3x2)
    double Ds[6];
    for (int j = 0; j < 3; ++j) {
        Ds[j*2+0] = p1[j] - p0[j];
        Ds[j*2+1] = p2[j] - p0[j];
    }

    // P: 格拉姆-施密特
    double len0_sq = Ds[0]*Ds[0] + Ds[2]*Ds[2] + Ds[4]*Ds[4];
    double len0 = sqrt(len0_sq > 1e-24 ? len0_sq : 1.0);
    double P0[3] = { Ds[0]/len0, Ds[2]/len0, Ds[4]/len0 };

    double dot = Ds[1]*P0[0] + Ds[3]*P0[1] + Ds[5]*P0[2];
    double v1[3] = { Ds[1]-dot*P0[0], Ds[3]-dot*P0[1], Ds[5]-dot*P0[2] };
    double len1_sq = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
    double len1 = sqrt(len1_sq > 1e-24 ? len1_sq : 1.0);
    double P1[3] = { v1[0]/len1, v1[1]/len1, v1[2]/len1 };

    // P^T * Ds (2x2)
    double PT_Ds[4];
    PT_Ds[0] = P0[0]*Ds[0] + P0[1]*Ds[2] + P0[2]*Ds[4];
    PT_Ds[1] = P0[0]*Ds[1] + P0[1]*Ds[3] + P0[2]*Ds[5];
    PT_Ds[2] = P1[0]*Ds[0] + P1[1]*Ds[2] + P1[2]*Ds[4];
    PT_Ds[3] = P1[0]*Ds[1] + P1[1]*Ds[3] + P1[2]*Ds[5];

    // F = PT_Ds * invDm (2x2)
    const double* invDm = d_invDm + tid * 4;
    double F2[4];
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            F2[r*2+c] = PT_Ds[r*2+0]*invDm[0*2+c] + PT_Ds[r*2+1]*invDm[1*2+c];

    const double* fiber = d_fiber_2d + tid * 2;
    double act = d_activation[tid];

    // md = (1-act) * P * F * fiber_direction
    double Ffiber[2];
    Ffiber[0] = F2[0]*fiber[0] + F2[1]*fiber[1];
    Ffiber[1] = F2[2]*fiber[0] + F2[3]*fiber[1];

    double scale = 1.0 - act;
    double md[3];
    md[0] = scale * (P0[0]*Ffiber[0] + P1[0]*Ffiber[1]);
    md[1] = scale * (P0[1]*Ffiber[0] + P1[1]*Ffiber[1]);
    md[2] = scale * (P0[2]*Ffiber[0] + P1[2]*Ffiber[1]);

    int off = d_offset + tid * 3;
    d_d_output[off+0] = md[0];
    d_d_output[off+1] = md[1];
    d_d_output[off+2] = md[2];
}

// ====================================================================
// ConstraintGPU 实现
// ====================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
    } \
} while(0)

namespace FEM {

void ConstraintGPU::UploadFromCPU(const std::vector<Constraint*>& constraints, int /*num_vertices*/) {
    Destroy();

    total_dofs = 0;
    total_scalars = 0;
    UploadCorotate(constraints);
    UploadLinearMuscle(constraints);
    UploadTriangleMuscle(constraints);
    UploadBending(constraints);
    UploadStrain(constraints);
    UploadAttachment(constraints);

    CUDA_CHECK(cudaMalloc(&d_d_output, total_dofs * 3 * sizeof(double)));

    initialized = true;
    std::cout << "[ConstraintGPU] Uploaded constraints: "
              << "corotate=" << corotate.num_tets
              << " linear_muscle=" << linear_muscle.num_muscles
              << " tri_muscle=" << tri_muscle.num_tris
              << " bending=" << bending.num_bends
              << " strain=" << strain.num_tris
              << " attachment=" << attachment.num_attach
              << " total_dofs=" << total_dofs << std::endl;
}

void ConstraintGPU::Destroy() {
    if (d_d_output) { cudaFree(d_d_output); d_d_output = nullptr; }
    // Corotate
    if (corotate.tet_indices) { cudaFree(corotate.tet_indices); corotate.tet_indices = nullptr; }
    if (corotate.invDm)      { cudaFree(corotate.invDm); corotate.invDm = nullptr; }
    if (corotate.volume)     { cudaFree(corotate.volume); corotate.volume = nullptr; }
    if (corotate.stiffness)  { cudaFree(corotate.stiffness); corotate.stiffness = nullptr; }
    if (corotate.stiffness2) { cudaFree(corotate.stiffness2); corotate.stiffness2 = nullptr; }
    if (corotate.poisson_ratio) { cudaFree(corotate.poisson_ratio); corotate.poisson_ratio = nullptr; }
    corotate.num_tets = 0;
    // LinearMuscle
    if (linear_muscle.tet_indices)  { cudaFree(linear_muscle.tet_indices); linear_muscle.tet_indices = nullptr; }
    if (linear_muscle.fiber)        { cudaFree(linear_muscle.fiber); linear_muscle.fiber = nullptr; }
    if (linear_muscle.activation)   { cudaFree(linear_muscle.activation); linear_muscle.activation = nullptr; }
    if (linear_muscle.invDm)        { cudaFree(linear_muscle.invDm); linear_muscle.invDm = nullptr; }
    if (linear_muscle.vol)          { cudaFree(linear_muscle.vol); linear_muscle.vol = nullptr; }
    if (linear_muscle.stiffness)    { cudaFree(linear_muscle.stiffness); linear_muscle.stiffness = nullptr; }
    if (linear_muscle.weight)       { cudaFree(linear_muscle.weight); linear_muscle.weight = nullptr; }
    linear_muscle.num_muscles = 0;
    // TriangleMuscle
    if (tri_muscle.tri_indices)  { cudaFree(tri_muscle.tri_indices); tri_muscle.tri_indices = nullptr; }
    if (tri_muscle.fiber_2d)     { cudaFree(tri_muscle.fiber_2d); tri_muscle.fiber_2d = nullptr; }
    if (tri_muscle.activation)   { cudaFree(tri_muscle.activation); tri_muscle.activation = nullptr; }
    if (tri_muscle.invDm)        { cudaFree(tri_muscle.invDm); tri_muscle.invDm = nullptr; }
    if (tri_muscle.area)         { cudaFree(tri_muscle.area); tri_muscle.area = nullptr; }
    if (tri_muscle.stiffness)    { cudaFree(tri_muscle.stiffness); tri_muscle.stiffness = nullptr; }
    tri_muscle.num_tris = 0;
    // Bending
    if (bending.quad_indices)  { cudaFree(bending.quad_indices); bending.quad_indices = nullptr; }
    if (bending.weights)       { cudaFree(bending.weights); bending.weights = nullptr; }
    if (bending.n)             { cudaFree(bending.n); bending.n = nullptr; }
    if (bending.voronoi_area)  { cudaFree(bending.voronoi_area); bending.voronoi_area = nullptr; }
    if (bending.stiffness)     { cudaFree(bending.stiffness); bending.stiffness = nullptr; }
    bending.num_bends = 0;
    // Strain
    if (strain.tri_indices)  { cudaFree(strain.tri_indices); strain.tri_indices = nullptr; }
    if (strain.invDm)        { cudaFree(strain.invDm); strain.invDm = nullptr; }
    if (strain.area)         { cudaFree(strain.area); strain.area = nullptr; }
    if (strain.stiffness)    { cudaFree(strain.stiffness); strain.stiffness = nullptr; }
    strain.num_tris = 0;
    // Attachment
    if (attachment.indices)   { cudaFree(attachment.indices); attachment.indices = nullptr; }
    if (attachment.target)    { cudaFree(attachment.target); attachment.target = nullptr; }
    if (attachment.stiffness) { cudaFree(attachment.stiffness); attachment.stiffness = nullptr; }
    attachment.num_attach = 0;

    total_dofs = 0;
    initialized = false;
}

void ConstraintGPU::UpdateActivations(const std::vector<Constraint*>& constraints) {
    if (!initialized) return;
    // 更新 LinearMuscle 和 TriangleMuscle 的 activation 值
    std::vector<double> lm_act(linear_muscle.num_muscles);
    std::vector<double> tm_act(tri_muscle.num_tris);

    int lm_idx = 0, tm_idx = 0;
    for (auto* c : constraints) {
        if (c->GetType() == ConstraintType::LINEAR_MUSCLE && lm_idx < linear_muscle.num_muscles) {
            auto* lm = static_cast<LinearMuscleConstraint*>(c);
            lm_act[lm_idx++] = lm->GetActivationLevel();
        } else if (c->GetType() == ConstraintType::TRIANGLE_MUSCLE && tm_idx < tri_muscle.num_tris) {
            auto* tm = static_cast<TriangleMuscleConstraint*>(c);
            tm_act[tm_idx++] = tm->GetActivationLevel();
        }
    }

    if (linear_muscle.num_muscles > 0)
        CUDA_CHECK(cudaMemcpy(linear_muscle.activation, lm_act.data(),
                    linear_muscle.num_muscles * sizeof(double), cudaMemcpyHostToDevice));
    if (tri_muscle.num_tris > 0)
        CUDA_CHECK(cudaMemcpy(tri_muscle.activation, tm_act.data(),
                    tri_muscle.num_tris * sizeof(double), cudaMemcpyHostToDevice));
}

// ---- 统一启动所有约束 GPU kernel ----

static int div_up(int a, int b) { return (a + b - 1) / b; }

void EvaluateDVectorGPU_launch(
    ConstraintGPU& data,
    const double* d_positions)
{
    int bs = 256;

    // 清零 d 向量
    if (data.total_dofs > 0)
        cudaMemset(data.d_d_output, 0, data.total_dofs * 3 * sizeof(double));

    // CorotateFEM (tet)
    if (data.corotate.num_tets > 0) {
        int gs = div_up(data.corotate.num_tets, bs);
        evaluate_corotate_kernel<<<gs, bs>>>(
            d_positions,
            data.corotate.tet_indices, data.corotate.invDm,
            data.corotate.volume, data.corotate.stiffness,
            data.corotate.stiffness2,
            data.d_d_output,
            data.corotate.num_tets, data.corotate.d_offset);
    }

    // LinearMuscle
    if (data.linear_muscle.num_muscles > 0) {
        int gs = div_up(data.linear_muscle.num_muscles, bs);
        evaluate_linear_muscle_kernel<<<gs, bs>>>(
            d_positions,
            data.linear_muscle.tet_indices,
            data.linear_muscle.fiber, data.linear_muscle.activation,
            data.linear_muscle.invDm,
            data.d_d_output,
            data.linear_muscle.num_muscles, data.linear_muscle.d_offset);
    }

    // TriangleMuscle
    if (data.tri_muscle.num_tris > 0) {
        int gs = div_up(data.tri_muscle.num_tris, bs);
        evaluate_tri_muscle_kernel<<<gs, bs>>>(
            d_positions,
            data.tri_muscle.tri_indices,
            data.tri_muscle.fiber_2d, data.tri_muscle.activation,
            data.tri_muscle.invDm,
            data.d_d_output,
            data.tri_muscle.num_tris, data.tri_muscle.d_offset);
    }

    // Strain
    if (data.strain.num_tris > 0) {
        int gs = div_up(data.strain.num_tris, bs);
        evaluate_strain_kernel<<<gs, bs>>>(
            d_positions,
            data.strain.tri_indices, data.strain.invDm,
            data.d_d_output,
            data.strain.num_tris, data.strain.d_offset);
    }

    // Bending
    if (data.bending.num_bends > 0) {
        int gs = div_up(data.bending.num_bends, bs);
        evaluate_bending_kernel<<<gs, bs>>>(
            d_positions,
            data.bending.quad_indices, data.bending.weights,
            data.bending.n,
            data.d_d_output,
            data.bending.num_bends, data.bending.d_offset);
    }

    // Attachment
    if (data.attachment.num_attach > 0) {
        int gs = div_up(data.attachment.num_attach, bs);
        evaluate_attachment_kernel<<<gs, bs>>>(
            data.attachment.indices, data.attachment.target,
            data.d_d_output,
            data.attachment.num_attach, data.attachment.d_offset);
    }

    cudaDeviceSynchronize();
}

// ---- Upload helpers —— 使用 SerializeForGPU 接口 ----

void ConstraintGPU::UploadCorotate(const std::vector<Constraint*>& constraints) {
    int count = 0;
    for (auto* c : constraints) if (c->GetType() == ConstraintType::COROTATE) count++;
    if (count == 0) return;

    corotate.num_tets = count;
    corotate.d_offset = total_dofs * 3;
    total_dofs += count * 6;

    std::vector<int> tet_idx(count * 4);
    std::vector<double> invDm(count * 9);
    std::vector<double> vol(count);
    std::vector<double> stiff(count);
    std::vector<double> stiff2(count);
    std::vector<double> pr(count);

    int idx = 0;
    for (auto* c : constraints) {
        if (c->GetType() != ConstraintType::COROTATE) continue;
        auto* ct = static_cast<CorotateFEMConstraint*>(c);
        std::vector<int> bi; std::vector<double> bd;
        int io = 0, d_o = 0;
        bi.resize(4); bd.resize(13);  // 4 indices + 9 invDm + vol + mu + lambda + pr
        ct->SerializeForGPU(bi, bd, io, d_o);
        tet_idx[idx*4+0] = bi[0]; tet_idx[idx*4+1] = bi[1];
        tet_idx[idx*4+2] = bi[2]; tet_idx[idx*4+3] = bi[3];
        for (int k = 0; k < 9; ++k)  invDm[idx*9+k] = bd[k];
        vol[idx]    = bd[9];
        stiff[idx]  = bd[10];
        stiff2[idx] = bd[11];
        pr[idx]     = bd[12];
        idx++;
    }

    CUDA_CHECK(cudaMalloc(&corotate.tet_indices, count * 4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&corotate.invDm, count * 9 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&corotate.volume, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&corotate.stiffness, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&corotate.stiffness2, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&corotate.poisson_ratio, count * sizeof(double)));

    cudaMemcpy(corotate.tet_indices, tet_idx.data(), count * 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(corotate.invDm, invDm.data(), count * 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(corotate.volume, vol.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(corotate.stiffness, stiff.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(corotate.stiffness2, stiff2.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(corotate.poisson_ratio, pr.data(), count * sizeof(double), cudaMemcpyHostToDevice);
}

void ConstraintGPU::UploadLinearMuscle(const std::vector<Constraint*>& constraints) {
    int count = 0;
    for (auto* c : constraints) if (c->GetType() == ConstraintType::LINEAR_MUSCLE) count++;
    if (count == 0) return;

    linear_muscle.num_muscles = count;
    linear_muscle.d_offset = total_dofs * 3;
    total_dofs += count * 1;  // 1 DOF

    std::vector<int> tet_idx(count * 4);
    std::vector<double> fiber(count * 3), act(count), invDm(count * 9);
    std::vector<double> vol(count), stiff(count), weight(count);

    int idx = 0;
    for (auto* c : constraints) {
        if (c->GetType() != ConstraintType::LINEAR_MUSCLE) continue;
        auto* lm = static_cast<LinearMuscleConstraint*>(c);
        std::vector<int> bi(4); std::vector<double> bd(18);
        int io = 0, d_o = 0;
        lm->SerializeForGPU(bi, bd, io, d_o);
        for (int k = 0; k < 4; ++k) tet_idx[idx*4+k] = bi[k];
        for (int k = 0; k < 3; ++k) fiber[idx*3+k]  = bd[k];
        act[idx]     = bd[3];
        for (int k = 0; k < 9; ++k) invDm[idx*9+k] = bd[4+k];
        vol[idx]     = bd[13];
        stiff[idx]   = bd[14];
        weight[idx]  = bd[15];
        idx++;
    }

    CUDA_CHECK(cudaMalloc(&linear_muscle.tet_indices, count * 4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&linear_muscle.fiber, count * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&linear_muscle.activation, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&linear_muscle.invDm, count * 9 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&linear_muscle.vol, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&linear_muscle.stiffness, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&linear_muscle.weight, count * sizeof(double)));

    cudaMemcpy(linear_muscle.tet_indices, tet_idx.data(), count * 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(linear_muscle.fiber, fiber.data(), count * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(linear_muscle.activation, act.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(linear_muscle.invDm, invDm.data(), count * 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(linear_muscle.vol, vol.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(linear_muscle.stiffness, stiff.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(linear_muscle.weight, weight.data(), count * sizeof(double), cudaMemcpyHostToDevice);
}

void ConstraintGPU::UploadTriangleMuscle(const std::vector<Constraint*>& constraints) {
    int count = 0;
    for (auto* c : constraints) if (c->GetType() == ConstraintType::TRIANGLE_MUSCLE) count++;
    if (count == 0) return;

    tri_muscle.num_tris = count;
    tri_muscle.d_offset = total_dofs * 3;
    total_dofs += count * 1;

    std::vector<int> tri_idx(count * 3);
    std::vector<double> fiber(count * 2), act(count), invDm(count * 4);
    std::vector<double> area(count), stiff(count);

    int idx = 0;
    for (auto* c : constraints) {
        if (c->GetType() != ConstraintType::TRIANGLE_MUSCLE) continue;
        auto* tm = static_cast<TriangleMuscleConstraint*>(c);
        std::vector<int> bi(3); std::vector<double> bd(10);
        int io = 0, d_o = 0;
        tm->SerializeForGPU(bi, bd, io, d_o);
        for (int k = 0; k < 3; ++k) tri_idx[idx*3+k] = bi[k];
        for (int k = 0; k < 2; ++k) fiber[idx*2+k]  = bd[k];
        act[idx]     = bd[2];
        for (int k = 0; k < 4; ++k) invDm[idx*4+k] = bd[3+k];
        area[idx]    = bd[7];
        stiff[idx]   = bd[8];
        idx++;
    }

    CUDA_CHECK(cudaMalloc(&tri_muscle.tri_indices, count * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tri_muscle.fiber_2d, count * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tri_muscle.activation, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tri_muscle.invDm, count * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tri_muscle.area, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tri_muscle.stiffness, count * sizeof(double)));

    cudaMemcpy(tri_muscle.tri_indices, tri_idx.data(), count * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tri_muscle.fiber_2d, fiber.data(), count * 2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(tri_muscle.activation, act.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(tri_muscle.invDm, invDm.data(), count * 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(tri_muscle.area, area.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(tri_muscle.stiffness, stiff.data(), count * sizeof(double), cudaMemcpyHostToDevice);
}

void ConstraintGPU::UploadBending(const std::vector<Constraint*>& constraints) {
    int count = 0;
    for (auto* c : constraints) if (c->GetType() == ConstraintType::BENDING) count++;
    if (count == 0) return;

    bending.num_bends = count;
    bending.d_offset = total_dofs * 3;
    total_dofs += count * 1;

    std::vector<int> quad_idx(count * 4);
    std::vector<double> weights(count * 4), n(count), voronoi(count), stiff(count);

    int idx = 0;
    for (auto* c : constraints) {
        if (c->GetType() != ConstraintType::BENDING) continue;
        auto* tb = static_cast<TriangleBendingConstraint*>(c);
        std::vector<int> bi(4); std::vector<double> bd(8);
        int io = 0, d_o = 0;
        tb->SerializeForGPU(bi, bd, io, d_o);
        for (int k = 0; k < 4; ++k) quad_idx[idx*4+k] = bi[k];
        for (int k = 0; k < 4; ++k) weights[idx*4+k] = bd[k];
        n[idx]       = bd[4];
        voronoi[idx] = bd[5];
        stiff[idx]   = bd[6];
        idx++;
    }

    CUDA_CHECK(cudaMalloc(&bending.quad_indices, count * 4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bending.weights, count * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bending.n, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bending.voronoi_area, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bending.stiffness, count * sizeof(double)));

    cudaMemcpy(bending.quad_indices, quad_idx.data(), count * 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bending.weights, weights.data(), count * 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bending.n, n.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bending.voronoi_area, voronoi.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bending.stiffness, stiff.data(), count * sizeof(double), cudaMemcpyHostToDevice);
}

void ConstraintGPU::UploadStrain(const std::vector<Constraint*>& constraints) {
    int count = 0;
    for (auto* c : constraints) if (c->GetType() == ConstraintType::STRAIN) count++;
    if (count == 0) return;

    strain.num_tris = count;
    strain.d_offset = total_dofs * 3;
    total_dofs += count * 2;  // 2 DOFs

    std::vector<int> tri_idx(count * 3);
    std::vector<double> invDm(count * 4), area(count), stiff(count);

    int idx = 0;
    for (auto* c : constraints) {
        if (c->GetType() != ConstraintType::STRAIN) continue;
        auto* ts = static_cast<TriangleStrainConstraint*>(c);
        std::vector<int> bi(3); std::vector<double> bd(7);
        int io = 0, d_o = 0;
        ts->SerializeForGPU(bi, bd, io, d_o);
        for (int k = 0; k < 3; ++k) tri_idx[idx*3+k] = bi[k];
        for (int k = 0; k < 4; ++k) invDm[idx*4+k] = bd[k];
        area[idx]  = bd[4];
        stiff[idx] = bd[5];
        idx++;
    }

    CUDA_CHECK(cudaMalloc(&strain.tri_indices, count * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&strain.invDm, count * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&strain.area, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&strain.stiffness, count * sizeof(double)));

    cudaMemcpy(strain.tri_indices, tri_idx.data(), count * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(strain.invDm, invDm.data(), count * 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(strain.area, area.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(strain.stiffness, stiff.data(), count * sizeof(double), cudaMemcpyHostToDevice);
}

void ConstraintGPU::UploadAttachment(const std::vector<Constraint*>& constraints) {
    int count = 0;
    for (auto* c : constraints) if (c->GetType() == ConstraintType::ATTACHMENT) count++;
    if (count == 0) return;

    attachment.num_attach = count;
    attachment.d_offset = total_dofs * 3;
    total_dofs += count * 1;

    std::vector<int> idx_vert(count);
    std::vector<double> target(count * 3), stiff(count);

    int idx = 0;
    for (auto* c : constraints) {
        if (c->GetType() != ConstraintType::ATTACHMENT) continue;
        auto* at = static_cast<AttachmentConstraint*>(c);
        std::vector<int> bi(1); std::vector<double> bd(4);
        int io = 0, d_o = 0;
        at->SerializeForGPU(bi, bd, io, d_o);
        idx_vert[idx]     = bi[0];
        target[idx*3+0]   = bd[0];
        target[idx*3+1]   = bd[1];
        target[idx*3+2]   = bd[2];
        stiff[idx]        = bd[3];
        idx++;
    }

    CUDA_CHECK(cudaMalloc(&attachment.indices, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&attachment.target, count * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&attachment.stiffness, count * sizeof(double)));

    cudaMemcpy(attachment.indices, idx_vert.data(), count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(attachment.target, target.data(), count * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(attachment.stiffness, stiff.data(), count * sizeof(double), cudaMemcpyHostToDevice);
}

} // namespace FEM
