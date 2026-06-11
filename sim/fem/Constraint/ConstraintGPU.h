#ifndef __CONSTRAINT_GPU_H__
#define __CONSTRAINT_GPU_H__

#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace FEM {

/// 约束类型枚举（与 CPU ConstraintType 一一对应）
/// 0=ATTACHMENT, 1=COROTATE, 2=LINEAR_MUSCLE,
/// 3=BENDING, 4=STRAIN, 5=TRIANGLE_MUSCLE
enum ConstraintTypeGPU { CT_ATTACHMENT=0, CT_COROTATE=1, CT_LINEAR_MUSCLE=2,
                         CT_BENDING=3, CT_STRAIN=4, CT_TRIANGLE_MUSCLE=5 };

/// -------------------------------------------------------------------
/// 各约束类型的 SoA 数据（一次性从 CPU 提取，常驻 GPU Device 内存）
/// -------------------------------------------------------------------

// CorotateFEM (tet 约束): 每个 tet 有 4 个顶点索引 + 材料参数
struct CorotateFEMData {
    int* tet_indices;       // [4 * num_tets] 全局顶点索引
    double* invDm;           // [9 * num_tets] 列主序 3×3 invDm
    double* volume;          // [num_tets]
    double* stiffness;       // [num_tets] = mu
    double* stiffness2;      // [num_tets] = lambda
    double* poisson_ratio;   // [num_tets]
    int num_tets;
    int d_offset;            // 在全局 d 数组中的起始偏移（每 tet 6 DOFs = 18 scalars）
};

// LinearMuscleConstraint (四面体肌肉线约束，1 DOF)
struct LinearMuscleData {
    int* tet_indices;       // [4 * num_muscles]
    double* fiber;           // [3 * num_muscles] 纤维方向
    double* activation;      // [num_muscles] 激活水平（每帧更新）
    double* invDm;           // [9 * num_muscles]
    double* vol;             // [num_muscles]
    double* stiffness;       // [num_muscles]
    double* weight;          // [num_muscles]
    int num_muscles;
    int d_offset;            // 每 muscle 1 DOF = 3 scalars
};

// TriangleMuscleConstraint (三角面肌肉约束，1 DOF)
struct TriangleMuscleData {
    int* tri_indices;       // [3 * num_tris]
    double* fiber_2d;        // [2 * num_tris]
    double* activation;      // [num_tris]
    double* invDm;           // [4 * num_tris]
    double* area;            // [num_tris]
    double* stiffness;       // [num_tris]
    int num_tris;
    int d_offset;
};

// TriangleBendingConstraint (三角面弯曲，1 DOF)
struct BendingData {
    int* quad_indices;      // [4 * num_bends]
    double* weights;         // [4 * num_bends]
    double* n;               // [num_bends] 静息二面角
    double* voronoi_area;    // [num_bends]
    double* stiffness;       // [num_bends]
    int num_bends;
    int d_offset;
};

// TriangleStrainConstraint (三角面应变，2 DOFs)
struct StrainData {
    int* tri_indices;       // [3 * num_tris]
    double* invDm;           // [4 * num_tris]
    double* area;            // [num_tris]
    double* stiffness;       // [num_tris]
    int num_tris;
    int d_offset;
};

// AttachmentConstraint (固定点约束，1 DOF)
struct AttachmentData {
    int* indices;           // [num_attach] 顶点索引
    double* target;          // [3 * num_attach] 目标位置
    double* stiffness;       // [num_attach]
    int num_attach;
    int d_offset;
};

// -------------------------------------------------------------------
// 聚合所有 GPU 约束数据
// -------------------------------------------------------------------
struct ConstraintGPU {
    CorotateFEMData   corotate;
    LinearMuscleData  linear_muscle;
    TriangleMuscleData tri_muscle;
    BendingData       bending;
    StrainData        strain;
    AttachmentData    attachment;

    // 全局 d 向量 (device side), 长度 = 3 * total_dofs (标量数)
    double* d_d_output;
    int total_dofs;       // sum of all constraint GetDof()
    int total_scalars;    // total_dofs * 3

    bool initialized;

    ConstraintGPU() : initialized(false), d_d_output(nullptr), total_dofs(0) {}
    ~ConstraintGPU() { Destroy(); }

    /// 从 CPU 侧的 constraints 列表一次性提取并上传所有约束数据到 GPU
    /// @param constraints World::GetConstraints()
    /// @param num_vertices 总顶点数
    void UploadFromCPU(const std::vector<class Constraint*>& constraints, int num_vertices);

    /// 释放所有 GPU 内存
    void Destroy();

    /// 从 CPU 更新肌肉激活水平（每帧调用）
    void UpdateActivations(const std::vector<class Constraint*>& constraints);

private:
    void UploadCorotate(const std::vector<class Constraint*>& constraints);
    void UploadLinearMuscle(const std::vector<class Constraint*>& constraints);
    void UploadTriangleMuscle(const std::vector<class Constraint*>& constraints);
    void UploadBending(const std::vector<class Constraint*>& constraints);
    void UploadStrain(const std::vector<class Constraint*>& constraints);
    void UploadAttachment(const std::vector<class Constraint*>& constraints);
};

} // namespace FEM

// 声明：GPU 约束投影统一启动函数（定义在 ConstraintGPU.cu）
namespace FEM {
void EvaluateDVectorGPU_launch(ConstraintGPU& data, const double* d_positions);
}

#endif
