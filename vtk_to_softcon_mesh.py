from pathlib import Path

import meshio
import numpy as np


def extract_tetra_cells(mesh: meshio.Mesh) -> np.ndarray:
    """
    从 meshio 读取结果中提取四面体单元。

    参数:
        mesh: meshio 读入的网格对象，可能包含多种单元类型（tetra/triangle/line 等）。
    返回:
        形状为 (N, 4) 的整型数组，每行是一个四面体的 4 个顶点索引（0-based）。
    """
    # VTK 里单元可能不止一种，这里只取 softcon 需要的 tetra。
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            return np.asarray(cell_block.data, dtype=np.int64)
    raise ValueError("No tetra cells found in input mesh.")


def extract_boundary_triangles(tets: np.ndarray) -> np.ndarray:
    """
    由四面体集合提取“边界三角面”。

    核心思路:
    - 每个四面体有 4 个三角面；
    - 内部面会被相邻两个四面体共享（出现 2 次）；
    - 外表面只出现 1 次。
    """
    # 枚举每个 tet 的 4 个三角面（保持原索引顺序，后续写入 Triangles）。
    f0 = tets[:, [0, 1, 2]]
    f1 = tets[:, [0, 1, 3]]
    f2 = tets[:, [0, 2, 3]]
    f3 = tets[:, [1, 2, 3]]
    faces = np.vstack([f0, f1, f2, f3])

    # 用排序后的面作为“无方向键”计数，避免 [1,2,3] 和 [2,1,3] 被当成不同面。
    faces_sorted = np.sort(faces, axis=1)
    _, inv, counts = np.unique(
        faces_sorted, axis=0, return_inverse=True, return_counts=True
    )
    # 只保留出现 1 次的面，即边界面。
    boundary_mask = counts[inv] == 1
    return faces[boundary_mask]


def write_medit_mesh(
    path: Path, points: np.ndarray, triangles: np.ndarray, tets: np.ndarray
) -> None:
    """
    将点、边界三角面、四面体写成 Medit .mesh 文本格式。

    说明:
    - 项目里的读取器（MeditMesh.cpp）要求 1-based 索引；
    - 所以写出 Triangles / Tetrahedra 时要把 numpy 的 0-based 索引 +1。
    """
    with path.open("w", encoding="utf-8") as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n")
        f.write("Vertices\n")
        f.write(f"{len(points)}\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} 0\n")

        f.write("Triangles\n")
        f.write("0\n")

        f.write("Tetrahedra\n")
        f.write(f"{len(tets)}\n")
        for tet in tets:
            a, b, c, d = tet + 1
            f.write(f"{a} {b} {c} {d} 0\n")

        f.write("End")


def pick_sampling_indices(points: np.ndarray, n_samples: int = 150) -> np.ndarray:
    """
    沿主轴方向均匀抽样，生成 fish.sampling 所需的顶点索引。

    逻辑:
    1) 对点云中心化；
    2) 用 SVD 求第一主方向（近似鱼体头尾方向）；
    3) 将所有点投影到主轴；
    4) 在投影区间做等分分箱；
    5) 每个箱取最接近箱中心的一个顶点索引。

    参数:
        points: 顶点坐标数组，形状 (V, 3)。
        n_samples: 期望采样段数，默认 150。
    返回:
        采样点顶点索引数组（0-based，已按顺序去重）。
    """
    # c: 点云几何中心；x: 中心化后的坐标。
    c = points.mean(axis=0, keepdims=True)
    x = points - c
    # vh[0] 对应最大方差方向（第一主轴）。
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    axis = vh[0]

    # s 是每个顶点在主轴上的“轴向坐标”。
    s = x @ axis
    s_min, s_max = s.min(), s.max()
    bins = np.linspace(s_min, s_max, n_samples + 1)

    indices = []
    for i in range(n_samples):
        lo, hi = bins[i], bins[i + 1]
        mask = (s >= lo) & (s <= hi)
        cand = np.where(mask)[0]
        if len(cand) == 0:
            # 某个箱没有点时跳过，避免空索引报错。
            continue
        target = 0.5 * (lo + hi)
        # 选箱内最接近箱中心的点，保证轴向分布尽量均匀。
        j = cand[np.argmin(np.abs(s[cand] - target))]
        indices.append(int(j))

    # 去重并保持原顺序，避免同一顶点在相邻箱被重复选中。
    uniq = list(dict.fromkeys(indices))
    return np.array(uniq, dtype=np.int64)


def write_sampling(path: Path, sample_idx: np.ndarray) -> None:
    """将采样索引写成一行整数（空格分隔），与现有 .sampling 格式一致。"""
    with path.open("w", encoding="utf-8") as f:
        f.write(" ".join(map(str, sample_idx.tolist())) + "\n")


if __name__ == "__main__":
    # 以脚本文件位置推导仓库根目录，避免依赖当前工作目录。
    repo_root = Path(__file__).resolve().parent.parent
    vtk_path = repo_root / "Model_v2.vtk"
    mesh_path = repo_root / "Metaworm_zfver" / "data" / "fish_body.mesh"
    sampling_path = repo_root / "Metaworm_zfver" / "data" / "fish.sampling"

    # 读取原始 VTK 体网格。
    m = meshio.read(vtk_path)
    pts = np.asarray(m.points, dtype=float)
    tets = extract_tetra_cells(m)
    tris = extract_boundary_triangles(tets)

    # 写出 softcon 可读的 .mesh 和 .sampling。
    write_medit_mesh(mesh_path, pts, tris, tets)
    idx = pick_sampling_indices(pts, n_samples=150)
    write_sampling(sampling_path, idx)

    print(f"Done: {mesh_path}, {sampling_path}, sampling={len(idx)}")
