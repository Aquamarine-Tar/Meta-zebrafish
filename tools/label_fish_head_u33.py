#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np


def read_medit_mesh(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    vi = lines.index("Vertices")
    nv = int(lines[vi + 1])
    vertices = []
    for line in lines[vi + 2 : vi + 2 + nv]:
        fields = line.split()
        vertices.append([float(fields[0]), float(fields[1]), float(fields[2])])

    ti = lines.index("Tetrahedra")
    nt = int(lines[ti + 1])
    tets = []
    for line in lines[ti + 2 : ti + 2 + nt]:
        fields = line.split()
        tets.append([int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])])

    return lines, np.asarray(vertices, dtype=float), np.asarray(tets, dtype=np.int64), ti, nt


def compute_u(vertices: np.ndarray, tets_1_based: np.ndarray):
    center = vertices.mean(axis=0)
    centered = vertices - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[2] < 0.0:
        axis = -axis

    vertex_s = centered @ axis
    s_min = float(vertex_s.min())
    s_max = float(vertex_s.max())

    tet_centers = vertices[tets_1_based - 1].mean(axis=1)
    tet_s = (tet_centers - center) @ axis
    tet_u = (tet_s - s_min) / (s_max - s_min)
    return axis, s_min, s_max, tet_u


def write_labeled_mesh(lines, tets_1_based, tet_u, tetrahedra_index, ntets, cut_u, out: Path):
    labels = np.where(tet_u <= cut_u, 1, 0)
    out_lines = list(lines)
    first_tet_line = tetrahedra_index + 2
    for i in range(ntets):
        a, b, c, d = tets_1_based[i]
        out_lines[first_tet_line + i] = f"{a} {b} {c} {d} {int(labels[i])}"

    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Label fish head tets by PCA-axis u <= cut.")
    parser.add_argument("--mesh", type=Path, default=Path("data/fish_body.mesh"))
    parser.add_argument("--out", type=Path, default=Path("data/fish_body_head_u033.mesh"))
    parser.add_argument("--cut-u", type=float, default=0.33)
    args = parser.parse_args()

    lines, vertices, tets, tetrahedra_index, ntets = read_medit_mesh(args.mesh)
    axis, s_min, s_max, tet_u = compute_u(vertices, tets)
    labels = write_labeled_mesh(lines, tets, tet_u, tetrahedra_index, ntets, args.cut_u, args.out)

    print(f"Wrote {args.out}")
    print(f"PCA axis: {axis}")
    print(f"s range: {s_min:.6f} .. {s_max:.6f}")
    print(f"cut_u: {args.cut_u:.6f}")
    print(f"label 1 head tets: {int((labels == 1).sum())}")
    print(f"label 0 body tets: {int((labels == 0).sum())}")


if __name__ == "__main__":
    main()
