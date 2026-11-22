# -*- coding: utf-8 -*-
"""Vina 式多项经验打分函数

 复现 AutoDock Vina (Trott & Olson, 2010) 五项原子对相互作用:
   gauss1     宽势阱高斯吸引项
   gauss2     窄势阱高斯吸引项
   repulsion  近距立方倒数斥力项
   hydrophobic 疏水接触阶跃-线性项
   hbond      氢键阶跃-线性项(方向性不显式建模, 由供/受体类型对承载)

 实现说明:
   1. 全部原子对相互作用向量化, 16x16 类型对的系数预先展开成
      (16,16,5) 张量, 以类型索引直接取用, 无 Python 逐对循环;
   2. hydrophobic / hbond 基元使用"表面距离" ds = d - ri - rj
      (原子间距减去双方范德华半径), 与 Vina 的平滑距离一致;
   3. 组合权重取 Vina 1.1.2 拟合值。
"""

import numpy as np

from ..chemistry.atom_types import (
    ATOM_TYPES,
    TYPE_COEFFICIENTS,
    TYPE_INDEX,
    pair_coefficients,
)

# Vina 1.1.2 全局权重
W_GAUSS1 = -0.035579
W_GAUSS2 = -0.005156
W_REPULSION = 0.840245
W_HYDROPHOBIC = -0.035069
W_HBOND = -0.587439
W_TORSION = 0.054641        # 每个可旋转键的构象熵惩罚

INTERACTION_CUTOFF = 8.0    # 原子对截断 / A
N_TYPES = len(ATOM_TYPES)

# 每种打分类型的代表 vdW 半径 / A
TYPE_VDW = {
    "C_H": 1.70, "C_A": 1.70, "C": 1.70, "N": 1.55, "N_D": 1.55,
    "NA": 1.55, "O": 1.52, "OA": 1.52, "S": 1.80, "SA": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "MET": 2.05,
}

# 全局权重向量 (与系数 5 元组同序)
_WEIGHTS = np.array([W_GAUSS1, W_GAUSS2, W_REPULSION, W_HYDROPHOBIC, W_HBOND])


def _build_coefficient_tensor():
    """展开 (16,16,5) 原子对系数张量(模块级缓存)。"""
    tensor = np.zeros((N_TYPES, N_TYPES, 5))
    for i, a in enumerate(ATOM_TYPES):
        for j, b in enumerate(ATOM_TYPES):
            tensor[i, j] = pair_coefficients(a, b)
    return tensor


_COEFF_TENSOR = _build_coefficient_tensor()
_VDW_VECTOR = np.array([TYPE_VDW[t] for t in ATOM_TYPES])


def _linear_step(values, x_low, x_high):
    """[x_low, x_high] 内 1->0 线性衰减的阶跃基元。"""
    out = np.zeros_like(values)
    if x_high > x_low:
        out[values < x_low] = 1.0
        band = (values >= x_low) & (values <= x_high)
        out[band] = (x_high - values[band]) / (x_high - x_low)
    else:
        out[values < x_high] = 1.0
    return out


def pairwise_term_sum(lig_type_ids, rec_type_ids, d_matrix, cutoff=None):
    """计算两组原子间五项能量之和(全向量化)。

    :param lig_type_ids: (N,) 配体类型索引
    :param rec_type_ids: (M,) 蛋白类型索引
    :param d_matrix: (N,M) 原子间距离矩阵 / A
    :param cutoff: 距离截断, 默认 8 A
    :return: (5,) 各项加权和向量 [gauss1, gauss2, repulsion, hydrophobic, hbond]
    """
    cutoff = INTERACTION_CUTOFF if cutoff is None else cutoff
    lig_ids = np.asarray(lig_type_ids)
    rec_ids = np.asarray(rec_type_ids)
    d = np.asarray(d_matrix, dtype=np.float64)

    close = d < cutoff
    if not close.any():
        return np.zeros(5)

    rows, cols = np.nonzero(close)
    dist = d[rows, cols]
    coeff = _COEFF_TENSOR[lig_ids[rows], rec_ids[cols]]        # (K,5)
    r_pair = _VDW_VECTOR[lig_ids[rows]] + _VDW_VECTOR[rec_ids[cols]]
    # 表面距离(Vina 平滑距离): ds = d - ri - rj
    ds = dist - r_pair

    g1 = np.exp(-(dist / 0.5) ** 2)
    g2 = np.exp(-((dist - 3.0) / 2.0) ** 2)
    # 斥力基元: 仅当原子间距小于 0.9 倍 vdW 半径和(即实际穿透)时惩罚
    rep_cut = 0.9 * r_pair
    rep = np.where(dist < rep_cut,
                   (rep_cut / np.maximum(dist, 1e-6) - 1.0) ** 2, 0.0)
    hydro = _linear_step(ds, 0.5, 2.0)
    hb = _linear_step(ds, -0.7, 0.0)

    primitives = np.stack([g1, g2, rep, hydro, hb], axis=1)     # (K,5)
    weighted = ((coeff * primitives) * _WEIGHTS).sum(axis=0)    # (5,)
    return weighted


def intermolecular_energy(lig_types_ids, rec_types_ids, d_matrix):
    """蛋白-配体相互作用总能量及分解。

    :return: (total_kcal, terms_dict)
    """
    weighted = pairwise_term_sum(lig_types_ids, rec_types_ids, d_matrix)
    terms = {
        "gauss1": float(weighted[0]),
        "gauss2": float(weighted[1]),
        "repulsion": float(weighted[2]),
        "hydrophobic": float(weighted[3]),
        "hbond": float(weighted[4]),
    }
    return float(weighted.sum()), terms


def intramolecular_energy(lig_type_ids, coords, n_torsions=0):
    """配体内部非键相互作用(仅斥力项) + 构象熵惩罚。

    近似处理:
      1. 排除 d < 1.9 A 的成键/1-3 对, 避免把共价几何记为斥力;
      2. 只保留斥力项用于防止自碰撞 —— 不计入分子内吸引项,
         否则构象搜索会退化为「远离蛋白的自折叠构象」。
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    energy = 0.0
    if n >= 2:
        d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        iu = np.triu_indices(n, k=1)
        dd = d[iu]
        valid = (dd >= 1.9) & (dd < INTERACTION_CUTOFF)
        if valid.any():
            pairs = np.stack([iu[0][valid], iu[1][valid]], axis=1)
            sub = pairwise_term_sum(
                np.asarray(lig_type_ids)[pairs[:, 0]],
                np.asarray(lig_type_ids)[pairs[:, 1]],
                d[pairs[:, 0], pairs[:, 1]][:, None],
            )
            energy = float(sub[2])        # 仅斥力分量
    return energy + torsion_penalty(n_torsions)


def torsion_penalty(n_torsions: int) -> float:
    """可旋转键构象熵惩罚。"""
    return W_TORSION * float(n_torsions)
