# -*- coding: utf-8 -*-
"""打分网格(Scoring Grid)预计算加速

 把蛋白对每种配体原子类型的相互作用能预涂到以口袋为中心的
 0.375 A 间距三维网格上; 构象搜索阶段对配体原子做三线性插值
 即可 O(1) 取得相互作用能, 避免每次评估全量原子对距离矩阵。
"""

import numpy as np

from ..chemistry.atom_types import ATOM_TYPES, TYPE_INDEX
from .vina_sf import (
    INTERACTION_CUTOFF,
    W_GAUSS1,
    W_GAUSS2,
    W_HBOND,
    W_HYDROPHOBIC,
    W_REPULSION,
    _COEFF_TENSOR,
    _VDW_VECTOR,
)

GRID_SPACING = 0.375          # 网格间距 / A (与 AutoDock 一致)
DEFAULT_BOX_SIZE = 24.0       # 盒子边长 / A
BOUNDARY_PENALTY = 50.0       # 原子越界惩罚(每原子)


class ScoreGrid:
    """口袋打分网格: 16 个类型通道的能量场。"""

    def __init__(self, center, size=DEFAULT_BOX_SIZE, spacing=GRID_SPACING):
        self.center = np.asarray(center, dtype=np.float64)
        self.size = float(size)
        self.spacing = float(spacing)
        self.n_cells = int(np.floor(self.size / self.spacing))
        half = (self.n_cells - 1) * self.spacing / 2.0
        self.origin = self.center - half          # 格点 0 的坐标
        # (n_cells, n_cells, n_cells, n_types) 能量场
        self.fields = np.zeros((self.n_cells, self.n_cells, self.n_cells,
                                len(ATOM_TYPES)), dtype=np.float32)
        self._built = False

    # ------------------------------------------------------------ 构建
    def build(self, rec_coords, rec_type_ids, verbose=False):
        """把蛋白原子相互作用贡献涂到网格上。

        对每个蛋白原子, 在其截断球覆盖的格点上叠加
        五项基元 x 类型对系数 x 全局权重 的能量值。
        """
        rec_coords = np.asarray(rec_coords, dtype=np.float64)
        rec_type_ids = np.asarray(rec_type_ids)
        n_types = len(ATOM_TYPES)

        # 轴向格点坐标(各轴独立: origin 为盒子角落坐标)
        axes = [self.origin[i] + np.arange(self.n_cells) * self.spacing
                for i in range(3)]
        cut_cells = int(np.ceil(INTERACTION_CUTOFF / self.spacing))
        cutoff2 = INTERACTION_CUTOFF ** 2

        for ai in range(len(rec_coords)):
            pos = rec_coords[ai]
            # 快速丢弃: 原子到盒子最 近/远 距离
            if np.abs(pos - self.center).max() > self.size / 2 + INTERACTION_CUTOFF:
                continue
            ri = _VDW_VECTOR[rec_type_ids[ai]]
            i0, j0, k0 = np.floor((pos - self.origin) / self.spacing).astype(int)
            lo = max(i0 - cut_cells, 0); hi = min(i0 + cut_cells + 1, self.n_cells)
            lo_j = max(j0 - cut_cells, 0); hi_j = min(j0 + cut_cells + 1, self.n_cells)
            lo_k = max(k0 - cut_cells, 0); hi_k = min(k0 + cut_cells + 1, self.n_cells)
            if lo >= hi or lo_j >= hi_j or lo_k >= hi_k:
                continue

            gx = axes[0][lo:hi] - pos[0]
            gy = axes[1][lo_j:hi_j] - pos[1]
            gz = axes[2][lo_k:hi_k] - pos[2]
            dx, dy, dz = np.meshgrid(gx, gy, gz, indexing="ij")
            d2 = dx * dx + dy * dy + dz * dz
            mask = d2 < cutoff2
            if not mask.any():
                continue
            d = np.sqrt(d2)

            # 基元值 (M1,M2,M3)
            g1 = np.exp(-(d / 0.5) ** 2)
            g2 = np.exp(-((d - 3.0) / 2.0) ** 2)
            # 斥力: 距离小于 0.9 倍 vdW 半径和(穿透)时惩罚
            r_pair = ri + _VDW_VECTOR                      # (n_types,)
            rep_cut = 0.9 * r_pair
            dexp = d[..., None]                            # (M..., 1)
            rep = np.where(dexp < rep_cut,
                           (rep_cut / np.maximum(dexp, 1e-6) - 1.0) ** 2,
                           0.0)
            ds = dexp - r_pair                             # 表面距离 (M..., n_types)
            hydro = _linear_step_batch(ds, 0.5, 2.0)      # (M..., n_types)
            hb = _linear_step_batch(ds, -0.7, 0.0)

            coeff = _COEFF_TENSOR[:, rec_type_ids[ai]]    # (n_types, 5)
            # 每类型能量: sum_k w_k * c_k * prim_k
            e = (coeff[:, 0] * W_GAUSS1) * g1[..., None] \
                + (coeff[:, 1] * W_GAUSS2) * g2[..., None] \
                + (coeff[:, 2] * W_REPULSION) * rep \
                + (coeff[:, 3] * W_HYDROPHOBIC) * hydro \
                + (coeff[:, 4] * W_HBOND) * hb
            block = np.where(mask[..., None], e, 0.0)
            self.fields[lo:hi, lo_j:hi_j, lo_k:hi_k] += block.astype(np.float32)
            if verbose and ai % 500 == 0:
                print("  grid building {}/{}".format(ai, len(rec_coords)))

        self._built = True
        return self

    # ------------------------------------------------------------ 查询
    def score(self, coords, type_ids) -> float:
        """对一组配体原子做三线性插值求相互作用能总和。"""
        if not self._built:
            raise RuntimeError("网格尚未 build()")
        coords = np.asarray(coords, dtype=np.float64)
        u = (coords - self.origin) / self.spacing
        in_bounds = ((u >= 0) & (u <= self.n_cells - 1)).all(axis=1)
        if not in_bounds.all():
            # 越界原子: 盒内部分正常打分, 盒外按惩罚计
            total = BOUNDARY_PENALTY * float((~in_bounds).sum())
            if not in_bounds.any():
                return total
        else:
            total = 0.0
        u = u[in_bounds]
        ids = np.asarray(type_ids)[in_bounds]

        i0 = np.floor(u).astype(int)
        frac = u - i0
        i1 = np.minimum(i0 + 1, self.n_cells - 1)
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        total_field = 0.0
        for atom_i in range(len(u)):
            a, b, c = i0[atom_i]
            d, e, f = i1[atom_i]
            corner = self.fields
            v000 = corner[a, b, c, ids[atom_i]]; v100 = corner[d, b, c, ids[atom_i]]
            v010 = corner[a, e, c, ids[atom_i]]; v110 = corner[d, e, c, ids[atom_i]]
            v001 = corner[a, b, f, ids[atom_i]]; v101 = corner[d, b, f, ids[atom_i]]
            v011 = corner[a, e, f, ids[atom_i]]; v111 = corner[d, e, f, ids[atom_i]]
            wx, wy, wz = fx[atom_i], fy[atom_i], fz[atom_i]
            val = (v000 * (1 - wx) * (1 - wy) * (1 - wz)
                   + v100 * wx * (1 - wy) * (1 - wz)
                   + v010 * (1 - wx) * wy * (1 - wz)
                   + v110 * wx * wy * (1 - wz)
                   + v001 * (1 - wx) * (1 - wy) * wz
                   + v101 * wx * (1 - wy) * wz
                   + v011 * (1 - wx) * wy * wz
                   + v111 * wx * wy * wz)
            total_field += float(val)
        return total + total_field

    def box_extent(self):
        """返回盒子两角坐标, 供可视化。"""
        half = self.size / 2.0
        return self.center - half, self.center + half


def _linear_step_batch(values, x_low, x_high):
    """_linear_step 的批量版, values 形状 (..., K) 对 K 种类型。"""
    out = np.zeros_like(values)
    if x_high > x_low:
        out[values < x_low] = 1.0
        band = (values >= x_low) & (values <= x_high)
        out[band] = (x_high - values[band]) / (x_high - x_low)
    else:
        out[values < x_high] = 1.0
    return out
