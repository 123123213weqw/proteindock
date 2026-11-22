# -*- coding: utf-8 -*-
"""蛋白质结构输入输出

 解析 PDB 文件的标准氨基酸重原子, 完成原子分型;
 提供基于占据空间分析的口袋检测(盲对接时自动寻找最大空腔)。
"""

from dataclasses import dataclass, field
import numpy as np

from ..chemistry.atom_types import (
    ATOM_TYPES,
    TYPE_INDEX,
    assign_protein_type,
)
from ..chemistry.elements import normalize_element, vdw_radius

STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "ASH", "GLH", "HID", "HIE", "HIP", "CYX",
}


@dataclass
class Protein:
    """蛋白质结构对象(重原子表示)。"""

    coords: np.ndarray          # (N,3) 原子坐标
    types: list                 # (N,) 打分类型字符串
    type_ids: np.ndarray        # (N,) 类型索引
    elements: list              # (N,) 元素符号
    atom_names: list            # (N,) PDB 原子名
    res_names: list             # (N,) 残基名
    res_ids: list               # (N,) 残基编号
    chains: list                # (N,) 链标识
    source: str = ""            # 来源文件

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    def centroid(self) -> np.ndarray:
        return self.coords.mean(axis=0)

    def pocket_atoms_mask(self, center: np.ndarray, radius: float = 10.0) -> np.ndarray:
        """返回距口袋中心 radius 以内的原子布尔掩码。"""
        d2 = ((self.coords - np.asarray(center)) ** 2).sum(axis=1)
        return d2 <= radius * radius

    def write_pdb(self, path: str) -> None:
        """把蛋白质(通常为加氢前后处理版本)写回 PDB。"""
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(self.n_atoms):
                x, y, z = self.coords[i]
                fh.write(
                    "ATOM  {:5d} {:>4s} {:3s} {:1s}{:4d}    "
                    "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00          {:>2s}\n".format(
                        i + 1, self.atom_names[i], self.res_names[i],
                        self.chains[i], int(self.res_ids[i]), x, y, z,
                        self.elements[i],
                    )
                )
            fh.write("TER\nEND\n")


def parse_pdb(path: str, keep_hetero: bool = False) -> Protein:
    """解析 PDB 文件, 提取标准氨基酸重原子(可选保留 HETATM 中的非水杂原子)。

    :param path: PDB 文件路径
    :param keep_hetero: 是否保留金属等 HETATM 记录(忽略水与常见辅因子)
    :return: Protein 对象
    """
    coords, types, elements = [], [], []
    atom_names, res_names, res_ids, chains = [], [], [], []
    skip_res = {"HOH", "WAT", "DOD", "SOL", "HEM", "NAG", "MAN", "SO4", "PO4", "GOL"}

    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            res = line[17:20].strip().upper()
            if rec == "HETATM":
                if not keep_hetero or res in skip_res:
                    continue
            elif res not in STANDARD_RESIDUES:
                continue
            name = line[12:16].strip()
            element = line[76:78].strip()
            if not element:
                # 依据原子名推断元素
                element = "".join(ch for ch in name if ch.isalpha())[:1]
            element = normalize_element(element)
            if element == "H":
                continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            chain = line[21].strip() or "A"
            try:
                res_id = int(line[22:26])
            except ValueError:
                res_id = 0
            coords.append([x, y, z])
            types.append(assign_protein_type(element, name, res))
            elements.append(element)
            atom_names.append(name)
            res_names.append(res)
            res_ids.append(res_id)
            chains.append(chain)

    if not coords:
        raise ValueError("PDB 文件中未找到可用原子: {}".format(path))

    return Protein(
        coords=np.array(coords, dtype=np.float64),
        types=types,
        type_ids=np.array([TYPE_INDEX[t] for t in types], dtype=np.int64),
        elements=elements,
        atom_names=atom_names,
        res_names=res_names,
        res_ids=res_ids,
        chains=chains,
        source=path,
    )


# ---------------------------------------------------------------- 口袋检测
def detect_pocket(protein: Protein, spacing: float = 1.0,
                  probe_radius: float = 1.6, min_dirs: int = 3) -> np.ndarray:
    """LigSite 风格的口袋检测(盲对接)。

    原理(Hendlich et al., 1997):
      1. 把蛋白包围盒体素化为 occupied(被重原子+探针半径覆盖)/free;
      2. 沿 ±x/±y/±z 六个方向扫描, 一个 free 体素若在某方向上
         一侧遇到蛋白、另一侧通向外部溶剂, 则该方向记为" pocket ";
      3. 六个方向的 pocket 计数 >= min_dirs 的体素构成结合口袋
         (表面凹陷被多方向夹住, 而蛋白内部空隙两侧都是蛋白、
          平坦表面几乎不被夹住);
      4. 返回最高得分体素簇的质心。

    :return: 口袋中心坐标 (3,)
    """
    coords = protein.coords
    pad = 2.5
    lo = coords.min(axis=0) - pad
    hi = coords.max(axis=0) + pad
    n_cells = np.ceil((hi - lo) / spacing).astype(int) + 1
    if float(np.prod(n_cells)) > 6e6:            # 超大蛋白保护
        spacing = float(np.cbrt(np.prod(hi - lo) / 2e6))
        n_cells = np.ceil((hi - lo) / spacing).astype(int) + 1

    # 体素化: 最近重原子距离 < vdW + probe 即占用(KD-tree 加速)
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    axis_pts = [lo[i] + np.arange(n_cells[i]) * spacing for i in range(3)]
    gx, gy, gz = np.meshgrid(*axis_pts, indexing="ij")
    vox = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    nearest, _ = tree.query(vox, k=1)
    mean_vdw = float(np.mean([vdw_radius(e) for e in protein.elements]))
    occ = (nearest <= mean_vdw + probe_radius).reshape(tuple(n_cells))
    free = ~occ

    # 六方向 LigSite 扫描: free 体素在某方向上两侧均被蛋白夹住
    pocket_votes = np.zeros_like(free, dtype=np.int8)
    for axis in range(3):
        cum = np.cumsum(occ, axis=axis)
        tot_idx = [slice(None)] * 3
        tot_idx[axis] = slice(-1, None)
        total = cum[tuple(tot_idx)]              # 该轴占用总数(axis 维为 1)
        before = cum - occ                       # 该点之前的占用数
        after = total - cum                      # 该点之后的占用数
        votes = free & (before > 0) & (after > 0)
        pocket_votes += votes.astype(np.int8)    # 正方向已覆盖双向语义

    candidates = pocket_votes >= min_dirs
    if not candidates.any():
        candidates = pocket_votes >= 2
        if not candidates.any():
            return protein.centroid()
    # 取最高得分子簇: 以得分最高体素为中心半径 4 A 内加权质心
    best_flat = np.argmax(pocket_votes * candidates)
    bi, bj, bk = np.unravel_index(best_flat, pocket_votes.shape)
    center_voxel = np.array([axis_pts[0][bi], axis_pts[1][bj],
                             axis_pts[2][bk]])
    # 邻域 4A 内候选点按票数加权
    lo_i = np.stack(np.meshgrid(
        *[np.arange(n_cells[i]) for i in range(3)], indexing="ij"), axis=-1)
    d2 = ((lo_i * spacing + lo - center_voxel) ** 2).sum(axis=-1)
    near = candidates & (d2 <= 16.0)
    if not near.any():
        return center_voxel
    weights = pocket_votes[near].astype(float)
    pts_voxel = lo_i[near] * spacing + lo
    return (pts_voxel * weights[:, None]).sum(axis=0) / weights.sum()
