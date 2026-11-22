# -*- coding: utf-8 -*-
"""AutoDock/Vina 风格的原子分型

 把蛋白质与配体重原子划分为 16 种打分类型, 每种类型携带一组
 打分函数系数(高斯 1 / 高斯 2 / 斥力 / 疏水 / 氢键),
 与 scoring.vina_sf 配合完成原子对相互作用计算。

 类型说明:
   C_H  脂肪族疏水碳      C_A  芳香碳           C    其他脂肪碳
   N_D  氢键供体氮        NA   氢键受体氮        N    中性氮
   OA   受体氧(羟基/羰基) O    中性氧
   S    疏水硫            SA   受体硫            P    磷
   F/CL/BR/I 卤素         MET  金属离子
"""

from .elements import normalize_element

# 全部打分类型(顺序即 one-hot 编码顺序)
ATOM_TYPES = [
    "C_H", "C_A", "C", "N", "N_D", "NA", "O", "OA",
    "S", "SA", "P", "F", "CL", "BR", "I", "MET",
]
TYPE_INDEX = {t: i for i, t in enumerate(ATOM_TYPES)}

# 氢键供体 / 受体类型集合
DONOR_TYPES = {"N_D"}
ACCEPTOR_TYPES = {"NA", "OA", "SA"}

# 芳香残基侧链中构成共轭环的原子名
AROMATIC_RING_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}

# 疏水性残基(其侧链碳按 C_H 处理)
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "PRO", "MET", "PHE", "TRP"}

# 每种类型的 5 维系数: (gauss1, gauss2, repulsion, hydrophobic, hbond)
# 取值参考 AutoDock Vina 1.1.2 打分函数原子对系数的典型量级
TYPE_COEFFICIENTS = {
    "C_H": (0.60, 0.30, 1.00, 1.00, 0.00),
    "C_A": (0.60, 0.30, 1.00, 0.70, 0.00),
    "C":   (0.50, 0.25, 1.00, 0.40, 0.00),
    "N":   (0.50, 0.25, 1.00, 0.00, 0.20),
    "N_D": (0.50, 0.25, 1.10, 0.00, 1.00),
    "NA":  (0.50, 0.25, 1.10, 0.00, 1.00),
    "O":   (0.45, 0.25, 1.10, 0.00, 0.30),
    "OA":  (0.45, 0.25, 1.10, 0.00, 1.00),
    "S":   (0.60, 0.30, 1.00, 0.90, 0.00),
    "SA":  (0.60, 0.30, 1.10, 0.70, 0.40),
    "P":   (0.50, 0.25, 1.10, 0.20, 0.20),
    "F":   (0.45, 0.25, 0.90, 0.20, 0.30),
    "CL":  (0.60, 0.30, 1.00, 0.55, 0.20),
    "BR":  (0.65, 0.30, 1.05, 0.60, 0.20),
    "I":   (0.70, 0.35, 1.10, 0.65, 0.20),
    "MET": (0.50, 0.25, 1.20, 0.00, 0.10),
}

METAL_ELEMENTS = {"MN", "FE", "ZN", "CU", "MG", "CA", "NA", "K"}


def pair_coefficients(type_a: str, type_b: str):
    """返回原子对 (a,b) 的 5 维打分系数。

    规则: gauss/repulsion 取两者均值;
          hydrophobic 仅在双方均为疏水类型时取均值, 否则为 0;
          hbond 仅在双方构成供体-受体对时取均值, 否则为 0。
    """
    ca = TYPE_COEFFICIENTS[type_a]
    cb = TYPE_COEFFICIENTS[type_b]
    g1 = (ca[0] + cb[0]) / 2.0
    g2 = (ca[1] + cb[1]) / 2.0
    rep = (ca[2] + cb[2]) / 2.0
    hydro = (ca[3] + cb[3]) / 2.0 if (ca[3] > 0 and cb[3] > 0) else 0.0
    hb = (ca[4] + cb[4]) / 2.0 if (ca[4] > 0 and cb[4] > 0) else 0.0
    return g1, g2, rep, hydro, hb


def assign_protein_type(element: str, atom_name: str, resname: str) -> str:
    """蛋白质重原子的启发式分型(不依赖外部工具, 只看元素/原子名/残基名)。

    :param element: 元素符号
    :param atom_name: PDB 原子名, 如 "CB", "OD1", "NZ"
    :param resname: 残基名三字母码, 如 "ALA", "HIS"
    :return: ATOM_TYPES 中的类型字符串
    """
    el = normalize_element(element)
    name = atom_name.strip().upper()
    res = resname.strip().upper()

    if el in METAL_ELEMENTS:
        return "MET"
    if el == "C":
        ring = AROMATIC_RING_ATOMS.get(res)
        if ring and name in ring:
            return "C_A"
        if res in HYDROPHOBIC_RESIDUES and name.startswith(("C", "S")):
            return "C_H"
        if name in ("CB", "CG", "CD", "CE", "CZ") or name.startswith(("CD", "CE", "CG")):
            # 侧链脂肪碳: 亲水残基记为普通 C
            return "C_H" if res in ("ALA", "VAL", "LEU", "ILE", "PRO") else "C"
        return "C"
    if el == "N":
        if res == "HIS" and name in ("ND1", "NE2"):
            return "NA"
        if res == "LYS" and name == "NZ":
            return "N_D"
        if res == "ARG" and name in ("NE", "NH1", "NH2"):
            return "N_D"
        if res == "TRP" and name == "NE1":
            return "N_D"
        return "N"
    if el == "O":
        return "OA"
    if el == "S":
        return "S" if res == "MET" else "SA"
    if el == "P":
        return "P"
    if el == "F":
        return "F"
    if el == "CL":
        return "CL"
    if el == "BR":
        return "BR"
    if el == "I":
        return "I"
    return "C"
