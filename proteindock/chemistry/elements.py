# -*- coding: utf-8 -*-
"""元素周期参数表

 提供常见生物体系元素的范德华半径、相对原子质量与 Pauling 电负性,
 供原子分型、碰撞检测与打分函数使用。
"""

# 元素符号 -> (vdW 半径 /A, 相对原子质量, 电负性)
ELEMENT_TABLE = {
    "H": (1.10, 1.008, 2.20),
    "B": (1.92, 10.811, 2.04),
    "C": (1.70, 12.011, 2.55),
    "N": (1.55, 14.007, 3.04),
    "O": (1.52, 15.999, 3.44),
    "F": (1.47, 18.998, 3.98),
    "SI": (2.10, 28.085, 1.90),
    "P": (1.80, 30.974, 2.19),
    "S": (1.80, 32.060, 2.58),
    "CL": (1.75, 35.450, 3.16),
    "BR": (1.85, 79.904, 2.96),
    "I": (1.98, 126.904, 2.66),
    # 常见金属离子(仅用于蛋白侧建模体识别)
    "MN": (2.05, 54.938, 1.55),
    "FE": (2.05, 55.845, 1.83),
    "ZN": (2.10, 65.380, 1.65),
    "CU": (2.00, 63.546, 1.90),
    "MG": (2.05, 24.305, 1.31),
    "CA": (2.30, 40.078, 1.00),
    "NA": (2.30, 22.990, 0.93),
    "K": (2.80, 39.098, 0.82),
}

# 支持参与对接打分的重元素(不含氢: Vina 式打分使用非氢原子表示)
DOCKABLE_ELEMENTS = set(ELEMENT_TABLE) - {"H"}


def normalize_element(symbol: str) -> str:
    """把 PDB/SDF 中可能出现的元素写法统一为大写规范形式。

    :param symbol: 原始元素符号, 如 "c", "CA", "Cl"
    :return: 规范化元素符号, 未知元素返回 "C" 并保持兼容
    """
    sym = symbol.strip().upper()
    if sym in ELEMENT_TABLE:
        return sym
    # 尝试首字母匹配(例如 "c1" 之类的 aromatic 标记已在上游剥离)
    if sym[:1] in ELEMENT_TABLE:
        return sym[:1]
    return "C"


def vdw_radius(element: str) -> float:
    """返回元素范德华半径, 未知元素回退为碳的半径。"""
    return ELEMENT_TABLE.get(normalize_element(element), ELEMENT_TABLE["C"])[0]


def atomic_mass(element: str) -> float:
    """返回元素相对原子质量。"""
    return ELEMENT_TABLE.get(normalize_element(element), ELEMENT_TABLE["C"])[1]


def electronegativity(element: str) -> float:
    """返回元素 Pauling 电负性, 用于氢键供受体判定。"""
    return ELEMENT_TABLE.get(normalize_element(element), ELEMENT_TABLE["C"])[2]
