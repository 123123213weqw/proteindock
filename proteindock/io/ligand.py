# -*- coding: utf-8 -*-
"""小分子配体输入输出

 基于 RDKit 读取 SDF/MOL2/PDB 配体, 自动补 3D 构象(ETKDG),
 完成原子分型与可旋转键检测, 输出对接内部表示 Ligand。
"""

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from ..chemistry.atom_types import TYPE_INDEX
from ..chemistry.elements import normalize_element

# RDKit 生成 3D 构象时的随机种子(保证可复现)
ETKDG_SEED = 2025


class Ligand:
    """配体对象: 重原子坐标 + 打分类型 + 可旋转键。

    :ivar coords: (N,3) 以质心为原点的重原子坐标
    :ivar elements: (N,) 元素符号
    :ivar types: (N,) 打分类型
    :ivar type_ids: (N,) 类型索引
    :ivar rotatable: list[(i, j, movable_atom_ids)] 可旋转键
    :ivar torsions: np.ndarray 当前扭转角(rad), 长度与 rotatable 一致
    """

    def __init__(self, coords, elements, types, rotatable, mol=None, name="ligand"):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.elements = list(elements)
        self.types = list(types)
        self.type_ids = np.array([TYPE_INDEX[t] for t in types], dtype=np.int64)
        self.rotatable = rotatable        # [(begin, end, [movable...])]
        self.torsions = np.zeros(len(rotatable))
        self.name = name
        self._mol = mol                   # 保留 RDKit 分子用于输出

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    @property
    def n_torsions(self) -> int:
        return len(self.rotatable)

    def copy(self) -> "Ligand":
        lig = Ligand(self.coords.copy(), list(self.elements), list(self.types),
                     self.rotatable, self._mol, self.name)
        lig.torsions = self.torsions.copy()
        return lig

    # ------------------------------------------------------------ 几何
    def conformer(self, transform=None) -> np.ndarray:
        """按当前扭转角重建构象, 可再叠加刚体变换。

        :param transform: chemistry.RigidTransform 或 None
        :return: (N,3) 世界坐标
        """
        coords = self.coords.copy()
        for (b, e, movable), angle in zip(self.rotatable, self.torsions):
            axis = coords[e] - coords[b]
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis = axis / norm
            c, s = np.cos(angle), np.sin(angle)
            # Rodrigues 旋转公式
            kx, ky, kz = axis
            K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
            R = np.eye(3) * c + s * K + (1 - c) * np.outer(axis, axis)
            coords[movable] = (coords[movable] - coords[e]) @ R.T + coords[e]
        if transform is not None:
            coords = transform.apply(coords)
        return coords


def load_ligand(path: str, name: str = None, keep_origin: bool = False) -> Ligand:
    """读取配体文件(SDF/MOL2/PDB), 必要时生成 3D 构象并分型。

    :param path: 配体文件路径
    :param keep_origin: True 保留文件原始坐标(数据集 native 用)
    :return: Ligand 对象
    """
    lower = path.lower()
    if lower.endswith((".sdf", ".sd", ".mol")):
        supplier = Chem.SDMolSupplier(path, removeHs=True, sanitize=True)
        mol = next((m for m in supplier if m is not None), None)
    elif lower.endswith(".mol2"):
        mol = Chem.MolFromMol2File(path, removeHs=True, sanitize=True)
    elif lower.endswith((".pdb", ".ent")):
        mol = Chem.MolFromPDBFile(path, removeHs=True, sanitize=True,
                                  proximityBonding=True)
    else:
        raise ValueError("不支持的配体文件格式: {}".format(path))
    if mol is None:
        raise ValueError("配体解析失败(请检查分子结构有效性): {}".format(path))
    return build_ligand(mol, name=name or path, keep_origin=keep_origin)


def build_ligand(mol, name: str = "ligand", keep_origin: bool = False) -> Ligand:
    """从 RDKit Mol 构造 Ligand(自动补 3D、分型、找可旋转键)。

    :param keep_origin: True 保留原始晶格坐标(数据集构建用);
                        False 平移到质心(对接搜索用, 默认)
    """
    mol = Chem.Mol(mol)
    if any(a.GetSymbol() == "H" for a in mol.GetAtoms()):
        mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0 or mol.GetNumAtoms() < 1:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = ETKDG_SEED
        AllChem.EmbedMolecule(mol, params)
        if mol.GetNumConformers() == 0:
            raise ValueError("无法为配体生成 3D 构象: {}".format(name))
        mol = Chem.RemoveHs(mol)

    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    elements = [a.GetSymbol().upper() for a in mol.GetAtoms()]
    types = [_assign_ligand_type(mol, i) for i in range(mol.GetNumAtoms())]
    rotatable = _find_rotatable_bonds(mol)
    if not keep_origin:
        coords = coords - coords.mean(axis=0)
    lig = Ligand(coords, elements, types, rotatable,
                 mol=mol, name=name)
    _refresh_torsions(lig)
    return lig


def _assign_ligand_type(mol, idx: int) -> str:
    """配体重原子分型(元素 + 杂化 + 邻居启发式)。"""
    atom = mol.GetAtomWithIdx(idx)
    el = normalize_element(atom.GetSymbol())
    if el in ("MN", "FE", "ZN", "CU", "MG", "CA", "NA", "K"):
        return "MET"
    if el == "C":
        return "C_A" if atom.GetIsAromatic() else "C_H"
    if el == "N":
        if atom.GetIsAromatic():
            return "NA"
        has_h = atom.GetTotalNumHs() > 0
        neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
        amide = "C" in neighbors and any(n in ("O", "S") for n in neighbors)
        if has_h:
            return "N_D"
        return "N" if amide else "NA"
    if el == "O":
        return "OA"
    if el == "S":
        return "SA"
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


def _find_rotatable_bonds(mol):
    """检测非终端可旋转单键(排除酰胺与环内键), 返回可旋转键列表。

    每个元素为 (begin, end, movable_ids): movable_ids 为旋转该键时
    需要随之运动的原子集合。
    """
    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(pattern)
    rotatable = []
    for b, e in matches:
        bond = mol.GetBondBetweenAtoms(b, e)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        # 排除与氢相连及金属键
        if mol.GetAtomWithIdx(b).GetSymbol() == "H" or mol.GetAtomWithIdx(e).GetSymbol() == "H":
            continue
        movable = _movable_side(mol, b, e)
        if movable and len(movable) < mol.GetNumAtoms() - 1:
            rotatable.append((b, e, movable))
    # 数量封顶(过长链分子保护)
    return rotatable[:16]


def _movable_side(mol, b, e):
    """把键 (b,e) 断开后, e 一侧的原子集合。"""
    n = mol.GetNumAtoms()
    adjacency = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (i, j) == (b, e) or (j, i) == (b, e):
            continue
        adjacency[i].append(j)
        adjacency[j].append(i)
    seen = {e}
    stack = [e]
    while stack:
        cur = stack.pop()
        for nxt in adjacency[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    movable = sorted(seen - {e})
    return [e] + movable if movable else []


def _refresh_torsions(lig: Ligand) -> None:
    """按当前坐标刷新扭转角基准。"""
    lig.torsions = np.zeros(lig.n_torsions)


def write_sdf(lig: Ligand, coords: np.ndarray, path: str, score: float = None,
              name: str = None) -> None:
    """把带坐标的配体写为 SDF(可用 VMD/Chimera/3Dmol.js 直接打开)。"""
    mol = lig._mol
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3DWrap(x, y, z))
    writer = Chem.SDWriter(path)
    if name or score is not None:
        mol.SetProp("_Name", name or lig.name)
    if score is not None:
        mol.SetProp("ProteinDock_Score", "{:.3f}".format(score))
    writer.write(mol)
    writer.close()


def Point3DWrap(x, y, z):
    """构造 RDKit Point3D(独立包装便于测试 mock)。"""
    from rdkit.Geometry import Point3D
    return Point3D(float(x), float(y), float(z))
