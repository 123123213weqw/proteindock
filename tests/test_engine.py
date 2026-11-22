# -*- coding: utf-8 -*-
"""核心引擎单元测试: 化学分型 / 几何 / 打分 / 搜索 / 配体构象

 运行: python -m pytest tests/ -v  (或 python tests/test_engine.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteindock.chemistry.geometry import (
    quaternion_identity, quaternion_rotate, random_quaternion, kabsch_rmsd)
from proteindock.chemistry.atom_types import (
    ATOM_TYPES, TYPE_INDEX, assign_protein_type, pair_coefficients)
from proteindock.io.protein import parse_pdb, detect_pocket
from proteindock.io.ligand import build_ligand
from proteindock.scoring.vina_sf import intermolecular_energy
from proteindock.scoring.grid import ScoreGrid

MINI_PROTEIN = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.200   0.500   0.300  1.00  0.00           C
ATOM      3  C   ALA A   1       2.100  -0.600   0.100  1.00  0.00           C
ATOM      4  O   ALA A   1       3.300  -0.400   0.200  1.00  0.00           O
ATOM      5  N   GLY A   2       1.700  -1.800  -0.300  1.00  0.00           N
ATOM      6  CA  GLY A   2       2.400  -3.000  -0.500  1.00  0.00           C
ATOM      7  C   GLY A   2       3.500  -3.100   0.500  1.00  0.00           C
ATOM      8  O   GLY A   2       4.600  -2.600   0.400  1.00  0.00           O
ATOM      9  CB  ALA A   1       1.100   1.200   1.500  1.00  0.00           C
ATOM     10  CG  PHE B   3       5.000   0.000   0.000  1.00  0.00           C
ATOM     11  CD1 PHE B   3       6.000   0.900   0.000  1.00  0.00           C
ATOM     12  ND1 HIS B   4       7.500   1.200   0.500  1.00  0.00           N
ATOM     13  NZ  LYS B   5       9.000  -0.500   1.000  1.00  0.00           N
ATOM     14  SD  MET B   6      10.500   0.800   2.000  1.00  0.00           S
END
"""

ASPIRIN_MOLBLOCK = """\
aspirin
     RDKit          2D

 13 13  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.9000    1.2000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.3000    1.2000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.8000    2.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.2000    2.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.7000    3.6000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.1000    3.6000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.6000    4.8000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    8.6000    5.8000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.6000    5.3000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    8.8000    7.2000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.3000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  3  4  1  0
  4  5  2  0
  5  6  1  0
  6  7  2  0
  7  8  1  0
  8  9  1  0
  9 10  1  0
 10 11  2  0
 10 12  1  0
  4 13  1  0
  6 13  1  0
M  END
$$$$
"""


def test_geometry_invariants():
    rng = np.random.default_rng(0)
    v = np.array([1.0, 2.0, 3.0])
    q = random_quaternion(rng)
    rotated = quaternion_rotate(q, v)
    assert np.isclose(np.linalg.norm(rotated), np.linalg.norm(v))
    # 单位四元数保持不动
    assert np.allclose(quaternion_rotate(quaternion_identity(), v), v)
    # Kabsch: 自身 RMSD 为 0
    pts = rng.normal(size=(10, 3))
    assert kabsch_rmsd(pts, pts) < 1e-8


def test_protein_typing():
    assert assign_protein_type("C", "CG", "PHE") == "C_A"
    assert assign_protein_type("C", "CB", "ALA") == "C_H"
    assert assign_protein_type("N", "NZ", "LYS") == "N_D"
    assert assign_protein_type("N", "ND1", "HIS") == "NA"
    assert assign_protein_type("O", "O", "GLU") == "OA"
    assert assign_protein_type("S", "SD", "MET") == "S"
    assert len(ATOM_TYPES) == 16
    assert all(len(pair_coefficients(a, b)) == 5
               for a in ATOM_TYPES for b in ATOM_TYPES)


def test_parse_mini_protein(tmp_path=None):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        fh.write(MINI_PROTEIN)
        path = fh.name
    protein = parse_pdb(path)
    assert protein.n_atoms == 14
    assert "C_A" in protein.types          # PHE CG
    assert "N_D" in protein.types          # LYS NZ
    center = detect_pocket(protein, spacing=1.5)
    assert center.shape == (3,)


def test_scoring_energies():
    lig_ids = np.array([TYPE_INDEX["C_H"], TYPE_INDEX["OA"]])
    rec_ids = np.array([TYPE_INDEX["C_H"], TYPE_INDEX["N_D"]])
    # 确定性距离场景: 3.0 A(接触), 1.0 A(穿透), 12 A(无相互作用)
    d = np.array([[3.0, 3.0],
                  [1.0, 1.0]])
    total, terms = intermolecular_energy(lig_ids, rec_ids, d)
    assert terms["gauss2"] < 0.0        # 近距高斯吸引
    assert terms["repulsion"] > 0.0     # 1.0 A 穿透产生斥力
    # 远距离应为零相互作用
    d_far = d + 100.0
    total_far, _ = intermolecular_energy(lig_ids, rec_ids, d_far)
    assert abs(total_far) < 1e-9
    # 疏水-疏水近距离接触应带来疏水吸引
    d_hydro = np.array([[3.0]])
    _, t_hydro = intermolecular_energy(np.array([TYPE_INDEX["C_H"]]),
                                       np.array([TYPE_INDEX["C_H"]]), d_hydro)
    assert t_hydro["gauss2"] < 0.0


def test_grid_consistency():
    """网格插值能量与直接 pairwise 计算应接近(同一粗化水平)。"""
    rng = np.random.default_rng(2)
    rec = rng.uniform(-4, 4, size=(25, 3))
    rec_ids = rng.integers(0, 16, size=25)
    grid = ScoreGrid(np.zeros(3), size=18).build(rec, rec_ids)
    lig = rng.uniform(-3, 3, size=(4, 3))
    lig_ids = np.array([TYPE_INDEX["C_H"], TYPE_INDEX["OA"],
                        TYPE_INDEX["N_D"], TYPE_INDEX["C_A"]])
    grid_e = grid.score(lig, lig_ids)
    d = np.sqrt(((lig[:, None, :] - rec[None, :, :]) ** 2).sum(axis=2))
    direct, _ = intermolecular_energy(lig_ids, rec_ids, d)
    # 粗化允许较大相对误差, 但量级应一致
    assert abs(grid_e - direct) < max(2.0, 0.35 * abs(direct))


def test_ligand_conformer():
    from rdkit import Chem
    mol = Chem.MolFromMolBlock(ASPIRIN_MOLBLOCK, removeHs=True)
    ligand = build_ligand(mol, name="aspirin")
    assert ligand.n_atoms == 13
    assert ligand.n_torsions >= 0
    # 零扭转时 conformer 应还原原坐标
    coords = ligand.conformer()
    assert np.allclose(coords, ligand.coords, atol=1e-6)


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS  {}".format(fn.__name__))
        except Exception as exc:
            failed += 1
            print("FAIL  {}: {}".format(fn.__name__, exc))
    print("\n{} passed, {} failed".format(len(tests) - failed, failed))
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())
