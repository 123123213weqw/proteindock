# -*- coding: utf-8 -*-
"""构建 EGNN 训练数据集(dockdata 格式)

 流程:
   1. 从 RCSB 下载一批经典蛋白-配体复合物(PDB)
   2. 拆分: 去配体蛋白 protein.pdb + 晶体配体 ligand.sdf
   3. 以晶体配体为 native, 刚体/扭转随机扰动生成 K 个 decoy pose
   4. 用 Vina 式经验打分函数为每个 pose 打教师标签(知识蒸馏)
   5. 落盘为 dockdata 格式供 proteindock.dl.dataset 加载

 用法: python scripts/build_dataset.py --out datasets/dockdata --n-decoys 24
"""

import argparse
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteindock.io.protein import parse_pdb, Protein
from proteindock.io.ligand import build_ligand
from proteindock.scoring.vina_sf import intermolecular_energy
from proteindock.search.mc_sa import Pose
from proteindock.chemistry.geometry import random_quaternion

# 经典药物靶标复合物(PDB ID), 含亲和力数据丰富的主流体系
PDB_IDS = [
    "1HVR",   # HIV-1 蛋白酶 / ritonavir
    "1DWD",   # 凝血酶 / D-Phe-Pro-Arg
    "1BTY",   # 胰蛋白酶 / benzamidine
    "1AKZ",   # HIV 蛋白酶 / 抑制剂
    "3ERT",   # 雌激素受体 / raloxifene
    "2BST",   # β-分泌酶
    "1OWH",   # CDK2
    "1M17",   # EGFR / erlotinib
    "1UWH",   # p38 MAP 激酶
    "3GCS",   # 激酶抑制剂
    "1SQN",   # 碳酸酐酶
    "2Z5U",   # 磷酸二酯酶
    "1K1J",   # 凝血因子 Xa
    "1R9O",   # 神经氨酸酶 / 奥司他韦
    "3CL0",   # SARS-CoV-2 主蛋白酶样体系
]

RCSB_URL = "https://files.rcsb.org/download/{}.pdb"


def download_pdb(pdb_id: str, dest: str) -> bool:
    """从 RCSB 下载 PDB 文件。"""
    url = RCSB_URL.format(pdb_id.upper())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ProteinDock/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as exc:
        print("下载失败 {}: {}".format(pdb_id, exc))
        return False
    with open(dest, "wb") as fh:
        fh.write(data)
    return True


def split_complex(pdb_path: str, min_lig_atoms: int = 5):
    """把复合物 PDB 拆为蛋白文本与晶体配体 HETATM 块。

    选择规则: 收集全部非水/非辅基 HETATM 残基,
    取原子数不少于 min_lig_atoms 的第一个(通常是主配体)。
    """
    protein_lines, het_by_res = [], {}
    skip = {"HOH", "WAT", "SO4", "GOL", "EDO", "NAG", "MAN", "CL",
            "NA", "MG", "ZN", "CA", "K", "ACT", "MES", "TRS", "PEG",
            "DMS", "EOH", "IPA", "FMT", "ACE"}
    with open(pdb_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "ATOM":
                protein_lines.append(line)
            elif rec == "HETATM":
                res = line[17:20].strip().upper()
                if res in skip:
                    continue
                het_by_res.setdefault(res, []).append(line)
    lig_res, lig_lines = None, []
    for res, lines in het_by_res.items():
        if len(lines) >= min_lig_atoms:
            lig_res, lig_lines = res, lines
            break
    if lig_res is None and het_by_res:      # 回退: 原子最多的 HETATM
        lig_res = max(het_by_res, key=lambda r: len(het_by_res[r]))
        lig_lines = het_by_res[lig_res]
    lig_pdb = "".join(lig_lines) + "END\n"
    return "".join(protein_lines) + "TER\nEND\n", lig_pdb, lig_res


def write_tmp(text: str, path: str):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def build_systems(out_root: str, n_decoys: int, max_atoms=70):
    """逐体系构造 dockdata 目录。"""
    from rdkit import Chem
    os.makedirs(out_root, exist_ok=True)
    tmp_dir = os.path.join(out_root, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    built = 0
    for pdb_id in PDB_IDS:
        sys_dir = os.path.join(out_root, pdb_id.lower())
        if os.path.isdir(sys_dir) and os.path.exists(
                os.path.join(sys_dir, "labels.npy")):
            print("跳过已存在: {}".format(sys_dir))
            built += 1
            continue
        raw = os.path.join(tmp_dir, pdb_id + ".pdb")
        if not os.path.exists(raw) and not download_pdb(pdb_id, raw):
            continue
        try:
            protein_text, lig_pdb, lig_res = split_complex(raw)
            if not lig_pdb.strip() or lig_res is None:
                print("{} 无配体, 跳过".format(pdb_id))
                continue
            write_tmp(protein_text, os.path.join(tmp_dir, "p.pdb"))
            write_tmp(lig_pdb, os.path.join(tmp_dir, "l.pdb"))
            protein = parse_pdb(os.path.join(tmp_dir, "p.pdb"))
            lig_mol = Chem.MolFromPDBFile(os.path.join(tmp_dir, "l.pdb"),
                                          removeHs=True, sanitize=True,
                                          proximityBonding=True)
            if lig_mol is None or lig_mol.GetNumAtoms() < 5 \
                    or lig_mol.GetNumAtoms() > max_atoms:
                print("{} 配体无效/过大({})".format(
                    pdb_id, lig_mol.GetNumAtoms() if lig_mol else 0))
                continue
            # 原位配体(保留晶体坐标, native pose / 标签用)
            ligand = build_ligand(lig_mol, name=pdb_id, keep_origin=True)
            # 居中副本(刚体 Pose 采样假设质心坐标系)
            ligand_c = build_ligand(lig_mol, name=pdb_id)
            native = ligand.coords + 0.0
            center = native.mean(axis=0)

            # decoys: native 附近小扰动 + 远处随机 pose
            rng = np.random.default_rng(42)
            from proteindock.chemistry.geometry import quaternion_rotate
            lig_center = native.mean(axis=0)
            poses = [native]
            for k in range(n_decoys):
                if k < n_decoys // 3:   # near-native 扰动(绕质心小幅刚体扰动)
                    q = random_quaternion(rng)
                    moved = quaternion_rotate(q, native - lig_center) + lig_center
                    moved += rng.normal(0, 0.6, size=3)
                    poses.append(moved)
                else:                    # 远处 pose: 刚体打散 + 扭转随机
                    pose = Pose.random(ligand_c, center, 16.0, rng)
                    poses.append(pose.coords(ligand_c))

            labels = []
            for coords in poses:
                d = np.sqrt(((coords[:, None, :] - protein.coords[None, :, :]) ** 2)
                            .sum(axis=2))
                pocket = d.min(axis=1) < 12.0
                if not pocket.any():     # 远离蛋白的 pose 给平庸标签
                    labels.append(0.0)
                    continue
                total, _ = intermolecular_energy(
                    ligand_c.type_ids, protein.type_ids, d)
                labels.append(float(total))
            labels = np.array(labels, dtype=np.float32)

            os.makedirs(sys_dir, exist_ok=True)
            with open(os.path.join(sys_dir, "protein.pdb"), "w",
                      encoding="utf-8") as fh:
                fh.write(protein_text)
            from proteindock.io.ligand import write_sdf
            write_sdf(ligand, native, os.path.join(sys_dir, "ligand.sdf"),
                      name=pdb_id + "_native")
            np.save(os.path.join(sys_dir, "decoys.npy"),
                    np.array(poses, dtype=np.float32))
            np.save(os.path.join(sys_dir, "labels.npy"), labels)
            print("构建完成 {} (decoys={}, 标签范围 {:.1f}~{:.1f})".format(
                pdb_id, len(poses), labels.min(), labels.max()))
            built += 1
        except Exception as exc:
            print("{} 处理失败: {}".format(pdb_id, exc))
    print("共构建 {} 个体系 -> {}".format(built, out_root))
    return built


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/dockdata")
    ap.add_argument("--n-decoys", type=int, default=24)
    args = ap.parse_args()
    build_systems(args.out, args.n_decoys)


if __name__ == "__main__":
    main()
