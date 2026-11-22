# -*- coding: utf-8 -*-
"""Pose -> 图数据特征构造

 把「配体 pose + 蛋白口袋原子」编码为异质图:
   节点 = 配体重原子 ∪ 口袋重原子
   节点特征 = 原子类型 one-hot(16) + 是否配体标志
   边 = 距离截断内的原子对(含配体-蛋白跨边)
   边特征 = 高斯距离展开(16 通道)

 输出为轻量自描述 dict(tensor), 不依赖 torch_geometric。
"""

import numpy as np
import torch

from ..chemistry.atom_types import ATOM_TYPES

N_FEATURE_TYPES = len(ATOM_TYPES)
EDGE_GAUSS_CENTERS = np.linspace(0.5, 9.5, 16)     # 高斯展开中心 / A
EDGE_GAUSS_WIDTH = 0.6
POSE_EDGE_CUTOFF = 10.0
MAX_POCKET_ATOMS = 260                              # 口袋原子采样上限


def gaussian_edge_features(dist: np.ndarray) -> np.ndarray:
    """距离 -> (K, 16) 高斯展开特征。"""
    d = np.atleast_1d(np.asarray(dist, dtype=np.float64))
    feats = np.exp(-((d[:, None] - EDGE_GAUSS_CENTERS[None, :])
                     / EDGE_GAUSS_WIDTH) ** 2)
    return feats.astype(np.float32)


def pocket_atoms(protein, center, radius=10.0, max_atoms=MAX_POCKET_ATOMS):
    """提取口袋范围内的蛋白原子(坐标/类型), 超出上限时随机采样。"""
    mask = protein.pocket_atoms_mask(center, radius)
    idx = np.nonzero(mask)[0]
    if len(idx) > max_atoms:
        rng = np.random.default_rng(7)
        idx = rng.choice(idx, size=max_atoms, replace=False)
        idx.sort()
    return protein.coords[idx], protein.type_ids[idx], idx


def pose_to_graph(ligand, lig_coords, protein, center, device="cpu"):
    """构造 EGNN 输入图。

    :param ligand: Ligand 对象
    :param lig_coords: (N,3) pose 坐标
    :param protein: Protein 对象
    :param center: 口袋中心(决定口袋原子集合)
    :return: dict(x=(V,3) 坐标, h=(V,F) 节点特征, edge_index=(2,E),
                  edge_attr=(E,16), ligand_mask=(V,)), 无有效口袋时 None
    """
    lig_coords = np.asarray(lig_coords, dtype=np.float64)
    rec_coords, rec_ids, _ = pocket_atoms(protein, center)
    if len(rec_coords) == 0:
        return None

    n_lig = ligand.n_atoms
    n_rec = len(rec_coords)
    coords = np.concatenate([lig_coords, rec_coords], axis=0)
    type_ids = np.concatenate([ligand.type_ids, rec_ids])
    is_ligand = np.concatenate([np.ones(n_lig), np.zeros(n_rec)])

    # 节点特征: 类型 one-hot + 配体标志
    onehot = np.eye(N_FEATURE_TYPES, dtype=np.float32)[type_ids]
    h = np.concatenate([onehot, is_ligand[:, None].astype(np.float32)], axis=1)

    # 距离截断边
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    src, dst = np.nonzero((d2 < POSE_EDGE_CUTOFF ** 2) & (d2 > 1e-6))
    edge_attr = gaussian_edge_features(np.sqrt(d2[src, dst]))

    graph = {
        "x": torch.tensor(coords, dtype=torch.float32),
        "h": torch.tensor(h, dtype=torch.float32),
        "edge_index": torch.tensor(np.stack([src, dst]), dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "ligand_mask": torch.tensor(is_ligand.astype(bool)),
        "n_lig": n_lig, "n_rec": n_rec,
    }
    if device != "cpu":
        graph = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in graph.items()}
    return graph
