# -*- coding: utf-8 -*-
"""对接数据集: 构造 / 加载 / 批拼(collate)

 数据集目录约定(dockdata 格式, 由 scripts/build_dataset.py 生成):

   <root>/<pdb_id>/protein.pdb     去配体蛋白
   <root>/<pdb_id>/ligand.sdf      晶体配体构象
   <root>/<pdb_id>/decoys.npy      (K, N, 3) 人工扰动 pose 坐标
   <root>/<pdb_id>/labels.npy      (K,) 教师标签(经验打分或实验亲和力)

 训练样本 = (pose 图, 标签)。
"""

import os
import glob

import numpy as np
import torch

from ..io.protein import parse_pdb
from ..io.ligand import load_ligand
from .features import pose_to_graph


class PoseSampleDataset(torch.utils.data.Dataset):
    """逐体系加载, 体系内每个 decoy 展开为独立样本。"""

    def __init__(self, root, cache_proteins=True):
        self.root = root
        self.systems = []
        self.index = []          # (system_i, pose_k)
        self._protein_cache = {}
        self._ligand_cache = {}
        self._cache = cache_proteins
        for sys_dir in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(sys_dir):
                continue
            need = ["protein.pdb", "ligand.sdf", "decoys.npy", "labels.npy"]
            if not all(os.path.exists(os.path.join(sys_dir, f)) for f in need):
                continue
            sys_id = os.path.basename(sys_dir)
            sys_idx = len(self.systems)
            self.systems.append(sys_dir)
            labels = np.load(os.path.join(sys_dir, "labels.npy"))
            for k in range(len(labels)):
                self.index.append((sys_idx, k))
            self._protein_cache[sys_idx] = None    # 延迟填充
            self._ligand_cache[sys_idx] = None

    def __len__(self):
        return len(self.index)

    def n_systems(self):
        return len(self.systems)

    def _load_system(self, sys_idx):
        if self._protein_cache[sys_idx] is None:
            sys_dir = self.systems[sys_idx]
            protein = parse_pdb(os.path.join(sys_dir, "protein.pdb"))
            # 原位加载 native 配体: 质心即晶体口袋中心
            ligand = load_ligand(os.path.join(sys_dir, "ligand.sdf"),
                                 keep_origin=True)
            pocket_center = ligand.coords.mean(axis=0)
            self._protein_cache[sys_idx] = (protein, ligand, pocket_center)
        return self._protein_cache[sys_idx]

    def __getitem__(self, i):
        sys_idx, pose_k = self.index[i]
        protein, ligand, center = self._load_system(sys_idx)
        sys_dir = self.systems[sys_idx]
        decoys = np.load(os.path.join(sys_dir, "decoys.npy"), mmap_mode="r")
        labels = np.load(os.path.join(sys_dir, "labels.npy"))
        coords = np.array(decoys[pose_k], dtype=np.float64)
        graph = pose_to_graph(ligand, coords, protein, center)
        return graph, float(labels[pose_k])


def collate_graphs(samples):
    """把多个 pose 图批拼为一个大图(edge_index 加偏移)。"""
    graphs, labels = zip(*samples)
    xs, hs, srcs, dsts, eas, masks = [], [], [], [], [], []
    offset = 0
    for g in graphs:
        if g is None:
            continue
        xs.append(g["x"])
        hs.append(g["h"])
        edge_index = g["edge_index"]
        srcs.append(edge_index[0] + offset)
        dsts.append(edge_index[1] + offset)
        eas.append(g["edge_attr"])
        masks.append(g["ligand_mask"])
        offset += g["x"].shape[0]
    if not xs:
        return None, torch.tensor(labels, dtype=torch.float32)
    batch = {
        "x": torch.cat(xs, dim=0),
        "h": torch.cat(hs, dim=0),
        "edge_index": torch.stack([torch.cat(srcs), torch.cat(dsts)]),
        "edge_attr": torch.cat(eas, dim=0),
        "ligand_mask": torch.cat(masks),
        "batch_ptr": torch.tensor([g["x"].shape[0] for g in graphs
                                   if g is not None]),
    }
    return batch, torch.tensor(labels, dtype=torch.float32)
