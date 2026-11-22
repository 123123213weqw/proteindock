# -*- coding: utf-8 -*-
"""E(n) 等变图神经网络打分模型(EGNN)

 参照 Satorras et al., "E(n) Equivariant Graph Neural Networks", 2021
 实现轻量版: 4 层等变消息传递(坐标仅在内部传播, 不回写输入),
 读出层取「配体节点池化 + 全局均值池化」拼接, 经 MLP 输出
 以 kcal/mol 为单位的结合能标量。

 等变性保证: 输入 pose 整体旋转/平移时, 网络输出严格不变,
 这正是分子打分函数应满足的物理对称性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_FEATURE_TYPES, POSE_EDGE_CUTOFF


class EGNNLayer(nn.Module):
    """单层等变消息传递。"""

    def __init__(self, hidden_dim, edge_dim=16):
        super().__init__()
        self.edge_dim = edge_dim
        in_dim = hidden_dim * 2 + edge_dim + 1     # h_i, h_j, e_ij, d^2
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1), nn.Tanh(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_h = nn.LayerNorm(hidden_dim)

    def forward(self, h, x, edge_index, edge_attr):
        src, dst = edge_index
        d2 = ((x[src] - x[dst]) ** 2).sum(dim=-1, keepdim=True)
        msg_in = torch.cat([h[src], h[dst], edge_attr, d2], dim=-1)
        m = self.edge_mlp(msg_in)
        # 坐标等变更新(仅内部使用)
        weight = self.coord_mlp(m)
        coord_delta = (x[src] - x[dst]) * weight
        x_new = x.clone()
        x_new = x_new.index_add(0, dst, coord_delta)
        # 节点特征更新
        agg = torch.zeros_like(h).index_add(0, dst, m)
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm_h(h_new), x_new


class EGNNPoseScorer(nn.Module):
    """EGNN pose/affinity 打分模型。

    :param n_types: 原子类型数(节点 one-hot 维度)
    :param hidden_dim: 隐藏层维度
    :param n_layers: 等变层数
    """

    def __init__(self, n_types=N_FEATURE_TYPES, hidden_dim=96, n_layers=4):
        super().__init__()
        self.n_types = n_types
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(n_types + 1, hidden_dim)   # +1 配体标志
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph):
        """graph 为 features.pose_to_graph 输出 dict(可含 batch_ptr 批拼)。"""
        h = self.embed(graph["h"])
        x = graph["x"]
        edge_index = graph["edge_index"]
        edge_attr = graph["edge_attr"]
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        ptr = graph.get("batch_ptr")
        if ptr is None:
            pooled = self._pool(h, graph["ligand_mask"])
            return self.readout(pooled).view(-1)
        # 批拼模式: 按段池化
        outputs = []
        start = 0
        for size in ptr:
            size = int(size)
            seg_h = h[start:start + size]
            seg_mask = graph["ligand_mask"][start:start + size]
            outputs.append(self.readout(self._pool(seg_h, seg_mask)))
            start += size
        return torch.cat([o.view(1) for o in outputs])

    @staticmethod
    def _pool(h, ligand_mask):
        """配体池化 + 全局池化拼接。"""
        lig = ligand_mask.float().unsqueeze(-1)
        n_lig = torch.clamp(lig.sum(), min=1.0)
        pooled_lig = (h * lig).sum(dim=0) / n_lig
        pooled_all = h.mean(dim=0)
        return torch.cat([pooled_lig, pooled_all])

    @classmethod
    def load_pretrained(cls, path, device="cpu", **kwargs):
        """加载训练好的权重(不存在时返回 None)。"""
        import os
        if not path or not os.path.exists(path):
            return None
        model = cls(**kwargs)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        model.to(device)
        model.eval()
        return model
