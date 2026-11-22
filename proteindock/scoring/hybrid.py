# -*- coding: utf-8 -*-
"""混合打分器: 物理经验打分 + 深度学习 GNN 打分

 对接搜索阶段使用网格加速的 Vina 式经验打分快速评估;
 对搜索产生的候选 pose, 再用 E(n) 等变图神经网络(EGNN)重打分,
 两者加权融合作为最终排序依据 —— 即「物理先验 + 数据驱动」的
 混合(hybrid)策略。
"""

import numpy as np

from .grid import ScoreGrid
from .vina_sf import intramolecular_energy


class HybridScorer:
    """经验 + 深度学习双通道打分器。

    :param grid: 已构建的 ScoreGrid
    :param dl_model: 已加载权重的 EGNN 打分模型(可为 None, 退化为纯经验)
    :param dl_weight: DL 分数融合权重 (final = vina + dl_weight * dl)
    :param device: torch 设备
    """

    def __init__(self, grid: ScoreGrid, dl_model=None, protein=None,
                 center=None, dl_weight=1.0, device="cpu"):
        self.grid = grid
        self.dl_model = dl_model
        self.protein = protein          # DL 通道需要蛋白对象构图
        self.center = None if center is None else np.asarray(center, float)
        self.dl_weight = float(dl_weight)
        self.device = device

    def vina_score(self, ligand, coords) -> dict:
        """经验打分通道: 网格相互作用 + 内部能 + 扭转熵惩罚。"""
        inter = self.grid.score(coords, ligand.type_ids)
        intra = intramolecular_energy(ligand.type_ids, coords,
                                      ligand.n_torsions)
        return {
            "inter": float(inter),
            "intra": float(intra),
            "total": float(inter + intra),
        }

    def dl_score(self, ligand, coords) -> float:
        """深度学习通道: EGNN 对整个 pose 输出结合能(kcal/mol)。"""
        if self.dl_model is None or self.protein is None or self.center is None:
            return 0.0
        from ..dl.features import pose_to_graph   # 延迟导入避免循环依赖
        graph = pose_to_graph(ligand, coords, self.protein, self.center)
        if graph is None:
            return 0.0
        import torch
        self.dl_model.eval()
        with torch.no_grad():
            graph = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                     for k, v in graph.items()}
            pred = self.dl_model(graph).view(-1)
            return float(pred[0].item())

    def score(self, ligand, coords) -> dict:
        """综合打分入口。

        :return: dict(vina_inter, vina_intra, vina, dl, final, terms...)
        """
        vina = self.vina_score(ligand, coords)
        result = {
            "vina_inter": vina["inter"],
            "vina_intra": vina["intra"],
            "vina": vina["total"],
        }
        if self.dl_model is not None:
            dl = self.dl_score(ligand, coords)
            result["dl"] = dl
            result["final"] = result["vina"] + self.dl_weight * dl
        else:
            result["dl"] = None
            result["final"] = result["vina"]
        return result
